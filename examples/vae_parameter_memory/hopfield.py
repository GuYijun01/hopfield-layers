"""Parameter-memory association network, separate from VAE0 and VAE2."""
import torch
from torch.nn import functional as F
from hflayers.modern import ModernHopfield


@torch.no_grad()
def retriever_metrics(hopfield, cues, targets, memories, initial):
    """Evaluate both terms of the actual objective without updating any weights."""
    retrieved, weights = retrieve_parameters(hopfield, cues, memories, initial)
    association = F.nll_loss(weights.clamp_min(1e-12).log(), targets)
    reconstruction = F.mse_loss(retrieved, memories[targets]) / memories.var(unbiased=False).clamp_min(1e-8)
    return {"total_loss": (association + reconstruction).item(),
            "association_loss": association.item(),
            "normalized_parameter_mse": reconstruction.item(),
            "cue_accuracy": (weights.argmax(-1) == targets).float().mean().item()}


def train_retriever(hopfield, cues, targets, memories, steps, lr, initial, *,
                    validation_cues=None, validation_targets=None, history_every=0):
    """Replay all observed variants, with one fixed final memory per digit."""
    if (validation_cues is None) != (validation_targets is None):
        raise ValueError("Provide both validation cues and targets")
    if history_every < 0:
        raise ValueError("history_every must be nonnegative")
    history = []

    def record(step):
        row = {"step": step, "train": retriever_metrics(hopfield, cues, targets, memories, initial)}
        if validation_cues is not None:
            row["validation"] = retriever_metrics(
                hopfield, validation_cues, validation_targets, memories, initial)
        history.append(row)

    if history_every:
        record(0)
    if len(memories) == 1:
        if history_every:
            record(steps)
        return {"loss": 0.0, "cue_accuracy": 1.0, **({"history": history} if history_every else {})}
    optimizer = torch.optim.Adam(hopfield.parameters(), lr=lr)
    hopfield.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        retrieved, weights = retrieve_parameters(hopfield, cues, memories, initial)
        loss = F.nll_loss(weights.clamp_min(1e-12).log(), targets)
        loss += F.mse_loss(retrieved, memories[targets]) / memories.var(unbiased=False).clamp_min(1e-8)
        loss.backward()
        optimizer.step()
        if history_every and (step % history_every == 0 or step == steps):
            record(step)
    hopfield.eval()
    with torch.no_grad():
        _, weights = retrieve_parameters(hopfield, cues, memories, initial)
        association_loss = F.nll_loss(weights.clamp_min(1e-12).log(), targets).item()
    return {"loss": association_loss,
            "cue_accuracy": (weights.argmax(-1) == targets).float().mean().item(),
            **({"history": history} if history_every else {})}


def make_retriever(parameter_count, feature_size=2048, association_size=64, *, projection_mode="shared",
                    value_projection="identity", value_block_size=16, value_seed=0):
    if projection_mode not in ("shared", "separate"):
        raise ValueError("projection_mode must be 'shared' or 'separate'")
    return ModernHopfield(parameter_count, association_size, feature_size,
                          beta=20.0, update_steps_max=0,
                          separate_projections=projection_mode == "separate",
                          value_projection=value_projection, value_block_size=value_block_size, value_seed=value_seed)


def checkpoint_projection_mode(checkpoint):
    """Old checkpoints have encoder.weight and default to the shared architecture."""
    configured = checkpoint.get("config", {}).get("projection_mode")
    mode = checkpoint.get("projection_mode", configured or "shared")
    if mode not in ("shared", "separate") or (configured is not None and configured != mode):
        raise ValueError("Unknown or inconsistent checkpoint projection_mode")
    return mode


def checkpoint_value_config(checkpoint):
    config = checkpoint.get("config", {})
    configured = config.get("value_projection")
    mode = checkpoint.get("value_projection", configured or "identity")
    size = checkpoint.get("value_block_size", config.get("value_block_size", 16))
    seed = checkpoint.get("value_seed", config.get("value_seed", config.get("seed", 0)))
    if mode not in ("identity", "block_diagonal") or (configured is not None and configured != mode):
        raise ValueError("Unknown or inconsistent checkpoint value_projection")
    if not isinstance(size, int) or size < 1:
        raise ValueError("Value block size must be a positive integer")
    return {"value_projection": mode, "value_block_size": size, "value_seed": seed}


def retriever_from_checkpoint(checkpoint, device):
    config = checkpoint["config"]
    model = make_retriever(checkpoint["parameter_count"], config["feature_size"],
                           config["association_size"],
                           projection_mode=checkpoint_projection_mode(checkpoint),
                           **checkpoint_value_config(checkpoint)).to(device)
    model.load_state_dict(checkpoint["hopfield"])
    return model


def retrieve_parameters(hopfield, delta_cues, memories, initial):
    """Keys use final-minus-initial deltas; return full absolute VAE parameters.

    Cues and keys share a coordinate origin. Identity values retrieve deltas
    and add initial parameters back. Block values encode full expert weights
    and reverse the transform after the single association.
    """
    if hopfield.value_projection is not None:
        # Project full expert parameters as values; only keys use the delta origin.
        return hopfield(delta_cues, memories - initial, return_associations=True, values=memories)
    delta, weights = hopfield(delta_cues, memories - initial, return_associations=True)
    return delta + initial, weights
