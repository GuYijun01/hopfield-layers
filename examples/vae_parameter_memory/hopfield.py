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


def make_retriever(parameter_count, feature_size=2048, association_size=64):
    return ModernHopfield(parameter_count, association_size, feature_size,
                          beta=20.0, update_steps_max=0)


def retrieve_parameters(hopfield, delta_cues, memories, initial):
    """Keys use final-minus-initial deltas; return full absolute VAE parameters.

    q and keys share a coordinate origin. Initial parameters are added back
    after the single Hopfield association, so VAE restoration uses full weights.
    """
    delta, weights = hopfield(delta_cues, memories - initial, return_associations=True)
    return delta + initial, weights
