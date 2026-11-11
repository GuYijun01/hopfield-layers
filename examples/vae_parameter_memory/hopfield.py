"""Parameter-memory association network, separate from VAE0 and VAE2."""
import torch
from torch.nn import functional as F
from hflayers.modern import ModernHopfield


def train_retriever(hopfield, cues, targets, memories, steps, lr, initial):
    """Replay all observed variants, with one fixed final memory per digit."""
    if len(memories) == 1:
        return {"loss": 0.0, "cue_accuracy": 1.0}
    optimizer = torch.optim.Adam(hopfield.parameters(), lr=lr)
    hopfield.train()
    for _ in range(steps):
        optimizer.zero_grad()
        retrieved, weights = retrieve_parameters(hopfield, cues, memories, initial)
        loss = F.nll_loss(weights.clamp_min(1e-12).log(), targets)
        loss += F.mse_loss(retrieved, memories[targets]) / memories.var(unbiased=False).clamp_min(1e-8)
        loss.backward()
        optimizer.step()
    hopfield.eval()
    with torch.no_grad():
        _, weights = retrieve_parameters(hopfield, cues, memories, initial)
        association_loss = F.nll_loss(weights.clamp_min(1e-12).log(), targets).item()
    return {"loss": association_loss,
            "cue_accuracy": (weights.argmax(-1) == targets).float().mean().item()}


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
