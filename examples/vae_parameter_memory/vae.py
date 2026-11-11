"""Fully connected MNIST VAE and parameter serialization helpers."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class VAE(nn.Module):
    """Encoder 784→85→85→(mu, logvar); decoder 5→85→85→784.

    Decoder outputs logits; use sigmoid for reconstructed pixel probabilities.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 85), nn.ReLU(),
                                     nn.Linear(85, 85), nn.ReLU())
        self.mu = nn.Linear(85, 5)
        self.logvar = nn.Linear(85, 5)
        self.decoder = nn.Sequential(nn.Linear(5, 85), nn.ReLU(),
                                     nn.Linear(85, 85), nn.ReLU(), nn.Linear(85, 784))

    def forward(self, images, sample=True, generator=None):
        hidden = self.encoder(images.reshape(-1, 784))
        mu, logvar = self.mu(hidden), self.logvar(hidden)
        if sample:
            noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
            z = mu + (0.5 * logvar).exp() * noise
        else:
            z = mu
        return self.decoder(z), mu, logvar


def vae_loss(logits, images, mu, logvar):
    """Per-image negative ELBO: summed pixel BCE plus analytic KL(q||N(0,I))."""
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, images.reshape(-1, 784), reduction="none").sum(-1)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)
    return reconstruction + kl


def parameter_vector(model):
    return parameters_to_vector(model.parameters()).detach().clone()


def restore_vector(model, parameters):
    if parameters.ndim != 1 or parameters.numel() != sum(p.numel() for p in model.parameters()):
        raise ValueError("Parameter vector does not match this VAE")
    vector_to_parameters(parameters.detach().clone(), model.parameters())


def initial_vector(initial_state):
    """Flatten this VAE state dict (which contains parameters only) in order."""
    return torch.cat([value.detach().reshape(-1) for value in initial_state.values()])
