"""Batched, single-update modern Hopfield retrieval of high-dimensional memories."""

import copy
import torch
from torch import nn
from torch.nn import functional as F


class ModernHopfield(nn.Module):
    """Learn an association space while preserving full memory values.

    Inputs are cues ``(..., input_size)`` and memories ``(M, input_size)``
    or ``(B, M, input_size)`` for cues ``(B, L, input_size)``. The output
    has the cue shape. Keys and queries use a shared projection by default;
    ``separate_projections=True`` gives them independent learned matrices.
    Values are the original memories, so retrieval can restore parameters.
    ``feature_size`` bounds projection cost for very large parameter vectors.
    Feature indices are deterministic and saved in the state dict.

    This implements exactly one softmax association (update_steps_max=0).
    """

    def __init__(self, input_size, association_size=64, feature_size=2048,
                 beta=20.0, update_steps_max=0, separate_projections=False):
        super().__init__()
        if min(input_size, association_size, feature_size) <= 0 or beta <= 0:
            raise ValueError("Dimensions and beta must be positive")
        if update_steps_max != 0:
            raise ValueError("ModernHopfield requires update_steps_max=0")
        self.input_size = input_size
        self.update_steps_max = 0
        self.beta = beta
        self.separate_projections = separate_projections
        size = min(feature_size, input_size)
        self.register_buffer("feature_indices", torch.linspace(
            0, input_size - 1, size).round().long())
        if separate_projections:
            self.query_encoder = nn.Linear(size, association_size, bias=False)
            # Equal starting functions make comparisons with tied weights controlled.
            # deepcopy creates independent Parameters and storage, not a weight tie.
            self.key_encoder = copy.deepcopy(self.query_encoder)
        else:
            # Keep the original parameter name for backward-compatible checkpoints.
            self.encoder = nn.Linear(size, association_size, bias=False)

    def project_queries(self, features):
        return self.query_encoder(features) if self.separate_projections else self.encoder(features)

    def project_keys(self, features):
        return self.key_encoder(features) if self.separate_projections else self.encoder(features)

    def forward(self, cues, memories, return_associations=False):
        if cues.ndim < 2 or memories.ndim not in (2, 3):
            raise ValueError("Use batched cues and shared or batched memories")
        if cues.shape[-1] != self.input_size or memories.shape[-1] != self.input_size:
            raise ValueError("Cue and memory feature dimensions must equal input_size")
        if memories.shape[-2] == 0:
            raise ValueError("At least one memory is required")
        if memories.ndim == 3 and (cues.ndim != 3 or cues.shape[0] != memories.shape[0]):
            raise ValueError("Batched memories require cues shaped (B, L, D)")
        # Standardize using only the memory bank, never evaluation examples.
        keys = memories.index_select(-1, self.feature_indices)
        queries = cues.index_select(-1, self.feature_indices)
        center = keys.mean(dim=-2, keepdim=True)
        scale = keys.std(dim=-2, keepdim=True, unbiased=False).clamp_min(1e-3)
        queries = F.normalize(self.project_queries((queries - center) / scale), dim=-1)
        keys = F.normalize(self.project_keys((keys - center) / scale), dim=-1)
        associations = (self.beta * (queries @ keys.transpose(-1, -2))).softmax(dim=-1)
        retrieved = associations @ memories
        return (retrieved, associations) if return_associations else retrieved
