"""Invertible, bounded block-diagonal value coordinates without a DxD matrix."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class InvertibleBlockDiagonal(nn.Module):
    """Each block is I+L, with L strictly lower triangular and bounded.

    |L_ij| <= 0.25/(block_size-1), so ||L||_2 <= 0.25 and the block
    condition number is at most 5/3. Diagonals are exactly one. Small
    float64 triangular solves reverse the current map, including partial
    final blocks. Parameters are trainable, although an exactly reversed
    value path cancels their gradient under the existing full-parameter loss.
    """
    def __init__(self, input_size, block_size=16, seed=0):
        super().__init__()
        if input_size < 1 or block_size < 1:
            raise ValueError("Input and block sizes must be positive")
        self.input_size, self.block_size = input_size, block_size
        self.num_blocks = math.ceil(input_size / block_size)
        rows, columns = torch.tril_indices(block_size, block_size, offset=-1)
        self.register_buffer("lower_rows", rows)
        self.register_buffer("lower_columns", columns)
        mask = torch.ones(self.num_blocks, len(rows), dtype=torch.bool)
        remainder = input_size % block_size
        if remainder:
            mask[-1] = rows < remainder
        self.register_buffer("valid_lower", mask)
        generator = torch.Generator().manual_seed(seed)
        self.raw_lower = nn.Parameter(torch.randn(self.num_blocks, len(rows),
                                                  dtype=torch.float64, generator=generator))

    def blocks(self):
        matrices = torch.eye(self.block_size, dtype=self.raw_lower.dtype,
                             device=self.raw_lower.device).expand(self.num_blocks, -1, -1).clone()
        matrices[:, self.lower_rows, self.lower_columns] = (
            self.raw_lower.tanh() * (0.25 / max(self.block_size - 1, 1)) * self.valid_lower)
        return matrices

    def _rhs(self, values):
        if values.shape[-1] != self.input_size:
            raise ValueError("Value dimension does not match the block-diagonal map")
        padded = F.pad(values.to(self.raw_lower), (0, self.num_blocks * self.block_size - self.input_size))
        return padded.reshape(-1, self.num_blocks, self.block_size).permute(1, 2, 0)

    def _restore(self, rhs, shape):
        flat = rhs.permute(2, 0, 1).reshape(-1, self.num_blocks * self.block_size)
        return flat[:, :self.input_size].reshape(shape)

    def _encode(self, values, matrices):
        return self._restore(matrices @ self._rhs(values), values.shape)

    def _decode(self, values, matrices):
        solved = torch.linalg.solve_triangular(matrices, self._rhs(values), upper=False, unitriangular=True)
        return self._restore(solved, values.shape)

    def encode(self, values):
        return self._encode(values, self.blocks())

    def decode(self, values):
        return self._decode(values, self.blocks())

    def round_trip(self, values):
        matrices = self.blocks()
        return self._decode(self._encode(values, matrices), matrices)

    def retrieve(self, associations, values):
        matrices = self.blocks()
        encoded = self._encode(values, matrices)
        mixed = associations.to(encoded) @ encoded
        return self._decode(mixed, matrices).to(values.dtype)

    @torch.no_grad()
    def diagnostics(self, values):
        error = self.round_trip(values) - values.double()
        return {"max_abs_roundtrip_error": float(error.abs().max()),
                "relative_roundtrip_error": float(error.norm() / values.double().norm().clamp_min(1e-12)),
                "condition_number_bound": 5 / 3}
