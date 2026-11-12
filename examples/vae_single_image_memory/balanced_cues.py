"""Five equally sized transform families and a fresh shuffle before each update."""
import math

import torch
from torch.nn import functional as F

from .cues import fit_block


PROFILE = "balanced80_resampled_v1"
SAMPLING = "fresh_permutation_first16_each_update_v1"


def augmentation_config():
    return {"rotation_degrees": 10.0, "scale_min": 0.9, "scale_max": 1.1,
            "rotations": 16, "horizontal": 16, "vertical": 16, "uniform": 16,
            "translations": 16, "translation_offsets_pixels": [-2, -1, 1, 2],
            "shuffle": True, "sampling": SAMPLING, "updates": 5, "batch_size": 16}


def affine_transforms(device="cpu"):
    """Return 80 output-to-input transforms; translations cover a 4x4 offset grid."""
    transforms = torch.zeros(80, 2, 3, device=device)
    transforms[:, 0, 0] = transforms[:, 1, 1] = 1
    angles = torch.linspace(-math.pi / 18, math.pi / 18, 16, device=device)
    transforms[:16, 0, 0] = transforms[:16, 1, 1] = angles.cos()
    transforms[:16, 0, 1] = -angles.sin()
    transforms[:16, 1, 0] = angles.sin()
    scales = torch.linspace(0.9, 1.1, 16, device=device).reciprocal()
    transforms[16:32, 0, 0] = scales
    transforms[32:48, 1, 1] = scales
    transforms[48:64, 0, 0] = transforms[48:64, 1, 1] = scales
    offsets = torch.tensor([-2., -1., 1., 2.], device=device)
    translations = torch.cartesian_prod(offsets, offsets)
    # align_corners=False: one pixel corresponds to 2/28 normalized units.
    # Negative inverse-map offsets move the image in the positive direction.
    transforms[64:, :, 2] = -translations * (2 / 28)
    return transforms


def augment_image(image):
    """Build an unshuffled 80-view pool: rotation, x/y scale, uniform scale, shift."""
    image = torch.as_tensor(image, dtype=torch.float32)
    if image.numel() != 784:
        raise ValueError("Expected one 28x28 image")
    images = image.reshape(1, 1, 28, 28).expand(80, -1, -1, -1)
    grid = F.affine_grid(affine_transforms(image.device), images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False).reshape(80, 784)


def update_indices(seed):
    """Five independent permutations; each update takes its first 16 views.

    No repetition within a batch. Repetition across batches is allowed, and
    five updates do not guarantee visiting every member of the 80-view pool.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(80, generator=generator)[:16] for _ in range(5)])


def fit_image_cue(initial_state, image, lr, device, seed):
    views = augment_image(image).to(device)
    batches = (views.index_select(0, indices.to(device)) for indices in update_indices(seed))
    return fit_block(initial_state, batches, lr, device, seed)
