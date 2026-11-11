"""Versioned 40-rotation/40-axis-scale augmentations of one MNIST image."""
import math
import torch
from torch.nn import functional as F


def augmentation_config(profile="legacy"):
    """Return the geometry and order saved in each checkpoint/report."""
    if profile == "legacy":
        return {"rotation_degrees": 45.0, "scale_min": 1.05, "scale_max": 1.5,
                "shuffle": False, "rotations": 40, "horizontal": 20, "vertical": 20}
    if profile == "mild":
        return {"rotation_degrees": 15.0, "scale_min": 0.9, "scale_max": 1.1,
                "shuffle": True, "rotations": 40, "horizontal": 20, "vertical": 20}
    raise ValueError(f"Unknown augmentation profile: {profile!r}; use 'legacy' or 'mild'")


def augment_image(image, *, profile="legacy", seed=0):
    """Return (80, 784), bilinear interpolation and zero background.

    Legacy: rotations +/-45 degrees, scales 1.05..1.5, fixed order.
    Mild: rotations +/-15 degrees, scales 0.9..1.1, seeded shuffle of all
    80 views before batching. Scaling is centered on the image; content
    outside the original 28x28 canvas is cropped. Seed does not change the
    global RNG or the independently seeded VAE latent-noise generator.
    """
    config = augmentation_config(profile)
    image = torch.as_tensor(image, dtype=torch.float32)
    if image.numel() != 784:
        raise ValueError("Expected one 28x28 image")
    transforms = torch.zeros(80, 2, 3, device=image.device)
    limit = math.radians(config["rotation_degrees"])
    angles = torch.linspace(-limit, limit, 40, device=image.device)
    transforms[:40, 0, 0] = angles.cos()
    transforms[:40, 0, 1] = -angles.sin()
    transforms[:40, 1, 0] = angles.sin()
    transforms[:40, 1, 1] = angles.cos()
    transforms[40:, 0, 0] = 1
    transforms[40:, 1, 1] = 1
    scales = torch.linspace(config["scale_min"], config["scale_max"], 20, device=image.device)
    # affine_grid maps output coordinates back to input coordinates.
    transforms[40:60, 0, 0] = scales.reciprocal()
    transforms[60:80, 1, 1] = scales.reciprocal()
    images = image.reshape(1, 1, 28, 28).expand(80, -1, -1, -1)
    grid = F.affine_grid(transforms, images.shape, align_corners=False)
    augmented = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False).reshape(80, 784)
    if config["shuffle"]:
        order = torch.randperm(80, generator=torch.Generator().manual_seed(seed))
        augmented = augmented.index_select(0, order.to(image.device))
    return augmented
