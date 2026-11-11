"""Deterministic 40 rotations and 40 axis stretches of one MNIST image."""
import math
import torch
from torch.nn import functional as F


def augment_image(image):
    """Return (80, 784), bilinear interpolation and zero background.

    Rotations include both endpoints of [-pi/4, pi/4]. Twenty horizontal
    and twenty vertical stretches use factors 1.05 through 1.5, centered
    on the image; content outside the original 28x28 canvas is cropped.
    """
    image = torch.as_tensor(image, dtype=torch.float32)
    if image.numel() != 784:
        raise ValueError("Expected one 28x28 image")
    transforms = torch.zeros(80, 2, 3, device=image.device)
    angles = torch.linspace(-math.pi / 4, math.pi / 4, 40, device=image.device)
    transforms[:40, 0, 0] = angles.cos()
    transforms[:40, 0, 1] = -angles.sin()
    transforms[:40, 1, 0] = angles.sin()
    transforms[:40, 1, 1] = angles.cos()
    transforms[40:, 0, 0] = 1
    transforms[40:, 1, 1] = 1
    scales = torch.linspace(1.05, 1.5, 20, device=image.device)
    # affine_grid maps output coordinates back to input coordinates.
    transforms[40:60, 0, 0] = scales.reciprocal()
    transforms[60:80, 1, 1] = scales.reciprocal()
    images = image.reshape(1, 1, 28, 28).expand(80, -1, -1, -1)
    grid = F.affine_grid(transforms, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False).reshape(80, 784)
