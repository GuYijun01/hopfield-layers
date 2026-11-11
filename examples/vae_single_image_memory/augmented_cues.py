"""Generate label-free five-update cues from individual images' augmentations."""
import torch

from .augment import augment_image
from .cues import fit_block


def generate_augmented_cues(initial_state, dataset, indices, count, lr, device, seed, *,
                             augmentation_profile="legacy"):
    """Select distinct source images, then use the exact test-time transforms/order.

    The caller supplies indices from one data split; labels are never read here.
    Each source independently resets VAE parameters and Adam inside fit_block.
    """
    indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    if count < 1 or count > len(indices) or indices.unique().numel() != len(indices):
        raise ValueError("Need enough distinct source indices for the requested cue count")
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    selected = indices[order[:count]]
    cues = []
    for position, index in enumerate(selected.tolist()):
        image = dataset[index][0]
        # Match predict_image: transform on the input image's device, then move.
        batches = augment_image(image, profile=augmentation_profile, seed=seed + position).to(device).split(16)
        cues.append(fit_block(initial_state, batches, lr, device, seed + position))
    return torch.stack(cues), selected
