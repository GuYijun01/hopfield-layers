"""VAE2 starts from the same parameters and fresh Adam state for every cue."""
import torch
from examples.vae_parameter_memory.vae import VAE, initial_vector, parameter_vector, vae_loss


def fit_block(initial_state, image_batches, lr, device, seed):
    """Return q = vec(theta_after_5) - vec(theta_init) after one block pass."""
    model = VAE().to(device)
    model.load_state_dict(initial_state)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device=device).manual_seed(seed)
    count = 0
    for images in image_batches:
        if len(images) != 16:
            raise ValueError("Each cue batch must contain exactly 16 images")
        images = images.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(images, generator=generator)
        vae_loss(logits, images, mu, logvar).mean().backward()
        optimizer.step()
        count += 1
    if count != 5:
        raise ValueError("Each VAE2 block must contain exactly 5 batches")
    return parameter_vector(model) - initial_vector(initial_state).to(device)


def generate_block_cues(initial_state, dataset, indices, batch_size, lr, device,
                        seed, passes=1):
    """Disjoint full 5-batch blocks per pass; discard the incomplete tail.

    Additional passes reshuffle the same training data. Blocks can overlap
    across passes, never within one pass. Validation/test images are excluded.
    """
    if batch_size != 16:
        raise ValueError("This experiment requires batch_size=16")
    block_size = 5 * batch_size
    if len(indices) < block_size:
        raise ValueError("Need at least 5 * batch_size training images per digit")
    cues, blocks = [], []
    for pass_id in range(passes):
        generator = torch.Generator().manual_seed(seed + pass_id)
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(shuffled) - block_size + 1, block_size):
            block = shuffled[start:start + block_size]
            batches = (torch.stack([dataset[int(i)][0] for i in part])
                       for part in block.split(batch_size))
            cues.append(fit_block(initial_state, batches, lr, device, seed + len(cues)))
            blocks.append(block.clone())
    return torch.stack(cues), torch.stack(blocks)
