"""VAE0 expert training, independent of Hopfield optimization."""
import copy
import torch
from .vae import VAE, initial_vector, parameter_vector, vae_loss


@torch.no_grad()
def validation_loss(model, batches, device, seed=0):
    # Fixed latent noise across epochs makes checkpoint comparisons reproducible.
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    total, count = 0.0, 0
    for images, _ in batches:
        images = images.to(device)
        logits, mu, logvar = model(images, generator=generator)
        total += vae_loss(logits, images, mu, logvar).sum().item()
        count += len(images)
    if count == 0:
        raise ValueError("Validation split must be nonempty")
    return total / count


def train_vae0(initial_state, train, validation, epochs, lr, device, cue_batches=50, seed=0):
    model = VAE().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device=device).manual_seed(seed)
    best_loss, best_state, best_epoch = float("inf"), None, None
    history, first_cue, updates = [], None, 0
    if len(train) < cue_batches:
        raise ValueError("VAE0 needs at least 50 minibatches in its first epoch")
    for epoch in range(epochs):
        model.train()
        for images, _ in train:
            images = images.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(images, generator=generator)
            vae_loss(logits, images, mu, logvar).mean().backward()
            optimizer.step()
            updates += 1
            if updates == cue_batches:
                first_cue = parameter_vector(model) - initial_vector(initial_state).to(device)
        score = validation_loss(model, validation, device, seed)
        history.append(score)
        if score < best_loss:
            best_loss, best_epoch = score, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return parameter_vector(model), first_cue, {
        "best_epoch": best_epoch, "best_validation_negative_elbo": best_loss,
        "validation_history": history, "updates_per_epoch": len(train)}
