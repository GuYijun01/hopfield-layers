"""MNIST loading and reproducible, disjoint train/validation splits."""
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class MNISTTensors(Dataset):
    def __init__(self, mnist):
        self.images = mnist.data.reshape(-1, 784).float().div(255)
        self.targets = mnist.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


def load_mnist(root, train):
    from torchvision.datasets import MNIST
    return MNISTTensors(MNIST(root, train=train, download=True))


def digit_split(dataset, digit, seed, limit=0):
    indices = (dataset.targets == digit).nonzero().flatten()
    indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))]
    if limit:
        indices = indices[:limit]
    split = int(0.9 * len(indices))
    return indices[:split], indices[split:]


def make_loader(dataset, indices, batch_size, seed, shuffle=True, drop_last=False):
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      generator=torch.Generator().manual_seed(seed))
