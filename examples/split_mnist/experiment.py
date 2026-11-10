"""Split MNIST parameter-memory experiment with labeled test-time adaptation.

Run from the repository root: python -m examples.split_mnist.experiment --help
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset

from hflayers.modern import ModernHopfield

TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))


class MLP(nn.Module):
    """784–400–400–10 classifier; forward returns logits for cross entropy."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 400), nn.ReLU(),
                                    nn.Linear(400, 400), nn.ReLU(), nn.Linear(400, 10))

    def forward(self, x):
        return self.layers(x)


def vector(model):
    return parameters_to_vector(model.parameters()).detach().clone()


def loader(dataset, indices, batch_size, shuffle=False, seed=0):
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


def subset_indices(dataset, pair, generator):
    mask = (dataset.targets == pair[0]) | (dataset.targets == pair[1])
    indices = mask.nonzero().flatten()
    return indices[torch.randperm(len(indices), generator=generator)]


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # All ten logits: no task ID or class mask.
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    return {"loss": loss / count, "accuracy": correct / count, "examples": count}


def train_expert(initial, train, validation, args, device):
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_state, cue, best_epoch = math.inf, None, None, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        for x, y in train:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        if epoch == 0:
            cue = vector(model)
        metrics = evaluate(model, validation, device)
        history.append(metrics)
        if metrics["loss"] < best_loss:
            best_loss, best_epoch = metrics["loss"], epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return vector(model), cue, {"best_epoch": best_epoch, "validation": history}


def fit_hopfield(hopfield, memories, cues, args):
    """Replay all observed cue/memory pairs after every arriving task."""
    optimizer = torch.optim.Adam(hopfield.parameters(), lr=args.hopfield_lr)
    labels = torch.arange(len(memories), device=memories.device)
    if len(memories) == 1:
        return {"association_loss": 0.0, "retrieval_accuracy": 1.0}
    hopfield.train()
    for _ in range(args.hopfield_steps):
        optimizer.zero_grad()
        retrieved, weights = hopfield(cues, memories, return_associations=True)
        association_loss = F.nll_loss(weights.clamp_min(1e-12).log(), labels)
        # Normalize reconstruction error by memory variance, keeping its scale useful.
        reconstruction = F.mse_loss(retrieved, memories) / memories.var(unbiased=False).clamp_min(1e-8)
        loss = association_loss + reconstruction
        loss.backward()
        optimizer.step()
    hopfield.eval()
    with torch.no_grad():
        _, weights = hopfield(cues, memories, return_associations=True)
    return {"association_loss": F.nll_loss(weights.clamp_min(1e-12).log(), labels).item(),
            "retrieval_accuracy": (weights.argmax(-1) == labels).float().mean().item()}


def adapt(initial, support, n, args, device):
    """Use the identical prefix and optimizer state for the paired N/2 and N cues."""
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    half = max(1, n // 2)
    cues = {}
    iterator = iter(support)
    model.train()
    for step in range(1, n + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(support)
            x, y = next(iterator)
        optimizer.zero_grad()
        F.cross_entropy(model(x.to(device)), y.to(device)).backward()
        optimizer.step()
        if step == half:
            cues["N/2"] = vector(model)
        if step == n:
            cues["N"] = vector(model)
    return cues


@torch.no_grad()
def test_retrieval(initial, hopfield, memories, cues, evaluation, device):
    rows = {}
    model = copy.deepcopy(initial).to(device)
    for budget, cue in cues.items():
        retrieved, weights = hopfield(cue.unsqueeze(0), memories, return_associations=True)
        vector_to_parameters(cue.clone(), model.parameters())
        direct = evaluate(model, evaluation, device)
        vector_to_parameters(retrieved.squeeze(0).clone(), model.parameters())
        rows[budget] = {"cue": direct, "retrieved": evaluate(model, evaluation, device),
                        "memory_weights": weights.squeeze(0).cpu().tolist(),
                        "selected_memory": weights.argmax(-1).item()}
    rows["loss_N_minus_N_half"] = rows["N"]["retrieved"]["loss"] - rows["N/2"]["retrieved"]["loss"]
    return rows


def run(args):
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    initial = MLP().to(device)
    assert sum(p.numel() for p in initial.parameters()) == 478410
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_data = MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = MNIST(args.data_dir, train=False, download=True, transform=transform)
    hopfield = ModernHopfield(478410, args.association_size, args.feature_size,
                              update_steps_max=0).to(device)
    bank, training_cues, test_cues, evaluations, stages, training = [], [], [], [], [], []
    generator = torch.Generator().manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"config": vars(args), "tasks": TASKS[:args.tasks], "parameter_count": 478410,
              "protocol": "Sequential expert-memory learning; common MLP initialization; replay of past cues; "
                          "labeled test support adaptation; disjoint test queries; ten logits, no task mask. "
                          "This is an adaptation-assisted Class-IL experiment, not standard zero-adaptation Class-IL.",
              "training": training, "stages": stages}
    for task_id, pair in enumerate(TASKS[:args.tasks]):
        indices = subset_indices(train_data, pair, generator)
        if args.train_limit:
            indices = indices[:args.train_limit]
        split = max(1, int(0.9 * len(indices)))
        train = loader(train_data, indices[:split], args.batch_size, True, args.seed + task_id)
        validation = loader(train_data, indices[split:], args.batch_size)
        memory, cue, info = train_expert(initial, train, validation, args, device)
        n = len(train)
        info.update({"task": pair, "N": n, "N_half": max(1, n // 2)})
        training.append(info)
        bank.append(memory)
        training_cues.append(cue)
        memories = torch.stack(bank)
        fit = fit_hopfield(hopfield, memories, torch.stack(training_cues), args)
        test_indices = subset_indices(test_data, pair, generator)
        if args.test_limit:
            test_indices = test_indices[:args.test_limit]
        support_size = len(test_indices) // 2
        support = loader(test_data, test_indices[:support_size], args.batch_size, True, args.seed + task_id)
        evaluations.append(loader(test_data, test_indices[support_size:], args.batch_size))
        test_cues.append(adapt(initial, support, n, args, device))
        results = [test_retrieval(initial, hopfield, memories, cues, batches, device)
                   for cues, batches in zip(test_cues, evaluations)]
        means = {budget: {metric: sum(r[budget]["retrieved"][metric] for r in results) / len(results)
                          for metric in ("loss", "accuracy")} for budget in ("N/2", "N")}
        stage = {"after_task": task_id, "hopfield_training": fit, "tasks": results, "mean": means,
                 "mean_loss_N_minus_N_half": means["N"]["loss"] - means["N/2"]["loss"]}
        stages.append(stage)
        print(json.dumps({"task": pair, "N": n, "mean": means,
                          "loss_difference": stage["mean_loss_N_minus_N_half"]}), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        torch.save({"initial_mlp": initial.state_dict(), "hopfield": hopfield.state_dict(),
                    "memories": memories.cpu(), "training_cues": torch.stack(training_cues).cpu(),
                    "test_cues": [{k: v.cpu() for k, v in c.items()} for c in test_cues],
                    "config": vars(args), "tasks": TASKS[:task_id + 1]}, output / "checkpoint.pt")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="work/data")
    parser.add_argument("--output-dir", default="work/split_mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tasks", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hopfield-lr", type=float, default=1e-3)
    parser.add_argument("--hopfield-steps", type=int, default=200)
    parser.add_argument("--association-size", type=int, default=64)
    parser.add_argument("--feature-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=0, help="Per-task train+validation cap; 0 uses all")
    parser.add_argument("--test-limit", type=int, default=0, help="Per-task support+query cap; 0 uses all")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hopfield_steps, args.threads) < 1:
        parser.error("Epochs, batch size, Hopfield steps and threads must be positive")
    if args.train_limit and args.train_limit < 20 or args.test_limit and args.test_limit < 4:
        parser.error("Use at least 20 training and 4 test examples per task")
    return args


if __name__ == "__main__":
    run(parse_args())
