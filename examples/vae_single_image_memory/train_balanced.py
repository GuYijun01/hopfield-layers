"""Larger image-cue training with balanced transforms and per-update resampling.

Outputs are named using a file prefix in an existing directory. No experiment
directories are created. Expert VAEs and the full-batch Hopfield objective are
reused; selected cue coordinates and a small memory Gram matrix bound memory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F

from .balanced_cues import PROFILE, SAMPLING, augmentation_config, fit_image_cue
from .evaluate import validate_checkpoint
from examples.vae_parameter_memory.data import digit_split, load_mnist
from examples.vae_parameter_memory.hopfield import make_retriever
from examples.vae_parameter_memory.vae import VAE, initial_vector


def output_file(prefix, suffix):
    path = Path(str(prefix) + suffix)
    if not path.parent.is_dir():
        raise ValueError(f"Output directory must already exist: {path.parent}")
    return path


def save_torch(value, path):
    # Use a Python file handle instead of PyTorch's native filename opener.
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def backup(args, suffix):
    if args.backup_prefix:
        shutil.copyfile(output_file(args.output_prefix, suffix), output_file(args.backup_prefix, suffix))


def learning_rate_schedule(steps, initial_lr, drop_after=None, final_lr=None):
    """Inclusive optimizer-step ranges; update drop_after+1 uses the new rate."""
    if steps < 1 or initial_lr <= 0:
        raise ValueError("Optimizer steps and initial learning rate must be positive")
    if (drop_after is None) != (final_lr is None):
        raise ValueError("Provide both --lr-drop-after and --hopfield-lr-final")
    if drop_after is None:
        return [{"start_step": 1, "end_step": steps, "lr": initial_lr}]
    if not 1 <= drop_after < steps or final_lr <= 0:
        raise ValueError("LR drop must occur between updates 1 and steps-1; final rate must be positive")
    return [{"start_step": 1, "end_step": drop_after, "lr": initial_lr},
            {"start_step": drop_after + 1, "end_step": steps, "lr": final_lr}]


def set_learning_rate(optimizer, step, schedule):
    """Change parameter-group rates in place, preserving all Adam moments."""
    phase = next(phase for phase in schedule if phase["start_step"] <= step <= phase["end_step"])
    for group in optimizer.param_groups:
        group["lr"] = phase["lr"]
    return phase["lr"]


def cache_features(initial_state, dataset, indices, count, feature_indices, lr, device, seed):
    if count < 1 or count > len(indices):
        raise ValueError(f"Requested {count} source images, but the split contains {len(indices)}")
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    sources = indices[order[:count]]
    seeds = torch.arange(count, dtype=torch.long) + seed
    features = torch.empty(count, len(feature_indices))
    for position, index in enumerate(sources.tolist()):
        cue = fit_image_cue(initial_state, dataset[index][0], lr, device, int(seeds[position]))
        features[position] = cue.index_select(0, feature_indices.to(device)).cpu()
        if (position + 1) % 50 == 0 or position + 1 == count:
            print(f"  generated {position + 1}/{count} cues", flush=True)
    return {"features": features, "sources": sources.cpu(), "seeds": seeds}


class CompactObjective:
    """Same cosine association and full-parameter MSE, without NxD retrieval.

    For memory deltas A, weights alpha and one-hot targets y:
    MSE = mean_i [(alpha_i-y_i) (A A^T / D) (alpha_i-y_i)^T].
    The memory Gram matrix uses float64 to limit cancellation near exact recall.
    This preserves gradients to the shared encoder; values remain full memories.
    """
    def __init__(self, hopfield, features, targets, memories, initial):
        self.hopfield, self.targets = hopfield, targets
        deltas = memories - initial
        keys = deltas.index_select(-1, hopfield.feature_indices)
        center = keys.mean(0, keepdim=True)
        scale = keys.std(0, keepdim=True, unbiased=False).clamp_min(1e-3)
        self.queries = (features - center) / scale
        self.keys = (keys - center) / scale
        self.gram = deltas.double() @ deltas.double().T / memories.shape[-1]
        self.variance = memories.var(unbiased=False).clamp_min(1e-8).double()
        self.one_hot = F.one_hot(targets, len(memories))

    def __call__(self):
        queries = F.normalize(self.hopfield.encoder(self.queries), dim=-1)
        keys = F.normalize(self.hopfield.encoder(self.keys), dim=-1)
        weights = (self.hopfield.beta * (queries @ keys.T)).softmax(-1)
        association = F.nll_loss(weights.clamp_min(1e-12).log(), self.targets)
        error = weights.double() - self.one_hot
        parameter = ((error @ self.gram) * error).sum(-1).mean().clamp_min(0) / self.variance
        accuracy = (weights.argmax(-1) == self.targets).float().mean()
        return association + parameter, association, parameter, accuracy

    @torch.no_grad()
    def metrics(self):
        return dict(zip(("total_loss", "association_loss", "normalized_parameter_mse", "cue_accuracy"),
                        [float(value) for value in self()]))


def run(args):
    source = Path(args.source).resolve()
    checkpoint_path = output_file(args.output_prefix, "_checkpoint.pt")
    if checkpoint_path.resolve() == source:
        raise ValueError("Choose a new output prefix to preserve the source checkpoint")
    if args.backup_prefix:
        if output_file(args.backup_prefix, "_checkpoint.pt").resolve() == source:
            raise ValueError("Backup would overwrite the source checkpoint")
        if Path(args.backup_prefix).resolve() == Path(args.output_prefix).resolve():
            raise ValueError("Backup prefix must differ from the local output prefix")
    torch.set_num_threads(args.threads)
    source_checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    validate_checkpoint(source_checkpoint)
    if source_checkpoint["digits"] != list(range(10)):
        raise ValueError("Requires all ten expert memories")
    config = dict(source_checkpoint["config"])
    seed = config["seed"]
    steps = args.hopfield_steps or config["hopfield_steps"]
    lr = args.hopfield_lr or config["hopfield_lr"]
    schedule = learning_rate_schedule(steps, lr, getattr(args, "lr_drop_after", None),
                                      getattr(args, "hopfield_lr_final", None))
    boundaries = {phase["start_step"] for phase in schedule[1:]} | {phase["end_step"] for phase in schedule[:-1]}
    print("Learning-rate schedule per trained stage: " + json.dumps(schedule), flush=True)
    torch.manual_seed(seed)
    VAE()  # Preserve the original projection initialization sequence.
    hopfield = make_retriever(source_checkpoint["parameter_count"], config["feature_size"],
                              config["association_size"]).to(args.device)
    fresh_state = copy.deepcopy(hopfield.state_dict())
    feature_indices = hopfield.feature_indices.cpu()
    initial = initial_vector(source_checkpoint["initial_vae"]).to(args.device)
    memories = source_checkpoint["memories"].detach().to(args.device)
    dataset = load_mnist(config["data_dir"], train=True)
    splits = [digit_split(dataset, digit, seed + digit, config.get("train_limit", 0)) for digit in range(10)]
    counts = []
    for digit, (train, validation) in enumerate(splits):
        train_count = args.images_per_digit or len(train)
        if train_count < 1 or train_count > len(train) or args.validation_per_digit > len(validation):
            raise ValueError(f"Digit {digit}: requested {train_count} training/{args.validation_per_digit} "
                             f"validation images; available {len(train)}/{len(validation)}")
        counts.append(train_count)
    # Fingerprint the expert bank and all cue-generation choices before reusing a cache.
    digest = hashlib.sha256()
    for value in (initial.cpu(), memories.cpu()):
        digest.update(value.numpy().tobytes())
    recipe = {"seed": seed, "train_limit": config.get("train_limit", 0), "lr": config["lr"],
              "counts": counts, "validation_per_digit": args.validation_per_digit,
              "augmentation": augmentation_config(), "feature_indices": feature_indices.tolist()}
    digest.update(json.dumps(recipe, sort_keys=True).encode())
    signature = digest.hexdigest()
    cache_path = output_file(args.output_prefix, "_cue_cache.pt")
    cache_source = Path(args.cue_cache) if getattr(args, "cue_cache", None) else cache_path
    if getattr(args, "cue_cache", None) and not cache_source.is_file():
        raise FileNotFoundError(f"Cue cache not found: {cache_source}. Restore the saved cache to local storage first.")
    cache = {"signature": signature, "recipe": recipe, "digits": []}
    if cache_source.exists():
        cache = torch.load(cache_source, map_location="cpu", weights_only=True)
        if cache["signature"] != signature:
            raise ValueError("Existing cue cache has different settings; choose another file prefix")
        print(f"Reusing cached cues for {len(cache['digits'])} digits", flush=True)
        if cache_source.resolve() != cache_path.resolve():
            save_torch(cache, cache_path)
    for digit in range(len(cache["digits"]), 10):
        train, validation = splits[digit]
        print(f"Digit {digit}: {counts[digit]} training and {args.validation_per_digit} validation source images", flush=True)
        cache["digits"].append({
            "digit": digit,
            "train": cache_features(source_checkpoint["initial_vae"], dataset, train, counts[digit],
                                    feature_indices, config["lr"], args.device, seed + 1000 + digit * 10000),
            "validation": cache_features(source_checkpoint["initial_vae"], dataset, validation, args.validation_per_digit,
                                         feature_indices, config["lr"], args.device, seed + 1000000 + digit * 10000)})
        save_torch(cache, cache_path)
        backup(args, "_cue_cache.pt")
    # Also back up a reused cache when the backup destination is newly supplied.
    backup(args, "_cue_cache.pt")

    def concatenate(split, key):
        return torch.cat([row[split][key] for row in cache["digits"]])

    train_features = concatenate("train", "features").to(args.device)
    val_features = concatenate("validation", "features").to(args.device)
    targets = torch.cat([torch.full((counts[d],), d, dtype=torch.long) for d in range(10)]).to(args.device)
    val_targets = torch.arange(10, device=args.device).repeat_interleave(args.validation_per_digit)
    hopfield.load_state_dict(source_checkpoint["hopfield"])
    frozen = {"train": CompactObjective(hopfield, train_features, targets, memories, initial).metrics(),
              "validation": CompactObjective(hopfield, val_features, val_targets, memories, initial).metrics()}
    report = {"source_checkpoint": str(source), "config": vars(args), "training_config": config,
              "lr_schedule": schedule,
              "augmentation_profile": PROFILE, "augmentation_config": augmentation_config(),
              "plot_title": "Balanced 80-view cues, per-update shuffle",
              "frozen_original": frozen, "frozen_original_description": "Source projection on the new cues",
              "train_cues": len(targets), "validation_cues": len(val_targets),
              "protocol": "Five independent 16-view draws from fresh permutations of one image's 80-view pool. "
                          "Views may recur across updates. Distinct source images from the original 90/10 split; "
                          "validation previously selected experts but never updates or selects Hopfield. "
                          "Unchanged full-batch CE plus normalized full-parameter MSE; no test data.", "stages": []}
    hopfield.load_state_dict(fresh_state)
    for digit in range(10):
        train_mask, val_mask = targets <= digit, val_targets <= digit
        train_objective = CompactObjective(hopfield, train_features[train_mask], targets[train_mask], memories[:digit + 1], initial)
        val_objective = CompactObjective(hopfield, val_features[val_mask], val_targets[val_mask], memories[:digit + 1], initial)
        history = []

        def record(step):
            rate = schedule[0]["lr"] if step == 0 else next(
                phase["lr"] for phase in schedule if phase["start_step"] <= step <= phase["end_step"])
            history.append({"step": step, "lr": rate, "train": train_objective.metrics(), "validation": val_objective.metrics()})

        record(0)
        if digit:
            optimizer = torch.optim.Adam(hopfield.parameters(), lr=lr)
            for step in range(1, steps + 1):
                previous_rate = optimizer.param_groups[0]["lr"]
                active_rate = set_learning_rate(optimizer, step, schedule)
                if active_rate != previous_rate:
                    print(f"Stage {digit + 1}, update {step}: learning rate -> {active_rate:g}", flush=True)
                optimizer.zero_grad()
                train_objective()[0].backward()
                optimizer.step()
                if step % args.log_every == 0 or step == steps or step in boundaries:
                    record(step)
        else:
            record(steps)
        report["stages"].append({"digits_seen": digit + 1, "train_cues": int(train_mask.sum()),
                                  "validation_cues": int(val_mask.sum()), "steps": steps, "lr": lr,
                                  "lr_schedule": schedule, "history": history})
        output_file(args.output_prefix, "_training.json").write_text(json.dumps(report, indent=2) + "\n")
        trained = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
                   "hopfield_cue_protocol": PROFILE, "augmentation_profile": PROFILE,
                   "augmentation_config": augmentation_config(), "cue_sampling": SAMPLING,
                   "config": {**config, "hopfield_steps": steps, "hopfield_lr": lr,
                              "hopfield_lr_final": getattr(args, "hopfield_lr_final", None),
                              "lr_drop_after": getattr(args, "lr_drop_after", None)},
                   "lr_schedule": schedule,
                   "parameter_count": source_checkpoint["parameter_count"],
                   "initial_vae": source_checkpoint["initial_vae"], "memories": source_checkpoint["memories"],
                   "digits": list(range(digit + 1)), "source_checkpoint": str(source),
                   "hopfield": {k: v.detach().cpu() for k, v in hopfield.state_dict().items()},
                   "cue_features": train_features[train_mask].detach().cpu(),
                   "targets": targets[train_mask].cpu(),
                   "cue_source_indices": concatenate("train", "sources")[train_mask.cpu()],
                   "cue_seeds": concatenate("train", "seeds")[train_mask.cpu()]}
        save_torch(trained, checkpoint_path)
        backup(args, "_checkpoint.pt")
        backup(args, "_training.json")
        print(f"Stage {digit + 1}: " + json.dumps(history[-1]), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Existing ten-expert checkpoint, baseline or augmented")
    parser.add_argument("--output-prefix", default="balanced80", help="File prefix in an existing directory")
    parser.add_argument("--backup-prefix", help="Optional file prefix in an existing Drive directory")
    parser.add_argument("--cue-cache", help="Reuse an existing cue cache under a new output prefix")
    parser.add_argument("--images-per-digit", type=int, default=500, help="Distinct training sources per digit; 0 uses all")
    parser.add_argument("--validation-per-digit", type=int, default=100)
    parser.add_argument("--hopfield-steps", type=int, help="Default: original optimizer steps per replay stage")
    parser.add_argument("--hopfield-lr", type=float, help="Default: original Hopfield learning rate")
    parser.add_argument("--lr-drop-after", type=int, help="Switch learning rate after this optimizer update in each stage")
    parser.add_argument("--hopfield-lr-final", type=float, help="Learning rate for updates after lr-drop-after")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.images_per_digit < 0 or min(args.validation_per_digit, args.log_every, args.threads) < 1
            or (args.hopfield_steps is not None and args.hopfield_steps < 1)
            or (args.hopfield_lr is not None and args.hopfield_lr <= 0)):
        parser.error("Counts and learning rates must be positive; images-per-digit may be zero")
    if (args.lr_drop_after is None) != (args.hopfield_lr_final is None):
        parser.error("Provide both --lr-drop-after and --hopfield-lr-final")
    if args.lr_drop_after is not None and (args.lr_drop_after < 1 or args.hopfield_lr_final <= 0):
        parser.error("LR drop step and final learning rate must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
