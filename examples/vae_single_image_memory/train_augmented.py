"""Refit Hopfield on single-image augmented cues, reusing a baseline expert bank."""
import argparse
import copy
import json
from pathlib import Path

import torch

from .augmented_cues import generate_augmented_cues
from .evaluate import validate_checkpoint
from examples.vae_parameter_memory.data import digit_split, load_mnist
from examples.vae_parameter_memory.hopfield import (
    make_retriever, retriever_metrics, train_retriever,
)
from examples.vae_parameter_memory.vae import VAE, initial_vector


def run(args):
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    if output == source.parent:
        raise ValueError("Use a separate output directory to preserve the baseline checkpoint")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    if checkpoint["digits"] != list(range(10)):
        raise ValueError("Requires a baseline checkpoint with all ten expert memories")
    if checkpoint.get("hopfield_cue_protocol", "distinct_image_blocks_v1") != "distinct_image_blocks_v1":
        raise ValueError("Use the original distinct-image-block checkpoint as --source")
    config = dict(checkpoint["config"])
    seed = config["seed"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    # Preserve the baseline's initial projection and its architecture.
    VAE()
    hopfield = make_retriever(checkpoint["parameter_count"], config["feature_size"],
                              config["association_size"]).to(args.device)
    fresh_state = copy.deepcopy(hopfield.state_dict())
    dataset = load_mnist(config["data_dir"], train=True)
    train_cues, train_targets, train_sources = [], [], []
    val_cues, val_targets, val_sources = [], [], []
    for digit in checkpoint["digits"]:
        train_indices, val_indices = digit_split(
            dataset, digit, seed + digit, config.get("train_limit", 0))
        cached_blocks = checkpoint["block_indices"][digit].reshape(-1)
        if not torch.isin(cached_blocks, train_indices).all():
            raise ValueError("Baseline cue indices do not match the configured data split")
        if torch.isin(cached_blocks, val_indices).any():
            raise ValueError("Baseline training and validation images overlap")
        train_count = args.cues_per_digit or int((checkpoint["targets"] == digit).sum())
        val_count = args.validation_cues_per_digit or max(1, len(val_indices) // 80)
        print(f"Digit {digit}: generating {train_count} training and {val_count} validation "
              "single-image cues", flush=True)
        cues, sources = generate_augmented_cues(
            checkpoint["initial_vae"], dataset, train_indices, train_count,
            config["lr"], args.device, seed + 1000 + digit * 10000)
        validation, validation_sources = generate_augmented_cues(
            checkpoint["initial_vae"], dataset, val_indices, val_count,
            config["lr"], args.device, seed + 1000000 + digit * 10000)
        train_cues.append(cues)
        train_targets.append(torch.full((len(cues),), digit, dtype=torch.long, device=args.device))
        train_sources.append(sources)
        val_cues.append(validation)
        val_targets.append(torch.full((len(validation),), digit, dtype=torch.long, device=args.device))
        val_sources.append(validation_sources)
    cues, targets = torch.cat(train_cues), torch.cat(train_targets)
    validation_cues, validation_targets = torch.cat(val_cues), torch.cat(val_targets)
    initial = initial_vector(checkpoint["initial_vae"]).to(args.device)
    memories = checkpoint["memories"].to(args.device)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"cues": validation_cues.cpu(), "targets": validation_targets.cpu(),
                "source_indices": torch.cat(val_sources),
                "cue_protocol": "single_image_augmented_v1"}, output / "validation_cues.pt")

    # Score the baseline on these same augmented cue sets for a direct reference.
    hopfield.load_state_dict(checkpoint["hopfield"])
    hopfield.eval()
    baseline = {"train": retriever_metrics(hopfield, cues, targets, memories, initial),
                "validation": retriever_metrics(hopfield, validation_cues, validation_targets, memories, initial)}
    print("Frozen baseline on augmented cues: " + json.dumps(baseline), flush=True)
    report = {"source_checkpoint": str(source), "config": vars(args), "training_config": config,
              "hopfield_cue_protocol": "single_image_augmented_v1",
              "frozen_original": baseline,
              "frozen_original_description": "Frozen block-trained baseline on the same augmented cue sets",
              "plot_title": "Hopfield training on single-image augmented cues",
              "train_cues": len(cues), "validation_cues": len(validation_cues),
              "protocol": "Reuse identical expert memories and initialization. Fresh shared Hopfield "
                          "projection, original sequential replay, steps, learning rate and objective. "
                          "Each source image yields the unchanged 40 rotations then 20 horizontal and "
                          "20 vertical stretches, in fixed order, five batches of sixteen. Fresh VAE2 "
                          "and Adam per source. Source images are sampled without replacement within "
                          "each original training/validation split. Validation images previously selected "
                          "expert checkpoints but never update or select Hopfield weights. No test data.",
              "stages": []}
    hopfield.load_state_dict(fresh_state)
    for digit in checkpoint["digits"]:
        train_mask, val_mask = targets <= digit, validation_targets <= digit
        summary = train_retriever(
            hopfield, cues[train_mask], targets[train_mask], memories[:digit + 1],
            config["hopfield_steps"], config["hopfield_lr"], initial,
            validation_cues=validation_cues[val_mask], validation_targets=validation_targets[val_mask],
            history_every=args.log_every)
        report["stages"].append({"digits_seen": digit + 1, "steps": config["hopfield_steps"],
                                  "lr": config["hopfield_lr"], "train_cues": int(train_mask.sum()),
                                  "validation_cues": int(val_mask.sum()), **summary})
        (output / "hopfield_training.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Stage {digit + 1}: " + json.dumps(summary["history"][-1]), flush=True)
        # Preserve expert tensors exactly; replace only the projection and its training cues.
        trained = {key: value for key, value in checkpoint.items()
                   if key not in ("block_indices", "diagnostic_source", "cues", "targets", "hopfield")}
        trained.update({"config": {**config, "output_dir": str(output)},
                        "hopfield_cue_protocol": "single_image_augmented_v1",
                        "source_checkpoint": str(source),
                        "hopfield": {k: v.detach().cpu() for k, v in hopfield.state_dict().items()},
                        "cues": cues[train_mask].detach().cpu(), "targets": targets[train_mask].cpu(),
                        "cue_source_indices": torch.cat(train_sources[:digit + 1]),
                        "digits": list(range(digit + 1))})
        torch.save(trained, output / "checkpoint.pt")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="work/vae_single_image_memory/checkpoint.pt")
    parser.add_argument("--output-dir", default="work/vae_single_image_augmented")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--cues-per-digit", type=int, default=0,
                        help="0 matches the baseline's number of training cues for each digit")
    parser.add_argument("--validation-cues-per-digit", type=int, default=0,
                        help="0 matches the original diagnostic's full validation-block count (minimum 1)")
    args = parser.parse_args()
    if args.threads < 1 or args.log_every < 1 or min(args.cues_per_digit, args.validation_cues_per_digit) < 0:
        parser.error("Threads and log interval must be positive; cue count overrides must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
