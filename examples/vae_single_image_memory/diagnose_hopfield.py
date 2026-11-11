"""Evaluate saved Hopfield weights, then record a replay refit's learning curves."""
import argparse
import copy
import json
from pathlib import Path

import torch

from .cues import generate_block_cues
from .evaluate import validate_checkpoint
from examples.vae_parameter_memory.data import digit_split, load_mnist
from examples.vae_parameter_memory.hopfield import (
    make_retriever, retriever_metrics, train_retriever,
)
from examples.vae_parameter_memory.vae import VAE, initial_vector


def run(args):
    source = Path(args.checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    if output == source.parent:
        raise ValueError("Use a separate diagnostic output directory to preserve your original run")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    if checkpoint.get("hopfield_cue_protocol", "distinct_image_blocks_v1") != "distinct_image_blocks_v1":
        raise ValueError("This diagnostic is for the block-trained baseline. For augmented training, "
                         "plot the hopfield_training.json already written by train_augmented.")
    if checkpoint["digits"] != list(range(10)):
        raise ValueError("Diagnostics require all ten stored memories")
    config = checkpoint["config"]
    seed = config["seed"]
    torch.manual_seed(seed)
    # Match the original construction order: initial VAE, then projection.
    VAE()
    hopfield = make_retriever(checkpoint["parameter_count"], config["feature_size"],
                              config["association_size"]).to(args.device)
    fresh_state = copy.deepcopy(hopfield.state_dict())
    initial = initial_vector(checkpoint["initial_vae"]).to(args.device)
    memories = checkpoint["memories"].to(args.device)
    cues, targets = checkpoint["cues"].to(args.device), checkpoint["targets"].to(args.device)
    dataset = load_mnist(config["data_dir"], train=True)
    val_cues, val_targets, val_blocks = [], [], []
    for digit in checkpoint["digits"]:
        train_indices, val_indices = digit_split(
            dataset, digit, seed + digit, config.get("train_limit", 0))
        if len(val_indices) < 80:
            raise ValueError(f"Digit {digit} needs at least 80 validation images; "
                             "use a full-data checkpoint or --train-limit >= 800 when training")
        # Check provenance against both the split and the actual cached cue blocks.
        cached_blocks = checkpoint["block_indices"][digit].reshape(-1)
        if not torch.isin(cached_blocks, train_indices).all():
            raise ValueError("Cached training block indices do not match the configured data split")
        if torch.isin(cached_blocks, val_indices).any():
            raise ValueError("Training and validation cue images overlap")
        if args.validation_blocks_per_digit:
            val_indices = val_indices[:80 * args.validation_blocks_per_digit]
        variants, blocks = generate_block_cues(
            checkpoint["initial_vae"], dataset, val_indices, 16, config["lr"],
            args.device, seed + 20000 + digit * 100, passes=1)
        val_cues.append(variants)
        val_targets.append(torch.full((len(variants),), digit, dtype=torch.long, device=args.device))
        val_blocks.append(blocks)
        print(f"Digit {digit}: {len(variants)} held-out validation block cues", flush=True)
    validation_cues, validation_targets = torch.cat(val_cues), torch.cat(val_targets)
    torch.save({"cues": validation_cues.cpu(), "targets": validation_targets.cpu(),
                "block_indices": val_blocks}, output / "validation_cues.pt")

    # First evaluate the exact original projection, with no optimization.
    hopfield.load_state_dict(checkpoint["hopfield"])
    hopfield.eval()
    frozen = {"train": retriever_metrics(hopfield, cues, targets, memories, initial),
              "validation": retriever_metrics(hopfield, validation_cues, validation_targets, memories, initial)}
    print("Original saved Hopfield: " + json.dumps(frozen), flush=True)
    report = {"source_checkpoint": str(source), "config": vars(args),
              "training_config": config, "frozen_original": frozen,
              "train_cues": len(cues), "validation_cues": len(validation_cues),
              "protocol": "Original five-update, 80-distinct-image block cues; no augmented-single-image "
                          "cues and no test data. Validation cues use the original 10% expert validation "
                          "split. These images never train Hopfield or expert gradients, but did select "
                          "expert checkpoints. Curves are a fresh sequential projection refit, not recovered "
                          "historical training. Validation never updates or selects projection weights.",
              "stages": []}
    (output / "hopfield_training.json").write_text(json.dumps(report, indent=2) + "\n")
    hopfield.load_state_dict(fresh_state)
    steps = args.steps if args.steps is not None else config["hopfield_steps"]
    lr = args.lr if args.lr is not None else config["hopfield_lr"]
    for digit in checkpoint["digits"]:
        train_mask, val_mask = targets <= digit, validation_targets <= digit
        summary = train_retriever(
            hopfield, cues[train_mask], targets[train_mask], memories[:digit + 1],
            steps, lr, initial, validation_cues=validation_cues[val_mask],
            validation_targets=validation_targets[val_mask], history_every=args.log_every)
        report["stages"].append({"digits_seen": digit + 1, "steps": steps, "lr": lr,
                                  "train_cues": int(train_mask.sum()),
                                  "validation_cues": int(val_mask.sum()), **summary})
        (output / "hopfield_training.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Stage {digit + 1}: " + json.dumps(summary["history"][-1]), flush=True)
    diagnostic_checkpoint = dict(checkpoint)
    diagnostic_checkpoint["hopfield"] = {k: v.detach().cpu() for k, v in hopfield.state_dict().items()}
    diagnostic_checkpoint["diagnostic_source"] = str(source)
    torch.save(diagnostic_checkpoint, output / "refit_checkpoint.pt")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="work/vae_single_image_memory/checkpoint.pt")
    parser.add_argument("--output-dir", default="work/vae_single_image_memory/hopfield_diagnostics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, help="Default: original steps per arriving digit")
    parser.add_argument("--lr", type=float, help="Default: original Hopfield learning rate")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--validation-blocks-per-digit", type=int, default=0,
                        help="0 uses all nonoverlapping full validation blocks")
    args = parser.parse_args()
    if (args.threads < 1 or args.log_every < 1 or args.validation_blocks_per_digit < 0
            or (args.steps is not None and args.steps < 1)
            or (args.lr is not None and args.lr <= 0)):
        parser.error("Counts and learning rates must be positive; validation block cap may be zero")
    return args


if __name__ == "__main__":
    run(parse_args())
