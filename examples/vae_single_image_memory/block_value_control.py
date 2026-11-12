"""Attach reversible block Wv to a trained baseline and compare identical cue banks."""
import argparse
import json
from pathlib import Path

import torch

from .evaluate import validate_checkpoint
from .seed_variants import assemble_cue_banks
from .train_balanced import CompactObjective, backup, output_file, save_torch
from examples.vae_parameter_memory.hopfield import (
    checkpoint_projection_mode, make_retriever, retriever_from_checkpoint,
)
from examples.vae_parameter_memory.vae import initial_vector


def attach_value_projection(checkpoint, block_size, device):
    config = checkpoint["config"]
    value_seed = config["seed"]
    model = make_retriever(checkpoint["parameter_count"], config["feature_size"], config["association_size"],
                           projection_mode=checkpoint_projection_mode(checkpoint), value_projection="block_diagonal",
                           value_block_size=block_size, value_seed=value_seed).to(device)
    state = model.state_dict()
    # Preserve every query/key weight and selected coordinate exactly. Only Wv is new.
    for name in state:
        if not name.startswith("value_projection."):
            state[name] = checkpoint["hopfield"][name]
    model.load_state_dict(state)
    attached = dict(checkpoint)
    fields = {"value_projection": "block_diagonal", "value_block_size": block_size, "value_seed": value_seed}
    attached.update(fields)
    attached["config"] = {**config, **fields}
    attached["value_control_only"] = True
    attached["hopfield"] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return model, attached


def run(args):
    source = Path(args.source).resolve()
    destination = output_file(args.output_prefix, "_checkpoint.pt")
    if source == destination.resolve():
        raise ValueError("Use a different output prefix to preserve the baseline checkpoint")
    if args.backup_prefix and output_file(args.backup_prefix, "_checkpoint.pt").resolve() == source:
        raise ValueError("Backup would overwrite the baseline checkpoint")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    if checkpoint["digits"] != list(range(10)):
        raise ValueError("Requires all ten trained expert memories")
    base = torch.load(args.cue_cache, map_location="cpu", weights_only=True)
    expanded = None
    if checkpoint.get("seed_variants", 1) > 1:
        if not args.seed_cache:
            raise ValueError("Provide the multi-seed run's --seed-cache to compare its held-out-seed bank")
        expanded = torch.load(args.seed_cache, map_location="cpu", weights_only=True)
        if expanded["recipe"]["base_signature"] != base["signature"]:
            raise ValueError("Base and seed caches do not belong to the same run")
    banks = assemble_cue_banks(base, expanded)
    for key, saved in (("features", "cue_features"), ("sources", "cue_source_indices"), ("seeds", "cue_seeds"), ("targets", "targets")):
        if saved not in checkpoint or not torch.equal(banks["train"][key], checkpoint[saved]):
            raise ValueError("Cue caches do not match the source checkpoint's training bank")
    original = retriever_from_checkpoint(checkpoint, args.device)
    model, attached = attach_value_projection(checkpoint, args.value_block_size, args.device)
    attached["value_control_source"] = str(source)
    memories = checkpoint["memories"].to(args.device)
    initial = initial_vector(checkpoint["initial_vae"]).to(args.device)
    report = {"source_checkpoint": str(source), "optimizer_updates": 0,
              "value_projection": "block_diagonal", "value_block_size": args.value_block_size,
              "roundtrip": model.value_projection.diagnostics(memories), "comparisons": {},
              "interpretation": "Exact reverse preserves the weighted memory mixture. Wq/Wk are copied "
                                "without retraining. Wv's analytical gradient under the original objective is zero. "
                                "A soft mixture decodes to that mixture, not automatically one exact expert."}
    print("Expert round-trip: " + json.dumps(report["roundtrip"]), flush=True)
    with torch.no_grad():
        for split, bank in banks.items():
            features, targets = bank["features"].to(args.device), bank["targets"].to(args.device)
            reference = CompactObjective(original, features, targets, memories, initial)
            projected = CompactObjective(model, features, targets, memories, initial)
            before, after = reference.metrics(), projected.metrics()
            old_weights, new_weights = reference.associations(), projected.associations()
            report["comparisons"][split] = {"source": before, "block_values": after,
                "max_association_difference": float((old_weights - new_weights).abs().max()),
                "same_predictions": bool(torch.equal(old_weights.argmax(-1), new_weights.argmax(-1)))}
            print(f"{split}: source accuracy {before['cue_accuracy']:.4%}, "
                  f"block accuracy {after['cue_accuracy']:.4%}, "
                  f"loss {before['total_loss']:.8f} -> {after['total_loss']:.8f}", flush=True)
        weights = projected.associations()[:min(16, len(targets))]
        encoded = model.value_projection.encode(memories)
        decoded_mixture = model.value_projection.decode(weights.to(encoded) @ encoded)
        reference_mixture = weights.double() @ memories.double()
        report["max_soft_mixture_roundtrip_error"] = float((decoded_mixture - reference_mixture).abs().max())
    save_torch(attached, destination)
    output_file(args.output_prefix, "_value_control.json").write_text(json.dumps(report, indent=2) + "\n")
    backup(args, "_checkpoint.pt")
    backup(args, "_value_control.json")
    print(f"Saved block-value checkpoint: {destination}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cue-cache", required=True)
    parser.add_argument("--seed-cache")
    parser.add_argument("--output-prefix", default="block_values")
    parser.add_argument("--backup-prefix")
    parser.add_argument("--value-block-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.value_block_size, args.threads) < 1:
        parser.error("Block size and thread count must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
