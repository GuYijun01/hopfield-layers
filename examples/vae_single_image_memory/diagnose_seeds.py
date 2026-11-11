"""Measure seed sensitivity on identical training images with frozen Hopfield weights."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .augment import augment_image, augmentation_config
from .cues import fit_block
from .evaluate import validate_checkpoint
from examples.vae_parameter_memory.data import digit_split, load_mnist
from examples.vae_parameter_memory.hopfield import make_retriever, retrieve_parameters
from examples.vae_parameter_memory.vae import initial_vector


def original_cue_seeds(checkpoint):
    """Older augmented checkpoints predate explicit seed metadata."""
    targets = checkpoint["targets"].cpu()
    if "cue_seeds" in checkpoint:
        seeds = checkpoint["cue_seeds"].cpu()
        if seeds.shape != targets.shape:
            raise ValueError("Saved cue seeds do not align with targets")
        return seeds
    counters, seeds = {}, []
    for digit in targets.tolist():
        position = counters.get(digit, 0)
        seeds.append(checkpoint["config"]["seed"] + 1000 + digit * 10000 + position)
        counters[digit] = position + 1
    return torch.tensor(seeds, dtype=torch.long)


def regenerate_cues(initial_state, dataset, source_indices, original_seeds, lr, device,
                    profile, *, offset=0, vary="both"):
    """Preserve the stored source order; change only the requested episode seeds."""
    if vary not in ("both", "shuffle", "latent"):
        raise ValueError("vary must be both, shuffle or latent")
    cues = []
    for index, seed in zip(source_indices.tolist(), original_seeds.tolist()):
        shuffle_seed = seed + offset if vary in ("both", "shuffle") else seed
        latent_seed = seed + offset if vary in ("both", "latent") else seed
        image = dataset[index][0]
        batches = augment_image(image, profile=profile, seed=shuffle_seed).to(device).split(16)
        cues.append(fit_block(initial_state, batches, lr, device, latent_seed).cpu())
    return torch.stack(cues)


@torch.no_grad()
def score_cues(hopfield, cues, targets, memories, initial, source_indices, batch_size=64,
               reference_cues=None, reference_predictions=None):
    """Chunked evaluation of the same objective; each cue contributes equal weight."""
    rows = []
    variance = memories.var(unbiased=False).clamp_min(1e-8)
    for start in range(0, len(cues), batch_size):
        stop = min(start + batch_size, len(cues))
        batch = cues[start:stop].to(memories.device)
        labels = targets[start:stop].to(memories.device)
        retrieved, weights = retrieve_parameters(hopfield, batch, memories, initial)
        predictions = weights.argmax(-1)
        association = -weights.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12).log()
        reconstruction = (retrieved - memories[labels]).square().mean(-1) / variance
        if reference_cues is not None:
            reference = reference_cues[start:stop].to(memories.device)
            relative_change = (batch - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)
            cosine = F.cosine_similarity(batch, reference, dim=-1).clamp(-1, 1)
        for position in range(stop - start):
            row = {"source_index": int(source_indices[start + position]),
                   "true_digit": int(labels[position]), "predicted_digit": int(predictions[position]),
                   "association_loss": float(association[position]),
                   "normalized_parameter_mse": float(reconstruction[position]),
                   "total_loss": float(association[position] + reconstruction[position]),
                   "correct": bool(predictions[position] == labels[position])}
            if reference_cues is not None:
                row.update({"relative_cue_l2_change": float(relative_change[position]),
                            "cue_cosine_similarity": float(cosine[position])})
            if reference_predictions is not None:
                row["same_prediction_as_cached"] = row["predicted_digit"] == reference_predictions[start + position]
            rows.append(row)
    metrics = {key: sum(row[key] for row in rows) / len(rows)
               for key in ("total_loss", "association_loss", "normalized_parameter_mse")}
    metrics["cue_accuracy"] = sum(row["correct"] for row in rows) / len(rows)
    if reference_cues is not None:
        for key in ("relative_cue_l2_change", "cue_cosine_similarity"):
            metrics[key] = sum(row[key] for row in rows) / len(rows)
    if reference_predictions is not None:
        metrics["prediction_agreement_with_cached"] = sum(row["same_prediction_as_cached"] for row in rows) / len(rows)
    return {"metrics": metrics, "per_image": rows}


def run(args):
    source, output = Path(args.checkpoint).resolve(), Path(args.output_dir).resolve()
    if source.parent == output:
        raise ValueError("Use a separate diagnostic output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    if (checkpoint.get("hopfield_cue_protocol") != "single_image_augmented_v1"
            or checkpoint["digits"] != list(range(10)) or "cue_source_indices" not in checkpoint):
        raise ValueError("Use a completed train_augmented checkpoint with stored training source indices")
    config = checkpoint["config"]
    dataset = load_mnist(config["data_dir"], train=True)
    indices = checkpoint["cue_source_indices"].cpu()
    targets, cached = checkpoint["targets"].cpu(), checkpoint["cues"].cpu()
    if (len(indices) != len(cached) or len(targets) != len(cached)
            or indices.unique().numel() != len(indices) or len(indices) == 0):
        raise ValueError("Stored training cues, sources and targets are not aligned and distinct")
    if not torch.equal(dataset.targets[indices].cpu(), targets):
        raise ValueError("Training source labels do not match the cached cue targets")
    for digit in checkpoint["digits"]:
        train_indices, _ = digit_split(dataset, digit, config["seed"] + digit, config.get("train_limit", 0))
        if not torch.isin(indices[targets == digit], train_indices).all():
            raise ValueError("Stored sources are outside the original training split")
    seeds = original_cue_seeds(checkpoint)
    profile = checkpoint.get("augmentation_profile", "legacy")
    if args.vary == "shuffle" and not augmentation_config(profile)["shuffle"]:
        raise ValueError("This checkpoint's legacy augmentation has no shuffle; choose --vary latent or both")
    hopfield = make_retriever(checkpoint["parameter_count"], config["feature_size"],
                              config["association_size"]).to(args.device)
    hopfield.load_state_dict(checkpoint["hopfield"])
    hopfield.requires_grad_(False)
    hopfield.eval()
    memories = checkpoint["memories"].to(args.device)
    initial = initial_vector(checkpoint["initial_vae"]).to(args.device)
    frozen_weights = {key: value.detach().clone() for key, value in hopfield.state_dict().items()}
    baseline = score_cues(hopfield, cached, targets, memories, initial, indices, args.batch_size)
    baseline.update({"name": "Cached training cues", "seed_offset": None})
    predictions = [row["predicted_digit"] for row in baseline["per_image"]]
    report = {"source_checkpoint": str(source), "config": vars(args), "images": len(indices),
              "augmentation_profile": profile, "augmentation_config": augmentation_config(profile),
              "seed_source": "saved cue_seeds" if "cue_seeds" in checkpoint else "reconstructed train_augmented_v1 seed schedule",
              "original_seeds": seeds.tolist(), "source_indices": indices.tolist(),
              "protocol": "Frozen saved Hopfield projection and expert bank. Identical training source images "
                          "in identical order for every condition; only cue-generation seeds change. Cached cues "
                          "are the reference. Same-seed replay checks reproducibility; new-seed runs are paired "
                          "seed-sensitivity measurements, not training epochs or held-out validation. Each cue "
                          "still uses eighty views and five updates. No Hopfield updates or test data.",
              "cases": [baseline]}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "seed_stability.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(baseline["name"] + ": " + json.dumps(baseline["metrics"]), flush=True)
    conditions = [("Same-seed replay", 0)] + [
        (f"New seeds {run_id}", args.seed_offset * run_id) for run_id in range(1, args.runs + 1)]
    for name, offset in conditions:
        print(f"{name}: regenerating {len(indices)} cues", flush=True)
        cues = regenerate_cues(checkpoint["initial_vae"], dataset, indices, seeds, config["lr"],
                               args.device, profile, offset=offset, vary=args.vary)
        result = score_cues(hopfield, cues, targets, memories, initial, indices, args.batch_size,
                            reference_cues=cached, reference_predictions=predictions)
        result.update({"name": name, "seed_offset": offset})
        result["metrics"]["total_loss_change_from_cached"] = result["metrics"]["total_loss"] - baseline["metrics"]["total_loss"]
        report["cases"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(name + ": " + json.dumps(result["metrics"]), flush=True)
        del cues
    if not all(torch.equal(value, hopfield.state_dict()[key]) for key, value in frozen_weights.items()):
        raise RuntimeError("Diagnostic unexpectedly changed Hopfield state")
    print(f"Saved report: {report_path}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="work/vae_single_image_augmented_mild/checkpoint.pt")
    parser.add_argument("--output-dir", default="work/vae_single_image_augmented_mild/seed_diagnostics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64, help="Hopfield scoring chunk size; no optimizer updates")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=10000000)
    parser.add_argument("--vary", choices=("both", "shuffle", "latent"), default="both")
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.runs, args.seed_offset) < 1:
        parser.error("Counts and seed offset must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
