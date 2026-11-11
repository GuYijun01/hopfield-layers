"""Retrieve a VAE0 for each randomly mixed, unknown MNIST test image."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from .cues import image_cue
from .data import load_mnist
from .hopfield import make_retriever
from .vae import VAE, restore_vector


def predict_image(image, checkpoint, hopfield, device, seed):
    """No digit label enters cue construction or memory selection."""
    config = checkpoint["config"]
    cue = image_cue(checkpoint["initial_vae"], image, config["batch_size"],
                    config["lr"], device, seed)
    memories = checkpoint["memories"].to(device)
    hopfield.eval()
    with torch.no_grad():
        retrieved, weights = hopfield(cue.unsqueeze(0), memories, return_associations=True)
        selected = weights.argmax(-1).item()
        model = VAE().to(device)
        restore_vector(model, retrieved.squeeze(0))
        model.eval()
        logits, _, _ = model(image.to(device).reshape(1, 784), sample=False)
        reconstruction = logits.sigmoid().squeeze(0).cpu()
        bce = F.binary_cross_entropy_with_logits(
            logits, image.to(device).reshape(1, 784), reduction="sum").item()
    return {"predicted_digit": checkpoint["digits"][selected], "selected_memory": selected,
            "associations": weights.squeeze(0).cpu().tolist(),
            "posterior_mean_reconstruction_bce": bce}, reconstruction


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["digits"] != list(range(10)):
        raise ValueError("Unknown 0–9 evaluation requires all ten stored memories")
    config = checkpoint["config"]
    hopfield = make_retriever(checkpoint["parameter_count"], config["feature_size"],
                              config["association_size"]).to(device)
    hopfield.load_state_dict(checkpoint["hopfield"])
    dataset = load_mnist(config["data_dir"], train=False)
    generator = torch.Generator().manual_seed(args.seed)
    selected = []
    for digit in range(10):
        indices = (dataset.targets == digit).nonzero().flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        selected.append(indices[:args.test_per_digit] if args.test_per_digit else indices)
    selected = torch.cat(selected)
    selected = selected[torch.randperm(len(selected), generator=generator)]
    rows, reconstructions = [], []
    confusion = torch.zeros(10, 10, dtype=torch.long)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for position, index in enumerate(selected.tolist()):
        image, label = dataset[index]
        prediction, reconstructed = predict_image(image, checkpoint, hopfield, device, args.seed + position)
        # Labels are read only for scoring, after prediction; no labels go to VAE2 or Hopfield.
        prediction.update({"test_index": index, "true_digit": int(label)})
        rows.append(prediction)
        reconstructions.append(reconstructed)
        confusion[int(label), prediction["predicted_digit"]] += 1
        if (position + 1) % 10 == 0:
            print(f"Evaluated {position + 1}/{len(selected)} images", flush=True)
    report = {"config": vars(args), "training_config": config,
              "protocol": "Randomly mixed test digits 0–9; independent unlabeled single-image "
                          "VAE2 adaptation for exactly 50 updates; no task labels or class masks. "
                          "Digit prediction is the selected memory label. Training blocks contain "
                          "multiple images, so single-image test adaptation is a distribution shift.",
              "accuracy": confusion.diag().sum().item() / len(selected),
              "per_digit_accuracy": (confusion.diag() / confusion.sum(1)).tolist(),
              "confusion_matrix": confusion.tolist(),
              "mean_posterior_mean_reconstruction_bce": sum(
                  row["posterior_mean_reconstruction_bce"] for row in rows) / len(rows),
              "predictions": rows}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"test_indices": selected, "reconstructions": torch.stack(reconstructions)},
               output / "reconstructions.pt")
    print(json.dumps({"accuracy": report["accuracy"],
                      "per_digit_accuracy": report["per_digit_accuracy"]}), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="work/vae_parameter_memory/checkpoint.pt")
    parser.add_argument("--output-dir", default="work/vae_parameter_memory")
    parser.add_argument("--test-per-digit", type=int, default=10,
                        help="Balanced random test subset; 0 evaluates all 10000 images (expensive)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.test_per_digit < 0 or args.threads < 1:
        parser.error("test-per-digit must be nonnegative and threads positive")
    return args


if __name__ == "__main__":
    run(parse_args())
