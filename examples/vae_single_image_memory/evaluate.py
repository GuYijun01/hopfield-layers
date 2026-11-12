"""Evaluate independent single-test-image, five-update VAE cues."""
import argparse
import json
from pathlib import Path
import torch
from .augment import augment_image, augmentation_config
from .cues import fit_block
from examples.vae_parameter_memory.data import load_mnist
from examples.vae_parameter_memory.hopfield import (
    checkpoint_projection_mode, retriever_from_checkpoint, retrieve_parameters,
)
from examples.vae_parameter_memory.vae import initial_vector
from examples.vae_parameter_memory.evaluate import reconstruction_bce


def validate_checkpoint(checkpoint):
    checkpoint_projection_mode(checkpoint)
    if (checkpoint.get("experiment") != "single_image_5update_v1"
            or checkpoint.get("cue_representation") != "parameter_delta_v1"
            or checkpoint["config"]["batch_size"] != 16):
        raise ValueError("Requires a checkpoint trained by vae_single_image_memory.train")
    expected = augmentation_config(checkpoint.get("augmentation_profile", "legacy"))
    if checkpoint.get("augmentation_config", expected) != expected:
        raise ValueError("Checkpoint augmentation settings do not match its saved profile")
    if checkpoint.get("augmentation_profile") == "balanced80_resampled_v1":
        from .balanced_cues import SAMPLING
        if checkpoint.get("cue_sampling") != SAMPLING:
            raise ValueError("Balanced checkpoint must use fresh permutation sampling per update")


def predict_image(image, checkpoint, hopfield, device, seed):
    """No label enters augmentation, adaptation or retrieval."""
    validate_checkpoint(checkpoint)
    if checkpoint.get("augmentation_profile") == "balanced80_resampled_v1":
        from .balanced_cues import fit_image_cue
        cue = fit_image_cue(checkpoint["initial_vae"], image, checkpoint["config"]["lr"], device, seed)
    else:
        augmented = augment_image(image, profile=checkpoint.get("augmentation_profile", "legacy"), seed=seed).to(device)
        cue = fit_block(checkpoint["initial_vae"], augmented.split(16),
                        checkpoint["config"]["lr"], device, seed)
    hopfield.eval()
    with torch.no_grad():
        retrieved, weights = retrieve_parameters(
            hopfield, cue.unsqueeze(0), checkpoint["memories"].to(device),
            initial_vector(checkpoint["initial_vae"]).to(device))
    selected = weights.argmax(-1).item()
    prediction = {"predicted_digit": checkpoint["digits"][selected],
                  "selected_memory": selected, "associations": weights[0].cpu().tolist(),
                  "optimizer_updates": 5, "augmented_images": 80}
    if checkpoint.get("augmentation_profile") == "balanced80_resampled_v1":
        from .balanced_cues import update_indices
        prediction["distinct_augmented_views_used"] = len(update_indices(seed).unique())
    return prediction, cue.cpu(), retrieved[0].cpu()


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    if checkpoint["digits"] != list(range(10)):
        raise ValueError("Evaluation requires all ten memories")
    config = checkpoint["config"]
    hopfield = retriever_from_checkpoint(checkpoint, args.device)
    dataset = load_mnist(config["data_dir"], train=False)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))
    if args.test_limit:
        order = order[:args.test_limit]
    prefix = getattr(args, "output_prefix", None)
    if prefix:
        output = Path(prefix).parent
        if not output.is_dir():
            raise ValueError(f"Output prefix must use an existing directory: {output}")
        report_path = Path(str(prefix) + "_evaluation.json")
    else:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "evaluation.json"
    rows, confusion = [], torch.zeros(10, 10, dtype=torch.long)
    for position, index in enumerate(order.tolist()):
        image, label = dataset[index]
        result, cue, retrieved = predict_image(image, checkpoint, hopfield, args.device,
                                               args.seed + position)
        digit = int(label)
        result.update({"test_index": index, "true_digit": digit,
                       "target_parameter_rmse": (retrieved - checkpoint["memories"][digit]).square().mean().sqrt().item(),
                       "posterior_mean_source_reconstruction_bce": reconstruction_bce(
                           retrieved, image.reshape(1, 784), args.device)})
        confusion[digit, result["predicted_digit"]] += 1
        rows.append(result)
        if args.save_cues:
            # Stream artifacts to disk rather than retaining ~12 GB for 10,000 images.
            torch.save({"delta_cue": cue, "retrieved_parameters": retrieved,
                        "test_index": index}, Path(str(prefix) + f"_test_cue_{index:05d}.pt")
                       if prefix else output / f"test_cue_{index:05d}.pt")
        print(json.dumps(result), flush=True)
    report = {"config": vars(args), "training_config": config,
              "experiment": "single_image_5update_v1",
              "projection_mode": checkpoint_projection_mode(checkpoint),
              "hopfield_cue_protocol": checkpoint.get("hopfield_cue_protocol", "distinct_image_blocks_v1"),
              "augmentation_profile": checkpoint.get("augmentation_profile", "legacy"),
              "augmentation_config": augmentation_config(checkpoint.get("augmentation_profile", "legacy")),
              "protocol": "Each shuffled test image independently generates an 80-view augmentation pool. "
                          "The saved profile defines geometry, view shuffling and per-update sampling. "
                          "Fresh VAE2 and Adam adapt on five batches of sixteen. Labels only score predictions. "
                          "Source-image reconstruction is an adaptation metric, not held-out reconstruction.",
              "accuracy": confusion.diag().sum().item() / len(rows),
              "test_images": len(rows), "confusion_matrix": confusion.tolist(), "predictions": rows}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="work/vae_single_image_memory/checkpoint.pt")
    parser.add_argument("--output-dir", default="work/vae_single_image_memory")
    parser.add_argument("--output-prefix", help="Save <prefix>_evaluation.json in an existing directory; overrides output-dir")
    parser.add_argument("--test-limit", type=int, default=0, help="0 evaluates all 10,000 test images")
    parser.add_argument("--save-cues", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.test_limit < 0 or args.threads < 1:
        parser.error("test-limit must be nonnegative and threads positive")
    return args


if __name__ == "__main__":
    run(parse_args())
