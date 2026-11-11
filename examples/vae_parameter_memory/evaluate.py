"""Retrieve VAE0 memories from 50-batch blocks of held-out MNIST images."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from .cues import fit_block
from .data import load_mnist
from .hopfield import make_retriever, retrieve_parameters
from .vae import VAE, initial_vector, restore_vector


def predict_block(image_batches, checkpoint, hopfield, device, seed):
    """Only image batches enter inference; no digit label or memory index."""
    if checkpoint.get("cue_representation") != "parameter_delta_v1":
        raise ValueError("Checkpoint must use parameter-delta cues; convert legacy checkpoints with refit")
    config = checkpoint["config"]
    cue = fit_block(checkpoint["initial_vae"], image_batches, config["lr"], device, seed)
    memories = checkpoint["memories"].to(device)
    initial = initial_vector(checkpoint["initial_vae"]).to(device)
    hopfield.eval()
    with torch.no_grad():
        retrieved, weights = retrieve_parameters(hopfield, cue.unsqueeze(0), memories, initial)
        selected = weights.argmax(-1).item()
    prediction = {"predicted_digit": checkpoint["digits"][selected], "selected_memory": selected,
                  "associations": weights.squeeze(0).cpu().tolist(), "optimizer_updates": 50}
    return prediction, cue.cpu(), retrieved.squeeze(0).cpu()


@torch.no_grad()
def reconstruction_bce(parameters, images, device):
    if len(images) == 0:
        return None
    model = VAE().to(device)
    restore_vector(model, parameters.to(device))
    logits, _, _ = model(images.to(device), sample=False)
    return F.binary_cross_entropy_with_logits(logits, images.to(device), reduction="none").sum(-1).mean().item()


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["digits"] != list(range(10)):
        raise ValueError("Evaluation requires all ten stored memories")
    if checkpoint.get("cue_representation") != "parameter_delta_v1":
        raise ValueError("Legacy raw-parameter checkpoint: run the refit module first")
    config = checkpoint["config"]
    hopfield = make_retriever(checkpoint["parameter_count"], config["feature_size"],
                              config["association_size"]).to(device)
    hopfield.load_state_dict(checkpoint["hopfield"])
    dataset = load_mnist(config["data_dir"], train=False)
    block_size = 50 * config["batch_size"]
    generator = torch.Generator().manual_seed(args.seed)
    episodes = []
    for digit in range(10):
        indices = (dataset.targets == digit).nonzero().flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = len(indices) // block_size
        if args.blocks_per_digit:
            count = min(count, args.blocks_per_digit)
        if count == 0:
            raise ValueError(f"Digit {digit} has insufficient test images for 50 full batches")
        used = count * block_size
        query_indices = indices[used:]
        for block in indices[:used].split(block_size):
            episodes.append((digit, block, query_indices))
    order = torch.randperm(len(episodes), generator=generator).tolist()
    rows, cues, retrieved_bank, blocks, query_bank = [], [], [], [], []
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for position, episode_id in enumerate(order):
        digit, block, query_indices = episodes[episode_id]
        batches = (dataset.images[part] for part in block.split(config["batch_size"]))
        prediction, cue, retrieved = predict_block(batches, checkpoint, hopfield, device, args.seed + position)
        # Only the evaluator knows the block label; it is not supplied to inference.
        target = checkpoint["memories"][digit]
        prediction.update({"true_digit": digit, "adaptation_examples": len(block),
                           "query_examples": len(query_indices),
                           "target_parameter_rmse": (retrieved - target).square().mean().sqrt().item(),
                           "posterior_mean_query_reconstruction_bce": reconstruction_bce(
                               retrieved, dataset.images[query_indices], device),
                           "test_block_indices": block.tolist(), "query_indices": query_indices.tolist()})
        confusion[digit, prediction["predicted_digit"]] += 1
        rows.append(prediction)
        cues.append(cue)
        retrieved_bank.append(retrieved)
        blocks.append(block)
        query_bank.append(query_indices)
        print(json.dumps({k: prediction[k] for k in ("true_digit", "predicted_digit", "target_parameter_rmse")}), flush=True)
    report = {"config": vars(args), "training_config": config, "cue_representation": "parameter_delta_v1",
              "protocol": "Digit-homogeneous test blocks presented in shuffled order. Each block is "
                          "50 full minibatches, used once from shared VAE2 initialization and fresh Adam. "
                          "No labels enter cue construction or Hopfield. Labels construct benchmark blocks "
                          "and score retrieval. This is block-level memory identification, not per-image "
                          "classification in an arbitrary mixed-digit block. Test data never trains Hopfield.",
              "accuracy": confusion.diag().sum().item() / len(rows),
              "per_digit_accuracy": (confusion.diag() / confusion.sum(1)).tolist(),
              "confusion_matrix": confusion.tolist(), "blocks": len(rows),
              "adaptation_images": sum(len(block) for block in blocks), "predictions": rows}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"delta_cues": torch.stack(cues), "retrieved_parameters": torch.stack(retrieved_bank),
                "test_block_indices": blocks, "query_indices": query_bank,
                "true_digits": [row["true_digit"] for row in rows]}, output / "test_cues.pt")
    print(json.dumps({"block_accuracy": report["accuracy"], "blocks": len(rows)}), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="work/vae_parameter_memory_delta/checkpoint.pt")
    parser.add_argument("--output-dir", default="work/vae_parameter_memory_delta")
    parser.add_argument("--blocks-per-digit", type=int, default=0,
                        help="Maximum nonoverlapping 50-batch test blocks per digit; 0 uses all full blocks")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.blocks_per_digit < 0 or args.threads < 1:
        parser.error("blocks-per-digit must be nonnegative and threads positive")
    return args


if __name__ == "__main__":
    run(parse_args())
