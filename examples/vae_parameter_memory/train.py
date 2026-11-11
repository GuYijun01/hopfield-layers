"""Train ten independent VAE0 memories and a parallel Hopfield retriever."""
import argparse
import copy
import json
from pathlib import Path
import torch

from .cues import generate_block_cues
from .data import digit_split, load_mnist, make_loader
from .hopfield import make_retriever, train_retriever
from .training import train_vae0
from .vae import VAE, parameter_vector


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dataset = load_mnist(args.data_dir, train=True)
    initial = VAE()  # The identical initialization is used for every VAE0/VAE2.
    initial_state = copy.deepcopy(initial.state_dict())
    dimension = parameter_vector(initial).numel()
    hopfield = make_retriever(dimension, args.feature_size, args.association_size).to(device)
    memories, cues, targets, blocks, main_cues, summaries = [], [], [], [], [], []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"config": vars(args), "parameter_count": dimension, "digits": summaries,
              "torch_version": str(torch.__version__),
              "protocol": "Ten independent digit-specific VAE0 experts from shared initialization. "
                          "VAE2 variants use one pass over 50-batch training blocks, resetting parameters "
                          "and optimizer per block. Hopfield replay learns variant-to-best-memory pairs. "
                          "Test images and test labels are excluded from this training."}
    for digit in range(10):
        train_indices, val_indices = digit_split(dataset, digit, args.seed + digit, args.train_limit)
        if len(train_indices) < 50 * args.batch_size or len(val_indices) == 0:
            raise ValueError("Each digit requires >=50*batch_size training images and validation data")
        train = make_loader(dataset, train_indices, args.batch_size, args.seed + digit)
        validation = make_loader(dataset, val_indices, args.batch_size, args.seed, shuffle=False)
        memory, main_cue, summary = train_vae0(
            initial_state, train, validation, args.epochs, args.lr, device, seed=args.seed + digit)
        variants, block_indices = generate_block_cues(
            initial_state, dataset, train_indices, args.batch_size, args.lr, device,
            args.seed + 1000 + digit * 100, args.cue_passes)
        memories.append(memory)
        main_cues.append(main_cue)
        cues.append(variants)
        targets.append(torch.full((len(variants),), digit, dtype=torch.long, device=device))
        blocks.append(block_indices)
        bank, cue_bank, target_bank = torch.stack(memories), torch.cat(cues), torch.cat(targets)
        retrieval = train_retriever(hopfield, cue_bank, target_bank, bank,
                                    args.hopfield_steps, args.hopfield_lr)
        summary.update({"digit": digit, "train_examples": len(train_indices),
                        "validation_examples": len(val_indices), "cue_variants": len(variants),
                        "examples_per_block": 50 * args.batch_size, "batches_per_cue": 50,
                        "hopfield_training": retrieval})
        summaries.append(summary)
        torch.save({"config": vars(args), "initial_vae": initial_state,
                    "hopfield": hopfield.state_dict(), "parameter_count": dimension,
                    "memories": bank.cpu(), "cues": cue_bank.cpu(), "targets": target_bank.cpu(),
                    "main_first50_cues": torch.stack(main_cues).cpu(),
                    "block_indices": blocks, "digits": list(range(digit + 1))}, output / "checkpoint.pt")
        (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="work/data")
    parser.add_argument("--output-dir", default="work/vae_parameter_memory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cue-passes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hopfield-steps", type=int, default=200)
    parser.add_argument("--hopfield-lr", type=float, default=1e-3)
    parser.add_argument("--feature-size", type=int, default=2048)
    parser.add_argument("--association-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=0,
                        help="Optional per-digit cap before 90/10 splitting; 0 uses all data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.cue_passes, args.hopfield_steps,
           args.feature_size, args.association_size, args.threads, args.lr, args.hopfield_lr) <= 0:
        parser.error("Counts, dimensions and learning rates must be positive")
    if args.train_limit < 0:
        parser.error("train-limit must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
