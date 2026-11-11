"""Reuse trained VAE0 experts/cached training blocks; refit Hopfield on deltas.

No test examples enter conversion or refitting. This avoids retraining VAE0
because changing the cue representation does not change its stored memories.
"""
import argparse
import json
from pathlib import Path
import torch
from .hopfield import make_retriever, train_retriever
from .vae import VAE, initial_vector


def run(args):
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    torch.manual_seed(config["seed"])
    VAE()  # Reproduce the initialization order used by train.py.
    initial = initial_vector(checkpoint["initial_vae"])
    representation = checkpoint.get("cue_representation", "raw_parameters")
    if representation == "raw_parameters":
        checkpoint["cues"] = checkpoint["cues"] - initial
        checkpoint["main_first50_cues"] = checkpoint["main_first50_cues"] - initial
    elif representation != "parameter_delta_v1":
        raise ValueError("Unrecognized cue representation")
    hopfield = make_retriever(checkpoint["parameter_count"], config["feature_size"], config["association_size"])
    stages = []
    for digit in checkpoint["digits"]:
        mask = checkpoint["targets"] <= digit
        scores = train_retriever(hopfield, checkpoint["cues"][mask], checkpoint["targets"][mask],
                                  checkpoint["memories"][:digit + 1], config["hopfield_steps"],
                                  config["hopfield_lr"], initial)
        stages.append({"after_digit": digit, **scores})
        print(json.dumps(stages[-1]), flush=True)
    checkpoint.update({"hopfield": hopfield.state_dict(), "cue_representation": "parameter_delta_v1",
                       "source_checkpoint": args.source})
    checkpoint["config"] = {**config, "output_dir": args.output_dir}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "checkpoint.pt")
    report = {"config": checkpoint["config"], "source_checkpoint": args.source,
              "cue_representation": "parameter_delta_v1", "parameter_count": checkpoint["parameter_count"],
              "memories": len(checkpoint["memories"]), "training_cues": len(checkpoint["cues"]),
              "variants_per_digit": checkpoint["targets"].bincount().tolist(), "stages": stages,
              "protocol": "Reused unchanged digit-specific VAE0 memories and cached training-block "
                          "VAE2 parameters; subtracted shared initialization to obtain parameter-delta cues. "
                          "Reinitialized Hopfield and trained sequentially with cue replay. "
                          "All keys use final-minus-initial deltas; outputs restore absolute parameters. "
                          "No test data or labels enter training or refitting."}
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="work/vae_parameter_memory/checkpoint.pt")
    parser.add_argument("--output-dir", default="work/vae_parameter_memory_delta")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    run(args)
