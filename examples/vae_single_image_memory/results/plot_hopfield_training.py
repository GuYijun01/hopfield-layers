"""Plot train/validation learning curves for a recorded Hopfield replay stage."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training(report_path, output_path=None, stage=10):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    matches = [s for s in report.get("stages", []) if s["digits_seen"] == stage]
    if not matches or not matches[0].get("history"):
        raise ValueError(f"No optimizer-step history for stage {stage}. "
                         "Run examples.vae_single_image_memory.diagnose_hopfield first.")
    data = matches[0]
    history = data["history"]
    steps = [row["step"] for row in history]
    panels = [("total_loss", "Total objective: CE + normalized parameter MSE"),
              ("association_loss", "Association cross entropy"),
              ("normalized_parameter_mse", "Normalized parameter MSE"),
              ("cue_accuracy", "Correct-memory retrieval accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for split, color in [("train", "steelblue"), ("validation", "darkorange")]:
            if all(split in row for row in history):
                multiplier = 100 if key == "cue_accuracy" else 1
                ax.plot(steps, [row[split][key] * multiplier for row in history],
                        label=split.capitalize(), color=color)
        ax.set(title=title, xlabel="Hopfield optimizer steps in this stage",
               ylabel="Accuracy (%)" if key == "cue_accuracy" else "Loss")
        if key == "cue_accuracy":
            ax.set_ylim(-2, 102)
            ax.axhline(100 / stage, linestyle=":", color="gray", label="Uniform-guess reference")
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        f"Hopfield projection refit — {stage} memories\n"
        f"{data['train_cues']} training cues, {data['validation_cues']} validation cues; "
        "experts and cue protocol fixed",
    )
    output_path = Path(output_path) if output_path else report_path.with_name(
        f"hopfield_stage_{stage:02d}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    print("Final refit metrics: " + json.dumps(history[-1]))
    print("Exact original saved model: " + json.dumps(report["frozen_original"]))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--stage", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_training(args.report, args.output, args.stage)
