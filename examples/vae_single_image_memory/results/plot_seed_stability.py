"""Compare frozen Hopfield losses across cue seeds for identical training images."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_stability(report_path, output_path=None):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    cases = report["cases"]
    labels = [case["name"].replace(" training cues", "").replace(" ", "\n", 1) for case in cases]
    colors = ["steelblue", "seagreen"] + ["darkorange"] * (len(cases) - 2)
    panels = [("total_loss", "Total objective"), ("association_loss", "Association cross entropy"),
              ("normalized_parameter_mse", "Normalized parameter MSE"), ("cue_accuracy", "Retrieval accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels):
        scale = 100 if key == "cue_accuracy" else 1
        values = [case["metrics"][key] * scale for case in cases]
        ax.bar(range(len(cases)), values, color=colors)
        ax.axhline(values[0], linestyle="--", color="steelblue", alpha=0.8)
        ax.set(title=title, ylabel="Accuracy (%)" if scale == 100 else "Loss",
               xlabel="Cue regeneration condition — no Hopfield updates",
               xticks=range(len(cases)), xticklabels=labels)
        if scale == 100:
            ax.set_ylim(0, 112)
        else:
            ax.set_ylim(0, max(max(values) * 1.18, 0.01))
        for position, value in enumerate(values):
            ax.annotate(f"{value:.1f}%" if scale == 100 else f"{value:.3g}",
                        (position, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"Seed stability on {report['images']} identical training images\n"
                 f"Frozen Hopfield; augmentation={report['augmentation_profile']}; varying={report['config']['vary']}")
    output_path = Path(output_path) if output_path else report_path.with_name("seed_stability.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_stability(args.report, args.output)
