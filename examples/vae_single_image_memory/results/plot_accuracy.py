"""Plot per-image retrieval accuracy from an experiment-2 evaluation JSON."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(results_path, output_path=None):
    results_path = Path(results_path)
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Evaluation file not found: {results_path}. The evaluator writes "
            "evaluation.json inside its --output-dir. Pass that file's actual path."
        )
    with results_path.open() as handle:
        results = json.load(handle)
    cm = np.asarray(results["confusion_matrix"], dtype=float)
    if (cm.shape != (10, 10) or not np.isfinite(cm).all()
            or (cm < 0).any() or (cm != np.floor(cm)).any()):
        raise ValueError("Expected a 10x10 confusion matrix of nonnegative integer counts")
    cm = cm.astype(np.int64)
    counts = cm.sum(axis=1)
    total = int(cm.sum())
    if total == 0:
        raise ValueError("No evaluated images found")
    correct = int(np.trace(cm))
    accuracy = correct / total
    percentages = 100 * np.divide(
        cm, counts[:, None], out=np.zeros_like(cm, dtype=float),
        where=counts[:, None] != 0,
    )

    fig, (ax_cm, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 6), constrained_layout=True,
    )
    fig.suptitle(
        f"Single-image VAE memory retrieval — accuracy: {accuracy:.2%} "
        f"({correct}/{total} images)", fontsize=15,
    )
    im = ax_cm.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    for true in range(10):
        for predicted in range(10):
            value = percentages[true, predicted]
            if cm[true, predicted]:
                ax_cm.text(
                    predicted, true, f"{value:.0f}%\n({cm[true, predicted]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if value > 50 else "black",
                )
        if counts[true] == 0:
            ax_cm.text(4.5, true, "No samples", ha="center", va="center", color="gray")
    ax_cm.set(title="Confusion matrix", xlabel="Predicted digit", ylabel="True digit",
              xticks=range(10), yticks=range(10))
    fig.colorbar(im, ax=ax_cm, label="% within each true digit", shrink=0.8)

    per_digit = np.diag(percentages)
    bars = ax_bar.bar(range(10), per_digit, color="steelblue")
    for digit, bar in enumerate(bars):
        label = f"{per_digit[digit]:.1f}%" if counts[digit] else "N/A"
        ax_bar.text(digit, bar.get_height() + 1, label, ha="center", fontsize=9)
    ax_bar.axhline(accuracy * 100, color="darkorange", linestyle="--",
                   label=f"Overall: {accuracy:.1%}")
    ax_bar.set(
        title="Accuracy per true digit (recall)", xlabel="Digit (number of test images)",
        ylabel="Correct predictions (%)", ylim=(0, 112), xticks=range(10),
        xticklabels=[f"{digit}\n(n={counts[digit]})" for digit in range(10)],
        yticks=range(0, 101, 20),
    )
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_axisbelow(True)
    ax_bar.legend()

    output_path = Path(output_path) if output_path else results_path.with_name(
        results_path.stem + "_accuracy.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Saved plot: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Path to the saved evaluation JSON")
    parser.add_argument("--output", type=Path, help="PNG output path; default: <results_stem>_accuracy.png")
    args = parser.parse_args()
    plot_accuracy(args.results, args.output)
