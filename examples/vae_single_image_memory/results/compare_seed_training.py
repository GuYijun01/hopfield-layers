"""Compare source and multi-seed models on the identical recorded cue banks."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def comparison_data(report):
    if report.get("seed_variants", 1) < 2:
        raise ValueError("Requires the training report from a multi-seed run")
    stages = report.get("stages", [])
    if not stages or stages[-1]["digits_seen"] != 10:
        raise ValueError("Wait until all ten digit stages have finished")
    stage = stages[-1]
    history = stage.get("history", [])
    if not history or history[-1]["step"] != stage["steps"]:
        raise ValueError("The final Hopfield stage has not finished")
    baseline, final = report["frozen_original"], history[-1]
    conditions = [("train", "Training cue bank", report["train_cues"]),
                  ("heldout_seed", "Same images, held-out seed", report["heldout_seed_cues"]),
                  ("validation", "Unseen validation images", report["validation_cues"])]
    rows = []
    metric_names = ("cue_accuracy", "total_loss", "association_loss", "normalized_parameter_mse")
    for key, label, count in conditions:
        if key not in baseline or key not in final:
            raise ValueError(f"Report is missing matched source/new metrics for {key}")
        old = {metric: float(baseline[key][metric]) for metric in metric_names}
        new = {metric: float(final[key][metric]) for metric in metric_names}
        if not all(math.isfinite(value) for metrics in (old, new) for value in metrics.values()):
            raise ValueError("Comparison metrics must be finite")
        rows.append({"condition": key, "label": label, "cues": count,
                     "source": old, "multi_seed": new,
                     "accuracy_change_pp": 100 * (new["cue_accuracy"] - old["cue_accuracy"]),
                     "total_loss_change": new["total_loss"] - old["total_loss"]})
    config = report["training_config"]
    source_steps, source_lr = config["hopfield_steps"], config["hopfield_lr"]
    drop, final_lr = config.get("lr_drop_after"), config.get("hopfield_lr_final")
    source_schedule = [{"start_step": 1, "end_step": source_steps, "lr": source_lr}]
    if drop is not None and final_lr is not None:
        source_schedule = [{"start_step": 1, "end_step": drop, "lr": source_lr},
                           {"start_step": drop + 1, "end_step": source_steps, "lr": final_lr}]
    return {"source_checkpoint": report["source_checkpoint"],
            "source_seed_variants": config.get("seed_variants", 1),
            "new_seed_variants": report["seed_variants"], "training_images": report["training_images"],
            "source_projection_mode": config.get("projection_mode", "shared"),
            "new_projection_mode": report["projection_mode"],
            "matched_projection_mode": config.get("projection_mode", "shared") == report["projection_mode"],
            "matched_lr_schedule": source_schedule == report["lr_schedule"],
            "source_value_projection": config.get("value_projection", "identity"),
            "new_value_projection": report.get("value_projection", "identity"),
            "matched_value_projection": config.get("value_projection", "identity") == report.get("value_projection", "identity"),
            "lr_schedule": report["lr_schedule"],
            "source_cached_training_reference": baseline.get("cached_training"),
            "comparisons": rows,
            "interpretation": "Both models are scored on exactly the same cue banks. The training bank "
                              "includes new training seed variants the source model did not optimize on. "
                              "Use held-out-seed and unseen-image changes to assess generalization. "
                              "These are paired comparisons, not independent new handwriting samples."}


def compare_seed_training(report_path, output_prefix=None):
    report_path = Path(report_path)
    summary = comparison_data(json.loads(report_path.read_text()))
    prefix = Path(output_prefix) if output_prefix else report_path.with_name(
        report_path.stem.removesuffix("_training") + "_seed_comparison")
    if not prefix.parent.is_dir():
        raise ValueError(f"Output prefix must use an existing directory: {prefix.parent}")
    rows = summary["comparisons"]
    print(f"{'Cue bank':<30} {'Source acc':>11} {'New acc':>11} {'Delta pp':>10} {'Source loss':>13} {'New loss':>11}")
    for row in rows:
        print(f"{row['label']:<30} {row['source']['cue_accuracy']:>10.2%} "
              f"{row['multi_seed']['cue_accuracy']:>10.2%} {row['accuracy_change_pp']:>+10.2f} "
              f"{row['source']['total_loss']:>13.5f} {row['multi_seed']['total_loss']:>11.5f}")
    if not (summary["matched_projection_mode"] and summary["matched_lr_schedule"] and summary["matched_value_projection"]):
        print("Projection mode, value map or LR schedule changed; this comparison does not isolate seed training alone.")
    labels = ["Training cues\n(all training seeds)", "Same training images\nheld-out seed", "Unseen validation\nimages"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title, multiplier in ((axes[0], "cue_accuracy", "Retrieval accuracy", 100),
                                          (axes[1], "total_loss", "Total objective", 1)):
        values = []
        for offset, key, label, color in ((-0.2, "source", "Source model", "steelblue"),
                                           (0.2, "multi_seed", "New model", "darkorange")):
            heights = [row[key][metric] * multiplier for row in rows]
            values.extend(heights)
            positions = [index + offset for index in range(3)]
            ax.bar(positions, heights, width=0.38, label=label, color=color)
            for x, value in zip(positions, heights):
                ax.annotate(f"{value:.1f}%" if multiplier == 100 else f"{value:.3g}",
                            (x, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=title, xticks=range(3), xticklabels=labels,
               ylabel="Accuracy (%)" if multiplier == 100 else "Loss")
        ax.set_ylim(0, 110 if multiplier == 100 else max(max(values) * 1.18, 0.01))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend()
    fig.suptitle(f"Model comparison on identical cue banks — {summary['training_images']} training images")
    json_path, plot_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".png")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary: {json_path}")
    print(f"Saved plot: {plot_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Completed multi-seed training JSON")
    parser.add_argument("--output-prefix", type=Path, help="Default: <run_prefix>_seed_comparison")
    args = parser.parse_args()
    compare_seed_training(args.report, args.output_prefix)
