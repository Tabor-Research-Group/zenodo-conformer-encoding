"""Native McUtils convergence diagnostics for the complete 144-run R/S study.

All figures use the immutable pooled histories. Bands are sample SD across
three matched training seeds, never molecule-level standard errors. The
labelled LR example and retained loss outlier are descriptive diagnostics;
neither selects experiments or checkpoints using held-out test performance.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

FACTORS = ("scope", "precision", "lr_profile", "feature_mode", "readout")
EXAMPLE = dict(scope="complete", precision="decoded", feature_mode="full", readout="both")
OUTLIER = dict(scope="complete", precision="psi8", lr_profile="source", feature_mode="reduced52", readout="both")
DEFAULT_SOURCE = Path("/Users/Mark/Documents/UW/Research/Development/McUtils")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def same(row, **values):
    return all(row[key] == value for key, value in values.items())


def configure(output, source):
    # Configure every native cache before importing McUtils/NumPy plotting.
    for key, leaf in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib"),
                      ("XDG_CACHE_HOME", "xdg")):
        folder = output / "runtime-cache" / "convergence" / leaf
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(source.resolve()))
    import McUtils
    from McUtils.Plots import GraphicsGrid, Plot, FilledPlot, PlotLegend, Styled
    require(source.resolve() in Path(McUtils.__file__).resolve().parents,
            "McUtils must come from the requested native source")
    return McUtils, GraphicsGrid, Plot, FilledPlot, PlotLegend, Styled


def load(inputs):
    manifest = json.loads((inputs / "files-sha256.json").read_text())
    for name, expected in manifest.items():
        require(sha(inputs / name) == expected, "Immutable pooled input changed: " + name)
    summary = json.loads((inputs / "summary.json").read_text())
    require(summary["status"] == "complete" and summary["status_counts"] == {"complete": 144},
            "All 144 completed runs must be retained")
    require(not summary["artifact_errors"] and not summary["sampler_mismatches"]
            and not summary["initialization_mismatches"], "Input bindings require review")
    ledger = json.loads((inputs / "run-ledger.json").read_text())
    with (inputs / "long-history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    for row in history:
        for key in ("seed", "epoch", "optimizer_updates"):
            row[key] = int(row[key])
        for key in ("train_accuracy", "validation_accuracy", "train_bce", "validation_bce",
                    "train_auxiliary", "validation_auxiliary", "epoch_seconds"):
            row[key] = float(row[key])
            require(math.isfinite(row[key]), "A logged metric is not finite")
    require(len(history) == 14400 and len(ledger) == 144, "Full 100-epoch histories are required")
    by_key = {}
    for row in history:
        by_key.setdefault(row["key"], []).append(row)
    for record in ledger:
        rows = sorted(by_key[record["key"]], key=lambda r: r["epoch"])
        require([r["epoch"] for r in rows] == list(range(1, 101)), "Incomplete epoch sequence")
        best = max(rows, key=lambda r: r["validation_accuracy"])
        require(best["epoch"] == record["config_summary"]["best_epoch"], "Checkpoint epoch differs")
        require(abs(best["validation_accuracy"] - record["best_so_far"]["validation_accuracy"]) < 1e-12,
                "Best validation accuracy differs from its history")
        require(abs(min(r["validation_bce"] for r in rows) - record["minimum_logged_validation_bce"]) < 1e-12,
                "Minimum BCE differs from history")
    return summary, ledger, history, by_key, manifest


def render(inputs, output, source):
    require(platform.machine() == "arm64", "Use native ARM Python")
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary, ledger, history, by_key, input_hashes = load(inputs)
    McUtils, GraphicsGrid, Plot, FilledPlot, PlotLegend, Styled = configure(output, source)
    import numpy as np
    theme = ("mccoy", {"font": {"family": "DejaVu Sans", "size": 12},
                       "svg": {"fonttype": "path"}})
    products = []

    def save(figure, stem, gaps):
        figure.prep_show()
        # Native figure-backend spacing is independent of data-coordinate units.
        figure.figure.set_figure_spacings(gaps)
        for extension in ("png", "svg", "pdf"):
            path = output / (stem + "." + extension)
            figure.figure.savefig(str(path), format=extension, dpi=190, facecolor=figure.background)
            products.append(path.name)
        figure.close()

    def moment(values):
        require(len(values) == 3, "Every uncertainty band must have three matched seeds")
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                "seed_values": values.tolist(), "n": 3}

    condition_rows = {}
    for r in ledger:
        condition_rows.setdefault(tuple(r["task"][f] for f in FACTORS), []).append(r)
    require(len(condition_rows) == 48, "The 48 scientific conditions must remain explicit")
    diagnostics = []
    for condition, records in sorted(condition_rows.items()):
        records = sorted(records, key=lambda r: r["task"]["seed"])
        require([r["task"]["seed"] for r in records] == [1, 2, 3], "Unmatched condition seeds")
        metrics = {}
        for metric in ("final_train_accuracy", "final_validation_accuracy", "final_train_bce",
                       "final_validation_bce", "test_accuracy"):
            metrics[metric] = moment([r["final_metrics"][metric] for r in records])
        metrics["final_train_minus_validation_accuracy_pp"] = moment([
            100 * (r["final_metrics"]["final_train_accuracy"] - r["final_metrics"]["final_validation_accuracy"])
            for r in records])
        metrics["final_validation_minus_train_bce"] = moment([
            r["final_metrics"]["final_validation_bce"] - r["final_metrics"]["final_train_bce"]
            for r in records])
        late_gains = []
        per_seed = []
        for r in records:
            rows = sorted(by_key[r["key"]], key=lambda a: a["epoch"])
            early = np.mean([a["validation_accuracy"] for a in rows[60:80]])
            late = np.mean([a["validation_accuracy"] for a in rows[80:100]])
            gain = 100 * float(late - early)
            late_gains.append(gain)
            per_seed.append({"seed": r["task"]["seed"], "best_accuracy_epoch": r["config_summary"]["best_epoch"],
                             "minimum_bce_epoch": min(rows, key=lambda a: a["validation_bce"])["epoch"],
                             "late_validation_gain_pp": gain,
                             "first_train_95pct_epoch": next((a["epoch"] for a in rows if a["train_accuracy"] >= .95), None),
                             "first_validation_97pct_epoch": next((a["epoch"] for a in rows if a["validation_accuracy"] >= .97), None),
                             "first_5epoch_validation_90pct_start": next((rows[k]["epoch"] for k in range(96)
                                 if all(a["validation_accuracy"] >= .90 for a in rows[k:k+5])), None),
                             "first_5epoch_validation_95pct_start": next((rows[k]["epoch"] for k in range(96)
                                 if all(a["validation_accuracy"] >= .95 for a in rows[k:k+5])), None)})
        metrics["late_validation_gain_pp"] = moment(late_gains)
        diagnostics.append({"condition": dict(zip(FACTORS, condition)), "metrics": metrics, "per_seed": per_seed})

    # A labelled fixed condition illustrates LR effects without averaging across
    # distinct architectures or choosing a favourable held-out result.
    colors = {"source": "#174C86", "source_lr10": "#C86D24"}
    fig = GraphicsGrid(nrows=2, ncols=2, theme=theme, subimage_size=(560, 315),
                       padding=((95, 35), (85, 160)), spacings=(145, 175),
                       figure_label=Styled("Learning rate and generalization: matched example\n"
                           "Complete · decoded · full descriptors · full readout; mean ± sample SD, n = 3", fontsize=15))
    example_numeric = {}
    for i, metric in enumerate(("accuracy", "bce")):
        for j, split in enumerate(("train", "validation")):
            panel = fig[i, j]
            panel.aspect_ratio = "auto"
            for lr in ("source", "source_lr10"):
                rows = [r for r in history if same(r, **EXAMPLE, lr_profile=lr)]
                series = []
                for epoch in range(1, 101):
                    chosen = sorted([r for r in rows if r["epoch"] == epoch], key=lambda a: a["seed"])
                    require([r["seed"] for r in chosen] == [1, 2, 3], "Example seeds differ")
                    y = np.array([r[split + "_" + metric] for r in chosen]) * (100 if metric == "accuracy" else 1)
                    series.append((float(y.mean()), float(y.std(ddof=1))))
                means, sd = np.asarray(series).T
                FilledPlot(np.arange(1, 101), means + sd, means - sd, figure=panel,
                           color=colors[lr], linewidth=0, plot_style={"alpha": .13})
                Plot(np.arange(1, 101), means, figure=panel, color=colors[lr], linewidth=2.0)
                example_numeric[lr + "_" + split + "_" + metric] = {
                    "epoch": list(range(1, 101)), "mean": means.tolist(), "sample_sd": sd.tolist(), "n": 3}
            panel.axes_labels = ["Epoch", "Accuracy (%)" if metric == "accuracy" else "Binary cross-entropy"]
            panel.plot_range = [(1, 100), (90, 100) if metric == "accuracy" else (-.06, .90)]
            panel.ticks = [[1, 20, 40, 60, 80, 100], [90, 92, 94, 96, 98, 100] if metric == "accuracy" else None]
            panel.plot_label = Styled(("Training" if split == "train" else "Resampled validation") +
                                      (" accuracy (90–100% zoom)" if metric == "accuracy" else " BCE"), fontsize=12)
    fig[0, 0].axes.legend(handles=PlotLegend.construct([
        {"marker": "line", "label": "LR = 5.69 × 10⁻⁴", "color": colors["source"], "lw": 2.5},
        {"marker": "line", "label": "LR = 5.69 × 10⁻⁵", "color": colors["source_lr10"], "lw": 2.5}]),
        loc="upper center", bbox_to_anchor=(1.12, 1.23), ncol=2, frameon=False, fontsize=11)
    save(fig, "learning_rate_convergence_example", (.22, .34))

    # Preserve the entire individual trajectories for delayed/variable learning;
    # a flat early average is not evidence that every seed remains at chance.
    fig = GraphicsGrid(nrows=1, ncols=2, theme=theme, subimage_size=(650, 340),
                       padding=((90, 35), (85, 160)), spacings=(155, 100),
                       figure_label=Styled("Learning onset varies with torsion selection and readout\n"
                           "Full descriptors · decoded geometry · source LR; thin = individual seeds, thick = mean (n = 3)", fontsize=15))
    scope_colors = {"complete": "#071F83", "minimal": "#008000", "augmented": "#D7478A"}
    onset_numeric = []
    for j, readout in enumerate(("both", "z_alpha")):
        panel = fig[0, j]
        panel.aspect_ratio = "auto"
        for scope in ("complete", "minimal", "augmented"):
            rows = [r for r in history if same(r, scope=scope, precision="decoded", lr_profile="source",
                                              feature_mode="full", readout=readout)]
            values = []
            for seed in (1, 2, 3):
                seed_rows = sorted([r for r in rows if r["seed"] == seed], key=lambda a: a["epoch"])
                require(len(seed_rows) == 100, "Every onset trajectory must retain100 epochs")
                y = np.array([r["validation_accuracy"] for r in seed_rows]) * 100
                values.append(y)
                Plot([r["optimizer_updates"] / 1000 for r in seed_rows], y, figure=panel,
                     color=scope_colors[scope], linewidth=.85, plot_style={"alpha": .38})
            values = np.asarray(values)
            Plot(np.arange(1, 101) * 3.442, values.mean(axis=0), figure=panel,
                 color=scope_colors[scope], linewidth=2.1)
            onset_numeric.append({"scope": scope, "readout": readout, "seeds": [1, 2, 3],
                                  "validation_accuracy_percent": values.tolist(),
                                  "epoch": list(range(1, 101)), "mean": values.mean(axis=0).tolist(),
                                  "sample_sd": values.std(axis=0, ddof=1).tolist()})
        panel.axes_labels = ["Optimizer updates (thousands)", "Resampled-validation accuracy (%)"]
        panel.plot_range = [(0, 350), (45, 101)]
        panel.ticks = [[0, 100, 200, 300], [50, 60, 70, 80, 90, 100]]
        panel.plot_label = Styled("Full latent readout" if readout == "both" else "Torsion latent only", fontsize=12)
    fig[0, 0].axes.legend(handles=PlotLegend.construct([
        {"marker": "line", "label": scope.title(), "color": scope_colors[scope], "lw": 2.5}
        for scope in ("complete", "minimal", "augmented")]),
        loc="upper center", bbox_to_anchor=(1.12, 1.28), ncol=3, frameon=False, fontsize=11)
    save(fig, "learning_onset_individual_seeds", (.22, .20))

    # Keep the loss outlier visible as actual individual seed trajectories.
    chosen_runs = sorted([r for r in ledger if same(r["task"], **OUTLIER)], key=lambda r: r["task"]["seed"])
    require(len(chosen_runs) == 3, "All three outlier-condition seeds must be retained")
    target = chosen_runs[2]
    target_rows = sorted(by_key[target["key"]], key=lambda r: r["epoch"])
    best = max(target_rows, key=lambda r: r["validation_accuracy"])
    min_bce = min(target_rows, key=lambda r: r["validation_bce"])
    seed_colors = {1: "#727272", 2: "#29659A", 3: "#C93678"}
    fig = GraphicsGrid(nrows=1, ncols=2, theme=theme, subimage_size=(650, 340),
                       padding=((90, 35), (85, 170)), spacings=(155, 100),
                       figure_label=Styled("Retained confidence-sensitive loss outlier\n"
                           "Complete · 8-bit torsions · reduced descriptors · full readout · source LR", fontsize=15))
    for j, metric in enumerate(("accuracy", "bce")):
        panel = fig[0, j]
        panel.aspect_ratio = "auto"
        for r in chosen_runs:
            seed = r["task"]["seed"]
            rows = sorted(by_key[r["key"]], key=lambda a: a["epoch"])
            for split, pattern in (("train", "-"), ("validation", "--")):
                Plot([a["epoch"] for a in rows],
                     [a[split + "_" + metric] * (100 if metric == "accuracy" else 1) for a in rows],
                     figure=panel, color=seed_colors[seed], linestyle=pattern,
                     linewidth=2.0 if seed == 3 else 1.15, plot_style={"alpha": 1 if seed == 3 else .8})
        scale = 100 if metric == "accuracy" else 1
        for point, marker in ((best, "o"), (min_bce, "D")):
            Plot([point["epoch"]], [scale * point["validation_" + metric]], figure=panel,
                 marker=marker, markersize=7, linestyle="None", color=seed_colors[3])
        panel.axes_labels = ["Epoch", "Accuracy (%)" if metric == "accuracy" else "Binary cross-entropy"]
        panel.plot_range = [(1, 100), (45, 101) if metric == "accuracy" else (0, 2.05)]
        panel.ticks = [[1, 20, 40, 60, 80, 100], [50, 60, 70, 80, 90, 100] if metric == "accuracy" else None]
        panel.plot_label = Styled(("Accuracy" if metric == "accuracy" else "BCE") +
            "\nSeed 3: circle = accuracy-selected epoch 80; diamond = minimum-BCE epoch 44", fontsize=10)
    items = [{"marker": "line", "label": "Seed " + str(seed), "color": seed_colors[seed], "lw": 2}
             for seed in (1, 2, 3)]
    items += [{"marker": "line", "label": "Training", "color": "#333333", "lw": 2, "linestyle": "-"},
              {"marker": "line", "label": "Validation", "color": "#333333", "lw": 2, "linestyle": "--"}]
    fig[0, 0].axes.legend(handles=PlotLegend.construct(items), loc="upper center",
                         bbox_to_anchor=(1.12, 1.39), ncol=5, frameon=False, fontsize=11,
                         columnspacing=1.2, handlelength=2.1)
    save(fig, "retained_loss_outlier_seed_traces", (.22, .20))

    paired_gap = []
    for a in diagnostics:
        if a["condition"]["lr_profile"] != "source": continue
        match = dict(a["condition"], lr_profile="source_lr10")
        b = next(r for r in diagnostics if r["condition"] == match)
        values = np.asarray(b["metrics"]["final_train_minus_validation_accuracy_pp"]["seed_values"]) - np.asarray(
            a["metrics"]["final_train_minus_validation_accuracy_pp"]["seed_values"])
        paired_gap.append({"fixed_condition": {k: v for k, v in a["condition"].items() if k != "lr_profile"},
                           "lr10_minus_source_generalization_gap_pp": moment(values.tolist())})
    gap_context_means = [r["lr10_minus_source_generalization_gap_pp"]["mean"] for r in paired_gap]
    late_contexts = {}
    for lr in ("source", "source_lr10"):
        rows = [r for r in diagnostics if r["condition"]["lr_profile"] == lr]
        gains = [r["metrics"]["late_validation_gain_pp"]["mean"] for r in rows]
        best_epochs = [s["best_accuracy_epoch"] for r in rows for s in r["per_seed"]]
        late_contexts[lr] = {"matched_conditions": len(rows), "positive_late_gain_conditions": sum(x > 0 for x in gains),
                             "late_gain_pp_range": [min(gains), max(gains)],
                             "late_gain_pp_median": float(np.median(gains)),
                             "best_accuracy_epoch_median_across_runs": float(np.median(best_epochs)),
                             "runs_with_best_epoch_at_least_80": sum(x >= 80 for x in best_epochs),
                             "runs": len(best_epochs)}
    diagnostic = {
        "status": "complete", "result_fingerprint": summary["result_fingerprint"],
        "input_sha256": input_hashes, "renderer_sha256": sha(__file__),
        "mcutils_source": str(Path(McUtils.__file__).resolve()),
        "unit_of_replication": "training seed, including its stochastic train/validation resampling",
        "uncertainty": "sample SD (ddof=1) across exactly three seeds at identical five-factor conditions",
        "not_molecule_level_standard_errors": True, "all_144_runs_and_100_epochs_retained": True,
        "limits": ["Only 100 epochs were observed; later convergence cannot be inferred.",
                   "Three seeds characterize variability only coarsely; no significance claim is made.",
                   "Training and validation epochs use native resampled paired populations; they are not a fixed full-set evaluation.",
                   "BCE is confidence sensitive but is not itself an explicit calibration metric; logits are needed to diagnose miscalibration.",
                   "The fixed held-out test selects no diagnostic example or training checkpoint."],
        "matched_lr_example": {"condition_without_lr": EXAMPLE, "curves": example_numeric,
                               "accuracy_axis_zoom_percent": [90, 100]},
        "condition_diagnostics": diagnostics, "paired_lr_generalization_gap": paired_gap,
        "learning_onset_diagnostic": {"fixed_factors": {"feature_mode": "full", "precision": "decoded", "lr_profile": "source"},
                                      "curves": onset_numeric,
                                      "threshold_definition": "First starting epoch with five consecutive logged validation accuracies at least90 or95percent; descriptive only, not a production gate or guarantee of later stability."},
        "descriptive_context_summary": {
            "not_an_error_bar_or_significance_test": True,
            "lr10_minus_source_gap_condition_means_pp": {"contexts": len(gap_context_means),
                "positive": sum(x > 0 for x in gap_context_means),
                "range": [min(gap_context_means), max(gap_context_means)],
                "median": float(np.median(gap_context_means))},
            "late_gain_definition": "Mean validation accuracy in epochs81–100 minus epochs61–80, in percentage points.",
            "late_convergence": late_contexts},
        "retained_outlier": {"condition": OUTLIER, "target_seed": 3, "key": target["key"],
                             "final_metrics": target["final_metrics"], "best_accuracy_row": best,
                             "minimum_validation_bce_row": min_bce,
                             "validation_bce_max": max(r["validation_bce"] for r in target_rows),
                             "validation_bce_above_0_5_epochs": sum(r["validation_bce"] > .5 for r in target_rows),
                             "post_epoch20_validation_bce_above_0_5_epochs": sum(
                                 r["validation_bce"] > .5 for r in target_rows if r["epoch"] > 20),
                             "checkpoint_rule": "Maximum resampled-validation accuracy, first tie; not minimum BCE.",
                             "no_exclusion_or_reselection": True},
        "outputs_sha256": {name: sha(output / name) for name in products}}
    (output / "convergence-diagnostics.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "figures": products, "diagnostic": str(output / "convergence-diagnostics.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--inputs", type=Path, default=here / "inputs")
    parser.add_argument("--output", type=Path, default=here / "convergence-figures")
    parser.add_argument("--mcutils-source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    render(args.inputs, args.output, args.mcutils_source)
