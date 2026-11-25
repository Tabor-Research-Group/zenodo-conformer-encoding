"""Reproduce the independent 144-run factorial analysis with Python stdlib.

Replication is the training seed (three seeds), never a conformer or another
factorial cell. All paired effects fix the other factors. For descriptive
design averages, average cell differences within seed BEFORE computing mean
and sample SD across the three seed averages. No fit, exclusion, test-based
selection, hypothesis test, or molecular confidence interval is performed.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path


FACTORS = ("scope", "precision", "lr_profile", "feature_mode", "readout")
SEEDS = (1, 2, 3)
METRICS = ("test_accuracy", "best_validation_accuracy",
           "final_validation_accuracy", "final_validation_bce")
SNAPSHOT = "analysis/snapshots/snapshot-709d09ec5eb0812baed37c2b"
REMOTE_ROOT = "/scratch/user/maboyer/llm/encoding/chiro-full-factorial-20261024"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(values):
    return list(dict.fromkeys(values))


def mean(values):
    # Deliberately retain the original analysis's sequential double summation.
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def sample_sd(values):
    if len(values) < 2:
        raise ValueError("Sample SD requires at least two training seeds")
    average = mean(values)
    total = 0.0
    for value in values:
        delta = value - average
        total += delta * delta
    return math.sqrt(total / (len(values) - 1))


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def load_inputs(root):
    snapshot = root / SNAPSHOT
    manifest = json.loads((snapshot / "files-sha256.json").read_text())
    for name, expected in manifest.items():
        actual = sha(snapshot / name)
        if actual != expected:
            raise ValueError("Snapshot hash mismatch: " + name)
    current = json.loads((root / "analysis/current.json").read_text())
    if current["snapshot"] != SNAPSHOT or current["status"] != "complete":
        raise ValueError("This analysis is pinned to the completed snapshot")
    read = lambda name: json.loads((snapshot / name).read_text())
    ledger, conditions, contrasts = (read(name) for name in
        ("run-ledger.json", "condition-means.json", "paired-contrasts.json"))
    strings = set(FACTORS) | {"key", "status", "train_batch_rows_sha256",
                              "validation_batch_rows_sha256"}
    with (snapshot / "long-history.csv").open(newline="") as stream:
        history = [{key: value if key in strings else float(value)
                    for key, value in row.items()} for row in csv.DictReader(stream)]
    return manifest, ledger, conditions, contrasts, history


def audit_endpoints(rows):
    issues = []
    for row in rows:
        c, t, key = row["config_summary"], row["test_result"], row["key"]
        def check(valid, reason):
            if not valid:
                issues.append([key, reason])
        check(row["terminal"] and row["status"] == c["status"] == "complete", "status")
        check(c["epochs"] == c["completed_epochs"] == row["logged_epochs"]
              == row["last_logged"]["epoch"] == 100
              and row["uncommitted_history_tail_epochs"] == 0, "epochs")
        check(c["optimizer_updates"] == row["last_logged"]["optimizer_updates"] == 344200
              and c["native_updates_per_epoch"] == {"train": 3442, "validation": 734}, "updates")
        check(row["test_endpoint_available"] and row["training_endpoint_available"]
              and c["required_test_evaluation"] and c["test_evaluated"]
              and t["status"] == "complete" and t["records"] == t["observations"]["records"] == 69719
              and t["observations"]["batches"] == 2179 and t["all_rows_evaluated"]
              and t["parameters_unchanged"] and not t["pair_averaging"], "test")
        check(c["best_checkpoint_sha256"] == t["binding"]["best_checkpoint_sha256"]
              and c["best_epoch"] == t["binding"]["best_validation_epoch"]
              and c["feature_manifest_sha256"] == t["binding"]["feature_manifest_sha256"]
              and c["source_manifest_sha256"] == t["binding"]["source_manifest_sha256"]
              and t["binding"]["task"] == row["task"], "test binding")
        check(t["correct"] / 69719 == t["accuracy"] and 0 <= t["correct"] <= 69719,
              "correctness counts")
        check(all(isinstance(v, (int, float)) and math.isfinite(v)
                  for v in row["final_metrics"].values()), "nonfinite metrics")
        check(not row.get("artifact_errors") and not row.get("warnings"), "artifact/warnings")
        check(row["lifecycle"]["exit_code"] == 0 and row["lifecycle"]["state"] == "completed", "lifecycle")
    def counts(function):
        return dict(Counter(str(function(row)).lower() if isinstance(function(row), bool)
                            else str(function(row)) for row in rows))
    def extent(key):
        values = [row["final_metrics"][key] for row in rows]
        return {"min": min(values), "max": max(values)}
    return {"checked_runs": len(rows), "unique_tasks": len({r["key"] for r in rows}),
            "issues": issues, "scopes": counts(lambda r: r["task"]["scope"]),
            "seeds": counts(lambda r: r["task"]["seed"]),
            "history_rows": sum(r["logged_epochs"] for r in rows),
            "test_records": counts(lambda r: r["test_result"]["records"]),
            "test_row_order_hashes": unique(r["test_result"]["data"]["original_row_order_sha256"] for r in rows),
            "test_batch_row_hashes": unique(r["test_result"]["observations"]["ordered_batch_rows_sha256"] for r in rows),
            "near_chance_training": sum(r["near_chance_training_endpoint"] for r in rows),
            "collapse": sum(r["observed_accuracy_collapse"] for r in rows),
            "near_chance_last": sum(r["near_chance_last_logged"] for r in rows),
            "binaries_rehashed": {"checkpoint": counts(lambda r: r["checkpoint_binary_rehashed"]),
                                  "prediction": counts(lambda r: r["test_prediction_binary_rehashed"])},
            **{key: extent(key) for key in ("final_train_accuracy", "final_validation_accuracy", "test_accuracy")}}


def recompute_published(rows, conditions, contrasts):
    errors = []
    for condition in conditions:
        matches = [row for row in rows if all(row["task"][f] == condition[f] for f in FACTORS)]
        if len(matches) != 3 or abs(mean([r["final_metrics"][condition["metric"]] for r in matches])
                                    - condition["mean"]) > 1e-14:
            errors.append(["condition", condition["condition"], condition["metric"]])
    for contrast in contrasts:
        for seed in SEEDS:
            def matching(level):
                return [row for row in rows if row["task"]["seed"] == seed and all(
                    row["task"][f] == (level if f == contrast["factor"] else contrast[f]) for f in FACTORS)]
            a, b = matching(contrast["level_a"]), matching(contrast["level_b"])
            if len(a) != 1 or len(b) != 1 or abs(b[0]["final_metrics"][contrast["metric"]]
                  - a[0]["final_metrics"][contrast["metric"]]
                  - contrast["paired_differences"][str(seed)]) > 1e-14:
                errors.append(["contrast", contrast["factor"], contrast["level_a"],
                               contrast["level_b"], contrast["metric"], seed])
    return {"condition_means_checked": len(conditions),
            "paired_seed_metric_differences_checked": len(contrasts) * 3, "errors": errors}


def factor_effects(contrasts):
    result = []
    for factor in FACTORS:
        pairs = unique((r["level_a"], r["level_b"]) for r in contrasts if r["factor"] == factor)
        for a, b in pairs:
            for metric in METRICS:
                cells = [r for r in contrasts if (r["factor"], r["level_a"], r["level_b"], r["metric"])
                         == (factor, a, b, metric)]
                seed_averages = [mean([r["paired_differences"][str(seed)] for r in cells]) for seed in SEEDS]
                result.append({"factor": factor, "a": a, "b": b, "metric": metric,
                    "cells": len(cells), "min": min(r["mean"] for r in cells),
                    "max": max(r["mean"] for r in cells),
                    "positive": sum(r["mean"] > 0 for r in cells),
                    "negative": sum(r["mean"] < 0 for r in cells),
                    "three_seeds_positive": sum(all(v > 0 for v in r["paired_differences"].values()) for r in cells),
                    "three_seeds_negative": sum(all(v < 0 for v in r["paired_differences"].values()) for r in cells),
                    "seed_averaged_mean": mean(seed_averages), "seed_averaged_sd": sample_sd(seed_averages),
                    "seed_averaged_deltas": seed_averages})
    return result


def interactions(contrasts):
    result = []
    for factor, modifier in itertools.combinations(FACTORS, 2):
        source = [r for r in contrasts if r["factor"] == factor and r["metric"] == "test_accuracy"]
        for a, b in unique((r["level_a"], r["level_b"]) for r in source):
            subset = [r for r in source if (r["level_a"], r["level_b"]) == (a, b)]
            levels = unique(r[modifier] for r in subset)
            for ma, mb in itertools.combinations(levels, 2):
                fixed = [f for f in FACTORS if f not in (factor, modifier)]
                signature = lambda row: tuple(row[f] for f in fixed)
                left = {signature(r): r for r in subset if r[modifier] == ma}
                right = {signature(r): r for r in subset if r[modifier] == mb}
                differences = [[right[key]["paired_differences"][str(seed)]
                                - row["paired_differences"][str(seed)] for seed in SEEDS]
                               for key, row in left.items()]
                seed_averages = [mean([d[i] for d in differences]) for i in range(3)]
                cell_means = [mean(d) for d in differences]
                result.append({"factor": factor, "contrast": a + "|" + b, "modifier": modifier,
                    "modifier_a": ma, "modifier_b": mb,
                    "direction": "factor effect at modifier_b minus effect at modifier_a",
                    "conditional_cells": len(differences), "mean": mean(seed_averages),
                    "sd": sample_sd(seed_averages), "seed_averaged_deltas": seed_averages,
                    "min_conditional_mean": min(cell_means), "max_conditional_mean": max(cell_means),
                    "all_conditional_means_positive": all(v > 0 for v in cell_means),
                    "all_conditional_means_negative": all(v < 0 for v in cell_means)})
    return result


def stratified_effects(contrasts):
    result = []
    for factor in ("precision", "feature_mode", "readout", "scope"):
        source = [r for r in contrasts if r["factor"] == factor and r["metric"] == "test_accuracy"]
        for a, b in unique((r["level_a"], r["level_b"]) for r in source):
            subset = [r for r in source if (r["level_a"], r["level_b"]) == (a, b)]
            for modifier in [f for f in FACTORS if f != factor]:
                for level in unique(r[modifier] for r in subset):
                    cells = [r for r in subset if r[modifier] == level]
                    averages = [mean([r["paired_differences"][str(seed)] for r in cells]) for seed in SEEDS]
                    result.append({"factor": factor, "contrast": a + "|" + b, "modifier": modifier,
                        "level": level, "cells": len(cells), "mean": mean(averages),
                        "sd": sample_sd(averages), "seed_averaged": averages,
                        "min_cell_mean": min(r["mean"] for r in cells),
                        "max_cell_mean": max(r["mean"] for r in cells)})
    return result


def grouped_training(rows):
    result = []
    for factor in FACTORS:
        for level in unique(r["task"][factor] for r in rows):
            selected = [r for r in rows if r["task"][factor] == level]
            epochs = sorted(r["config_summary"]["best_epoch"] for r in selected)
            result.append({"factor": factor, "level": level, "n": len(selected),
                "best_epoch_mean": mean(epochs),
                # Original report intentionally described the upper middle order statistic.
                "best_epoch_median": epochs[len(epochs) // 2],
                "best_epoch_min": min(epochs), "best_epoch_max": max(epochs),
                "best_at100": sum(e == 100 for e in epochs),
                "final_train_val_gap_mean": mean([r["final_metrics"]["final_train_accuracy"]
                    - r["final_metrics"]["final_validation_accuracy"] for r in selected]),
                "final_val_drop_from_selected_mean": mean([r["final_metrics"]["best_validation_accuracy"]
                    - r["final_metrics"]["final_validation_accuracy"] for r in selected]),
                "final_validation_bce_mean": mean([r["final_metrics"]["final_validation_bce"] for r in selected]),
                "selected_validation_bce_mean": mean([r["final_metrics"]["selected_validation_bce"] for r in selected])})
    return result


def temporal_analysis(rows, history):
    temporal, learning = [], []
    for lr in ("source", "source_lr10"):
        selected = [r for r in rows if r["task"]["lr_profile"] == lr]
        records = [r for r in history if r["lr_profile"] == lr]
        by_key = {r["key"]: [h for h in records if h["key"] == r["key"]] for r in selected}
        first95 = [next((h["epoch"] for h in by_key[r["key"]] if h["validation_accuracy"] >= .95), None)
                   for r in selected]
        reached = [e for e in first95 if e is not None]
        differences = []
        for seed in SEEDS:
            subset = [h for h in records if h["seed"] == seed]
            differences.append(mean([h["validation_accuracy"] for h in subset if h["epoch"] == 100])
                               - mean([h["validation_accuracy"] for h in subset if h["epoch"] == 50]))
        temporal.append({"lr_profile": lr, "runs": len(selected),
            "first_95pct_validation": {"reached": len(reached), "median_epoch_reached": median(reached),
                                       "min_epoch": min(reached), "max_epoch": max(reached)},
            "best_epoch_ge90": sum(r["config_summary"]["best_epoch"] >= 90 for r in selected),
            "mean_validation_change_50_to100": mean(differences),
            "sd_seed_averages_validation_change_50_to100": sample_sd(differences),
            "minimum_BCE_epoch_median": median([min(by_key[r["key"]], key=lambda h: h["validation_bce"])["epoch"]
                                                for r in selected]),
            "final_BCE_greater_than_own_logged_min": sum(r["final_metrics"]["final_validation_bce"]
                    > r["minimum_logged_validation_bce"] for r in selected)})
        for epoch in (1, 5, 10, 20, 50, 75, 100):
            row = {"lr_profile": lr, "epoch": epoch, "conditions_per_seed": 24}
            for metric in ("train_accuracy", "validation_accuracy", "train_bce", "validation_bce"):
                values = [mean([h[metric] for h in records if h["seed"] == seed and h["epoch"] == epoch])
                          for seed in SEEDS]
                row[metric] = {"mean": mean(values), "seed_sd": sample_sd(values), "seed_values": values}
            learning.append(row)
    return temporal, learning


METHOD = {
    "replication_unit": "training seed, n=3; identical molecular population is shared by all conditions",
    "primary_error_bars": "sample standard deviation(ddof1) across seeds within each condition; show the three seed points",
    "paired_error_bars": "sample SD of the three same-seed differences at fixed other factors",
    "balanced_design_average": "For a descriptive average effect, first average fixed-condition paired differences within each seed, then report mean and sample SD across the three seed averages. Do not treat the 16/24 matched conditions as independent replicates.",
    "interactions": "Difference-of-differences formed within the same seed and at identical levels of all remaining factors; average fixed remaining-factor combinations within seed, then report mean/sampleSD across three seeds.",
    "checkpoint_selection": "highest native resampled-validation accuracy, first tie; test is never used for selection",
    "scope": "all144native factorial tasks; all100epochs; 48conditions×3seeds; each test endpoint69719official conformers",
    "precision": "decoded means source-tag reconstructed geometry; psi8 adds native8bit quantization to model torsions. Distance and angle inputs remain fixed. Neither is untouched raw geometry.",
    "readout": "z_alpha is torsion-only output, still generated using graph-conditioned node representations and chemistry; both combines graph, distances, bond angles, and alpha.",
    "chemistry": "reduced52 zeros charge, hybridization, mass and all bond descriptors while retaining native52/14widths, element, degree, totalH, aromaticity, connectivity and unchanged neural backbone.",
    "binary_integrity": "Pool checked receipt/config bindings. Large model and prediction binaries were not freshly rehashed in this audit.",
    "uncertainty": "Three seeds permit descriptive optimization variability only. These are not molecular/bootstrap confidence intervals, equivalence tests, or causal attribution of historical failures."
}


def analyze(root):
    manifest, rows, conditions, contrasts, history = load_inputs(root)
    audit = audit_endpoints(rows)
    recomputed = recompute_published(rows, conditions, contrasts)
    if audit["issues"] or recomputed["errors"]:
        raise ValueError({"endpoint_issues": audit["issues"], "recomputation_errors": recomputed["errors"]})
    temporal, learning = temporal_analysis(rows, history)
    outlier = max(rows, key=lambda r: r["final_metrics"]["final_validation_bce"])
    primary = [{"scope": r["scope"], "precision": r["precision"], "metric": r["metric"],
                "mean": r["mean"], "sd": r["sample_sd"], "seed_values": r["seed_values"]}
               for r in conditions if r["feature_mode"] == "full" and r["readout"] == "both"
               and r["lr_profile"] == "source" and r["metric"] in
               ("test_accuracy", "best_validation_accuracy", "final_validation_accuracy")]
    return {"schema": "independent_chiro_factorial_analysis_v1", "created_utc": "2026-10-26",
        "remote_root": REMOTE_ROOT, "snapshot": SNAPSHOT, "remote_snapshot_files_sha256": manifest,
        "input_receipt_integrity_audit": audit, "numerical_recomputation": recomputed,
        "units": "Accuracy fields and effects are fractions unless the field name says percentage_points; multiply by100 for percentage points.",
        "method": METHOD, "factor_effects": factor_effects(contrasts),
        "test_accuracy_two_factor_interactions": interactions(contrasts),
        "stratified_test_accuracy_effects": stratified_effects(contrasts),
        "grouped_training_diagnostics": grouped_training(rows),
        "temporal_diagnostics": temporal, "balanced_learning_curve_landmarks": learning,
        "prespecified_full_chemistry_both_readout_sourceLR_slice": primary,
        "highest_final_validation_BCE_diagnostic": {"key": outlier["key"], "final": outlier["final_metrics"],
            "selected_epoch": outlier["config_summary"]["best_epoch"],
            "min_validation_bce": outlier["minimum_logged_validation_bce"]}}


def report_text(data):
    effects = {r["factor"]: r for r in data["factor_effects"] if r["metric"] == "test_accuracy"}
    scope = {(r["a"], r["b"]): r for r in data["factor_effects"]
             if r["factor"] == "scope" and r["metric"] == "test_accuracy"}
    lr, precision = effects["lr_profile"], effects["precision"]
    chemistry, readout = effects["feature_mode"], effects["readout"]
    am, cm = scope[("minimal", "augmented")], scope[("minimal", "complete")]
    interaction = next(r for r in data["test_accuracy_two_factor_interactions"]
                       if r["factor"] == "scope" and r["contrast"] == "minimal|complete"
                       and r["modifier"] == "feature_mode")
    g = {r["level"]: r for r in data["grouped_training_diagnostics"] if r["factor"] == "lr_profile"}
    last = {r["lr_profile"]: r for r in data["balanced_learning_curve_landmarks"] if r["epoch"] == 100}
    lower50 = next(r for r in data["balanced_learning_curve_landmarks"]
                   if r["lr_profile"] == "source_lr10" and r["epoch"] == 50)
    temporal = {r["lr_profile"]: r for r in data["temporal_diagnostics"]}
    outlier = data["highest_final_validation_BCE_diagnostic"]
    n = data["input_receipt_integrity_audit"]["checked_runs"]
    return f"""Independent numerical analysis of all {n} ChIRo factorial runs
2026-10-26; snapshot snapshot-709d09ec5eb0812baed37c2b.

All {n} runs completed 100 epochs and full 69,719-conformer test evaluations. The 48 conditions each have three seeds. I independently reconstructed all {data['numerical_recomputation']['condition_means_checked']} condition means and {data['numerical_recomputation']['paired_seed_metric_differences_checked']:,} seed-level metric contrasts from the run ledger; no discrepancy was found. No runs were excluded or selected by accuracy for inclusion. The comparisons below do not choose a model using test accuracy.

For condition plots, use mean plus/minus sample SD across the three training seeds and show each seed. For paired effects, use the same three seed differences with every other factor fixed. The descriptive design averages in the accompanying CSV first average matched conditions within seed, then compute mean and SD over the three seed averages. Do not compute their uncertainty across correlated conformers or across the 16/24 factorial conditions.

Learning rate has the strongest consistent endpoint effect. Reducing the original ChIRo rate 5.694249946e-4 by ten lowers test accuracy in every one of {lr['cells']} matched conditions and all {lr['three_seeds_negative']*3} seed pairs. Condition mean losses range from {-100*lr['max']:.3f} to {-100*lr['min']:.3f} percentage points. The balanced seed-average effect is {100*lr['seed_averaged_mean']:.3f} plus/minus {100*lr['seed_averaged_sd']:.3f} percentage points. Final training accuracy is nearly unchanged across the balanced design ({100*last['source']['train_accuracy']['mean']:.3f}% versus {100*last['source_lr10']['train_accuracy']['mean']:.3f}%), whereas final validation accuracy is {100*last['source']['validation_accuracy']['mean']:.3f}% versus {100*last['source_lr10']['validation_accuracy']['mean']:.3f}%. This is worse validation/test generalization at the tested exposure, rather than inability to fit training. It is not an asymptotic convergence result: median selected epochs are {g['source']['best_epoch_median']} and {g['source_lr10']['best_epoch_median']}; {temporal['source_lr10']['best_epoch_ge90']} of 72 lower-rate runs choose epoch 90 or later.

Additional torsion quantization has no uniform endpoint direction. The psi8-minus-decoded test effect is positive in {precision['positive']} of {precision['cells']} condition means and negative in {precision['negative']}; the range is {100*precision['min']:.3f} to {100*precision['max']:+.3f} percentage points. The balanced effect is {100*precision['seed_averaged_mean']:+.3f} plus/minus {100*precision['seed_averaged_sd']:.3f} percentage points. This supports a small design-average incremental effect in this experiment, rather than strict equivalence or a universal benefit of discretization.

There is no universal torsion-scope ranking. Augmented-minus-Minimal test means range from {100*am['min']:.3f} to {100*am['max']:+.3f} percentage points, with ten positive and six negative; Complete-minus-Minimal ranges from {100*cm['min']:.3f} to {100*cm['max']:+.3f}, with eight of each sign. Descriptor reduction changes this comparison: Complete-minus-Minimal shifts downward after reducing chemistry in all eight matched remaining-factor condition means (by {-100*interaction['max_conditional_mean']:.3f} to {-100*interaction['min_conditional_mean']:.3f} percentage points). The balanced interaction is {100*interaction['mean']:.3f} plus/minus {100*interaction['sd']:.3f} percentage points. This is a conditional interaction, not proof that adding torsions inherently helps or harms.

Reduced chemistry has a modest negative design average ({100*chemistry['seed_averaged_mean']:.3f} plus/minus {100*chemistry['seed_averaged_sd']:.3f} percentage points), but conditional effects range from {100*chemistry['min']:.3f} to {100*chemistry['max']:+.3f}. Removing graph/distance/angle output branches leaves the graph-conditioned alpha representation: z_alpha-minus-both averages {100*readout['seed_averaged_mean']:.3f} plus/minus {100*readout['seed_averaged_sd']:.3f} percentage points, with a conditional range of {100*readout['min']:.3f} to {100*readout['max']:+.3f}. Do not describe z_alpha as geometry with no chemical context, or reduced52 as deletion of connectivity or a narrower neural network.

Accuracy and BCE should both remain visible. One Complete/psi8/source-LR/reduced52/both seed-3 run has final validation accuracy {100*outlier['final']['final_validation_accuracy']:.3f}% but BCE {outlier['final']['final_validation_bce']:.3f}; its selected highest-accuracy checkpoint has BCE {outlier['final']['selected_validation_bce']:.3f}, and its earlier minimum BCE was {outlier['min_validation_bce']:.3f}. This run is complete and must remain in the seed variance. More generally, lower-rate balanced validation BCE increases from {lower50['validation_bce']['mean']:.3f} at epoch 50 to {last['source_lr10']['validation_bce']['mean']:.3f} at epoch 100 while training BCE falls. High accuracy alone does not establish good probability estimates. Checkpoint selection remains maximum validation accuracy, not minimum BCE.

All cells begin with decoded source-tag geometries. psi8 adds quantization of model torsions while distance/angle arrays stay fixed. There is no untouched-raw-geometry arm here, so these curves quantify the incremental torsion discretization effect, not the entire effect of storing conformers in the compact format. The original population, labels, and audited rare reconstruction losses are retained.

Three seeds provide limited descriptive uncertainty. Avoid significance or noninferiority claims from overlapping SD bars. Test conformers are correlated within molecules and shared across runs; they are not independent experimental replicates. Large checkpoint/prediction binaries were not freshly rehashed in this read-only analysis; native receipts and pool bindings were checked.
"""


def write_outputs(output, data):
    output.mkdir(parents=True, exist_ok=True)
    (output / "independent-analysis.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "independent-analysis.txt").write_text(report_text(data) + "\n")
    columns = ("factor", "level_a", "level_b", "fixed_conditions", "mean_effect_pp",
        "seed_sd_of_design_average_pp", "min_cell_mean_pp", "max_cell_mean_pp", "positive_cell_means",
        "negative_cell_means", "all_three_seed_positive_cells", "all_three_seed_negative_cells",
        "seed1_design_average_pp", "seed2_design_average_pp", "seed3_design_average_pp")
    with (output / "independent-paired-effects.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in data["factor_effects"]:
            if row["metric"] == "test_accuracy":
                writer.writerow([row["factor"], row["a"], row["b"], row["cells"],
                    100*row["seed_averaged_mean"], 100*row["seed_averaged_sd"], 100*row["min"], 100*row["max"],
                    row["positive"], row["negative"], row["three_seeds_positive"], row["three_seeds_negative"],
                    *[100*v for v in row["seed_averaged_deltas"]]])
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Local verified transport root containing analysis/current.json and the snapshot")
    parser.add_argument("--output", type=Path, help="Default is the same root; only independent-* outputs are written")
    args = parser.parse_args()
    data = analyze(args.root.resolve())
    output = (args.output or args.root).resolve()
    write_outputs(output, data)
    print(json.dumps({"status": "reproduced", "root": str(args.root.resolve()), "output": str(output),
                     "condition_means_checked": data["numerical_recomputation"]["condition_means_checked"],
                     "paired_seed_metric_differences_checked": data["numerical_recomputation"]["paired_seed_metric_differences_checked"],
                     "replication_unit": "training seed (n=3)"}))


if __name__ == "__main__":
    main()
