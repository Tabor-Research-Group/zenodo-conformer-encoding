"""Native McUtils endpoint and matched-factor figures for the locked R/S study.

Input is the immutable pooling snapshot, never model checkpoints or predictions.
Variability is the sample SD of three training seeds (or paired seed differences),
not a conformer-level standard error or a statistical significance test.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys


FACTORS = ("scope", "precision", "lr_profile", "feature_mode", "readout")
LEVELS = {"scope": ("minimal", "augmented", "complete"),
          "precision": ("decoded", "psi8"), "lr_profile": ("source", "source_lr10"),
          "feature_mode": ("full", "reduced52"), "readout": ("both", "z_alpha")}
COLORS = {"minimal": "#008000", "augmented": "#D7478A", "complete": "#071F83"}
SHORT = {"minimal": "Min", "augmented": "Aug", "complete": "Comp",
         "decoded": "D", "psi8": "8b", "source": "LR", "source_lr10": "LR/10",
         "full": "F", "reduced52": "R", "both": "All", "z_alpha": "T"}
SOURCE_DEFAULT = Path("/Users/Mark/Documents/UW/Research/Development/McUtils")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def same(row, **values):
    return all(row[key] == value for key, value in values.items())


def load_inputs(folder):
    manifest = json.loads((folder / "files-sha256.json").read_text())
    files = ("summary.json", "condition-means.csv", "paired-contrasts.csv")
    for name in files:
        require(name in manifest and sha(folder / name) == manifest[name],
                "Immutable pooled input changed: " + name)
    summary = json.loads((folder / "summary.json").read_text())
    require(not summary["artifact_errors"] and not summary["sampler_mismatches"]
            and not summary["initialization_mismatches"],
            "Input provenance/sampler/initialization requires review")
    endpoints = [row for row in read_csv(folder / "condition-means.csv")
                 if row["metric"] == "test_accuracy"]
    require(len(endpoints) == 48, "All48 planned condition slots must remain explicit")
    for row in endpoints:
        row["n"] = int(row["n"])
        row["seed_values"] = json.loads(row["seed_values"])
        row["mean"] = float(row["mean"]) if row["mean"] else None
        row["sample_sd"] = float(row["sample_sd"]) if row["sample_sd"] else None
    contrasts = [row for row in read_csv(folder / "paired-contrasts.csv")
                 if row["metric"] == "test_accuracy"]
    for row in contrasts:
        row["n"] = int(row["n"])
        row["paired_differences"] = json.loads(row["paired_differences"])
        row["mean"] = float(row["mean"]) if row["mean"] else None
        row["sample_sd"] = float(row["sample_sd"]) if row["sample_sd"] else None
    return summary, endpoints, contrasts, {name: manifest[name] for name in files}


def configure(output, source):
    for variable, name in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib"),
                           ("XDG_CACHE_HOME", "xdg")):
        path = output / "runtime-cache" / "factor-plots" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(source.resolve()))
    import McUtils
    from McUtils.Plots import GraphicsGrid, Plot, ErrorBarPlot, PlotLegend, Styled
    require(source.resolve() in Path(McUtils.__file__).resolve().parents,
            "McUtils must be imported from the requested source")
    return McUtils, GraphicsGrid, Plot, ErrorBarPlot, PlotLegend, Styled


def render(inputs, output, source=SOURCE_DEFAULT):
    require(platform.machine() == "arm64", "Use the native ARM Python environment")
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary, endpoints, contrasts, input_hashes = load_inputs(inputs)
    McUtils, GraphicsGrid, Plot, ErrorBarPlot, PlotLegend, Styled = configure(output, source)
    import numpy as np
    theme = ("mccoy", {"font": {"family": "DejaVu Sans", "size": 12},
                       "svg": {"fonttype": "path"}})
    outputs = []
    plotted = {"endpoint_cells": [], "binary_contrasts": [], "scope_contrasts": []}

    def grid(rows, cols, size, title, subtitle, padding, spacings):
        return GraphicsGrid(nrows=rows, ncols=cols, theme=theme,
                            subimage_size=size, padding=padding, spacings=spacings,
                            figure_label=Styled(title + "\n" + subtitle, fontsize=15))

    def save(figure, name, spacing):
        figure.prep_show()
        figure.figure.set_figure_spacings(spacing)
        for extension in ("svg", "png", "pdf"):
            path = output / (name + "." + extension)
            figure.figure.savefig(str(path), format=extension, dpi=190,
                                  facecolor=figure.background)
            outputs.append(path.name)
        figure.close()

    def check_stats(row, values):
        require(len(values) == row["n"] and all(math.isfinite(v) for v in values),
                "Pooled seed count or finite values differ")
        if values:
            require(abs(float(np.mean(values)) - row["mean"]) < 1e-12,
                    "Pooled mean differs from individual seed values")
        if len(values) >= 2:
            require(abs(float(np.std(values, ddof=1)) - row["sample_sd"]) < 1e-12,
                    "Pooled sample SD differs from individual seed values")

    # All48 cells, including low-performing/partial cells, remain in the same grid.
    figure = grid(2, 2, (610, 365), "Held-out R/S accuracy across all 48 conditions",
                  "Small dots: individual seeds; large markers: mean ± sample SD (n = 3)\n"
                  "Circles: LR = 5.69 × 10⁻⁴; squares: LR = 5.69 × 10⁻⁵",
                  ((100, 45), (105, 160)), (135, 160))
    available = [100 * float(v) for row in endpoints for v in row["seed_values"].values()]
    require(available, "No held-out endpoints are available")
    lo = max(0, math.floor(min(available) - 1))
    hi = min(101, math.ceil(max(available) + .7))
    if hi - lo < 5:
        lo = max(0, hi - 5)
    groups = list(itertools.product(LEVELS["scope"], LEVELS["precision"]))
    for i, feature in enumerate(LEVELS["feature_mode"]):
        for j, readout in enumerate(LEVELS["readout"]):
            panel = figure[i, j]
            panel.aspect_ratio = "auto"
            for group, (scope, precision) in enumerate(groups):
                for lr, shift, marker in (("source", -.17, "o"), ("source_lr10", .17, "s")):
                    selected = [row for row in endpoints if same(row, scope=scope, precision=precision,
                                feature_mode=feature, readout=readout, lr_profile=lr)]
                    require(len(selected) == 1, "Endpoint cell duplicated or omitted")
                    row = selected[0]
                    values = [float(row["seed_values"][str(seed)]) for seed in (1, 2, 3)
                              if str(seed) in row["seed_values"]]
                    check_stats(row, values)
                    x = group + shift
                    if values:
                        offsets = np.linspace(-.045, .045, len(values))
                        Plot(x + offsets, np.asarray(values) * 100, figure=panel,
                             color=COLORS[scope], marker="o", markersize=4,
                             linestyle="None", plot_style={"alpha": .52})
                        options = dict(figure=panel, color=COLORS[scope], marker=marker,
                                       markersize=6.5, linestyle="None", linewidth=1.6)
                        if len(values) >= 2:
                            ErrorBarPlot([x], [100 * row["mean"]],
                                         yerr=[100 * row["sample_sd"]], capsize=4, **options)
                        else:
                            Plot([x], [100 * row["mean"]], **options)
                    plotted["endpoint_cells"].append({key: row[key] for key in FACTORS + ("n", "seed_values", "mean", "sample_sd")})
            panel.plot_range = [(-.55, 5.55), (lo, hi)]
            panel.ticks = [(list(range(6)), [scope.title() + "\n" + ("Decoded" if precision == "decoded" else "8-bit ψ")
                                            for scope, precision in groups]), None]
            panel.axes_labels = ["Torsion coverage and precision", "Held-out accuracy (%)"]
            panel.plot_label = Styled(("Full descriptors" if feature == "full" else "Reduced descriptors")
                                      + " · " + ("Full readout" if readout == "both" else "Torsion-only readout"), fontsize=13)
    save(figure, "all48_test_accuracy_seed_sd", (.24, .5))

    def contrast_panel(panel, factor, a, b, title, record_key):
        other = [key for key in FACTORS if key != factor]
        rows = [row for row in contrasts if same(row, factor=factor, level_a=a, level_b=b)]
        rows.sort(key=lambda row: tuple(LEVELS[key].index(row[key]) for key in other))
        expected = math.prod(len(LEVELS[key]) for key in other)
        require(len(rows) == expected, "A matched factor grid duplicated or omitted fixed backgrounds")
        extent = 1.
        labels = []
        for y, row in enumerate(rows):
            values = [float(row["paired_differences"][str(seed)]) for seed in (1, 2, 3)
                      if str(seed) in row["paired_differences"]]
            check_stats(row, values)
            color = COLORS[row["scope"]] if factor != "scope" else COLORS[b]
            label = " · ".join(SHORT[row[key]] for key in other)
            if row["n"] != 3:
                label += " (n=" + str(row["n"]) + ")"
            labels.append(label)
            if values:
                values = np.asarray(values) * 100
                extent = max(extent, float(np.max(np.abs(values))))
                Plot(values, y + np.linspace(-.12, .12, len(values)), figure=panel,
                     marker="o", markersize=3.8, linestyle="None", color=color,
                     plot_style={"alpha": .45})
                options = dict(figure=panel, marker="o", markersize=5, color=color,
                               linestyle="None", linewidth=1.4)
                if row["n"] >= 2:
                    ErrorBarPlot([100 * row["mean"]], [y], xerr=[100 * row["sample_sd"]],
                                 capsize=3, **options)
                    extent = max(extent, abs(100 * row["mean"]) + 100 * row["sample_sd"])
                else:
                    Plot([100 * row["mean"]], [y], **options)
            plotted[record_key].append({key: row[key] for key in ("factor", "level_a", "level_b") + FACTORS
                                       + ("n", "paired_differences", "mean", "sample_sd")})
        bound = 1.12 * extent
        Plot([0, 0], [-.6, len(rows) - .4], figure=panel, color="#999999", linewidth=.8)
        panel.plot_range = [(-bound, bound), (len(rows) - .4, -.6)]
        panel.ticks = [None, (list(range(len(rows))), labels)]
        panel.axes_labels = ["Test accuracy change (percentage points)", None]
        panel.plot_label = Styled(title + "\nOther factors fixed within every row", fontsize=12)
        panel.aspect_ratio = "auto"

    figure = grid(2, 2, (630, 580), "Matched effects of precision, learning rate, descriptors and readout",
                  "Dots: paired seeds 1–3; bars: mean ± sample SD of paired differences\n"
                  "D / 8b = decoded / 8-bit; F / R = full / reduced descriptors; All / T = full / torsion-only readout\n"
                  "LR = 5.69 × 10⁻⁴; LR/10 = 5.69 × 10⁻⁵",
                  ((185, 50), (100, 165)), (210, 135))
    binary = [("precision", "decoded", "psi8", "8-bit torsions − decoded torsions"),
              ("lr_profile", "source", "source_lr10", "Source LR/10 − source LR"),
              ("feature_mode", "full", "reduced52", "Reduced descriptors − full descriptors"),
              ("readout", "both", "z_alpha", "Torsion-only readout − full readout")]
    for index, (factor, a, b, title) in enumerate(binary):
        contrast_panel(figure[index // 2, index % 2], factor, a, b, title, "binary_contrasts")
    save(figure, "binary_factor_test_effects_paired_seed_sd", (.72, .3))

    figure = grid(1, 3, (510, 510), "Matched effects of torsion coverage",
                  "Individual paired seed differences and mean ± sample SD; every other factor fixed\n"
                  "D / 8b = decoded / 8-bit; F / R = full / reduced descriptors; All / T = full / torsion-only readout\n"
                  "LR = 5.69 × 10⁻⁴; LR/10 = 5.69 × 10⁻⁵",
                  ((160, 45), (100, 165)), (190, 0))
    for column, (a, b) in enumerate(itertools.combinations(LEVELS["scope"], 2)):
        contrast_panel(figure[0, column], "scope", a, b,
                       b.title() + " − " + a.title(), "scope_contrasts")
    save(figure, "scope_test_effects_paired_seed_sd", (.85, .15))

    # CSV tables preserve the exact endpoints/paired values behind every glyph.
    for name, rows in plotted.items():
        path = output / (name + "_plotted.json")
        path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        outputs.append(path.name)
    receipt = {"status": "rendered", "renderer_sha256": sha(__file__),
               "result_fingerprint": summary["result_fingerprint"], "input_files_sha256": input_hashes,
               "python": sys.version, "machine": platform.machine(), "mcutils_path": str(Path(McUtils.__file__).resolve()),
               "mcutils_module_sha256": sha(McUtils.__file__),
               "outputs_sha256": {name: sha(output / name) for name in outputs},
               "figures": outputs,
               "abbreviations": {"D": "Decoded tag geometry", "8b": "Additional 8-bit torsion discretization",
                   "LR": "5.694249946353567e-4", "LR/10": "5.694249946353567e-5", "F": "Full chemical descriptors",
                   "R": "Reduced descriptors at unchanged 52/14 widths", "All": "Native full 256-component readout",
                   "T": "Native torsion-only 64-component readout"},
               "interpretation": "All planned cells retained. SD is across training seeds (paired seed deltas for contrasts), not conformers, SEM or confidence intervals. Each factor contrast holds every other factor fixed. Three seeds describe optimization variability only; no significance or causal-mechanism claim. Test predictions come from the validation-selected checkpoint."}
    (output / "factor-figure-manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcutils-source", type=Path, default=SOURCE_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(render(args.inputs, args.output, args.mcutils_source), indent=2))
