"""Paper-style validation curves showing the individual training replicates.

Each replicate uses its own cumulative measured train-and-validation time
and validation accuracy. Only the vertical epoch-100 guide averages across
replicates: it marks the mean of their maximum recorded elapsed times.
"""
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

ARCHIVE = Path(__file__).resolve().parents[1]
ROOT = ARCHIVE / 'outputs/training'
DATA_ROOT = ARCHIVE / 'training_results/zmatrix-results-20261028/combined-comparison'
SOURCE = ARCHIVE / 'software/McUtils'
SCOPES = ('complete', 'minimal', 'augmented', 'zmatrix')
LABELS = dict(complete='Complete', minimal='Minimal', augmented='Augmented', zmatrix='Z-matrix')
FAMILIES = (('both', 'source'), ('both', 'source_lr10'))
PALETTE = {
    'complete': ('#071F83', '#375EC9'),
    'minimal': ('#008000', '#45A64D'),
    'augmented': ('#D7478A', '#A91F64'),
    'zmatrix': ('#663399', '#A17CB8'),
}
STYLE = dict(image_size=(440, 300), padding=((45, 10), (35, 12)), font_size=12,
             linewidth=1.1, replicate_alpha=.88, completion_linewidth=2.0,
             completion_alpha=.75, legend_fontsize=10.0, legend_anchor=(.38, .18), dpi=220)
PATTERNS = ('-', '--')
STEM = 'validation_curves_with_zmatrix'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    for var, leaf in (('MPLCONFIGDIR', 'mpl'), ('NUMBA_CACHE_DIR', 'numba'), ('XDG_CACHE_HOME', 'xdg')):
        path = ROOT / 'runtime-cache' / leaf
        path.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(path)
    os.environ['MPLBACKEND'] = 'Agg'
    if sys.platform == 'darwin':
        os.environ.setdefault('DYLD_FALLBACK_LIBRARY_PATH', '/opt/homebrew/lib')
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import McUtils
    from McUtils.Plots import Graphics, Plot, PlotLegend
    import cairosvg
    require(SOURCE in Path(McUtils.__file__).resolve().parents, 'Unexpected McUtils source')
    original = ARCHIVE / 'workflows/chiro-publication-curves-20261026/inputs'
    manifest = json.loads((original / 'files-sha256.json').read_text())
    for name, expected in manifest.items():
        require(sha(original / name) == expected, 'Immutable input changed: ' + name)
    with (original / 'long-history.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['precision'] == 'decoded' and row['feature_mode'] == 'full'
                and row['readout'] == 'both']
    zroot = DATA_ROOT.parent
    zpointer = json.loads((zroot / 'analysis/current.json').read_text())
    zsnapshot = zroot / zpointer['snapshot']
    zmanifest = json.loads((zsnapshot / 'files-sha256.json').read_text())
    for name, expected in zmanifest.items():
        require(sha(zsnapshot / name) == expected, 'Z-matrix snapshot changed: ' + name)
    with (zsnapshot / 'long-history.csv').open(newline='') as stream:
        rows.extend(row for row in csv.DictReader(stream)
                    if row['precision'] == 'decoded' and row['feature_mode'] == 'full'
                    and row['readout'] == 'both')
    prior_file = ARCHIVE / 'workflows/chiro-publication-curves-20261026/validation-throughput/merged-validation-values.json'
    prior = json.loads(prior_file.read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[(row['scope'], row['readout'], row['lr_profile'])].append(row)
    require(len(rows) == 2400 and len(groups) == 8, 'All combined-readout conditions must be retained')
    curves, exports = [], []
    for scope in SCOPES:
        for family_index, (readout, lr) in enumerate(FAMILIES):
            group = sorted(groups[(scope, readout, lr)], key=lambda row: (int(row['seed']), int(row['epoch'])))
            require([(int(row['seed']), int(row['epoch'])) for row in group] ==
                    [(seed, epoch) for seed in (1, 2, 3) for epoch in range(1, 101)],
                    'Missing or duplicate replicate epochs')
            seconds = np.array([float(row['epoch_seconds']) for row in group]).reshape(3, 100)
            accuracy = np.array([float(row['validation_accuracy']) for row in group]).reshape(3, 100)
            times = np.cumsum(seconds, axis=1) / 3600
            require(np.isfinite(times).all() and np.isfinite(accuracy).all() and
                    (np.diff(times, axis=1) > 0).all(), 'Invalid observed trajectory')
            label = LABELS[scope]
            if lr == 'source_lr10':
                label += ' (LR/10)'
            curve = dict(scope=scope, readout=readout, lr_profile=lr,
                color=PALETTE[scope][family_index], linestyle=PATTERNS[family_index], label=label,
                epoch=list(range(1, 101)), seed_ids=[1, 2, 3],
                time_hours_seed_values=times.tolist(), accuracy_seed_values=accuracy.tolist(),
                epoch100_time_hours_mean=float(times[:, -1].mean()), epoch100_time_hours_min=float(times[:, -1].min()),
                epoch100_time_hours_max=float(times[:, -1].max()))
            curves.append(curve)
            for seed in range(3):
                for epoch in range(100):
                    exports.append(dict(scope=scope, readout=readout, lr_profile=lr, seed=seed + 1, epoch=epoch + 1,
                        time_hours=times[seed, epoch], validation_accuracy=accuracy[seed, epoch]))
    for old in prior:
        matching = [curve for curve in curves if all(curve[key] == old[key] for key in ('scope','readout','lr_profile'))]
        require(len(matching) == 1, 'Original condition missing')
        current = matching[0]
        for field in ('time_hours_seed_values','accuracy_seed_values'):
            require(np.array_equal(current[field], old[field]), 'An original replicate changed: ' + old['scope'])
        require(current['epoch100_time_hours_mean'] == old['epoch100_time_hours_mean'], 'Original completion guide changed')
    theme = ('mccoy', {'font': {'family': 'DejaVu Sans', 'size': STYLE['font_size']},
                       'svg': {'fonttype': 'path'}})
    figure = Graphics(image_size=STYLE['image_size'], padding=STYLE['padding'], theme=theme)
    figure.image_size = STYLE['image_size']
    figure.aspect_ratio = 'auto'
    limits = (.475, 1.01)
    # Exactly one condition-colored guide at its mean recorded epoch-100 time.
    for curve in curves:
        finish = curve['epoch100_time_hours_mean']
        Plot([finish, finish], limits, figure=figure, color=curve['color'], linestyle=':',
             linewidth=STYLE['completion_linewidth'],
             plot_style={'alpha': STYLE['completion_alpha'], 'zorder': 0, 'dash_capstyle': 'round'})
    for curve in curves:
        for x, y in zip(curve['time_hours_seed_values'], curve['accuracy_seed_values']):
            Plot(x, y, figure=figure, color=curve['color'], linestyle=curve['linestyle'],
                 linewidth=STYLE['linewidth'], plot_style={'alpha': STYLE['replicate_alpha'], 'zorder': 3})
    figure.axes_labels = ['Time (hours)', 'Accuracy']
    figure.plot_range = [(0, max(c['epoch100_time_hours_max'] for c in curves) * 1.045), limits]
    figure.ticks = [[0, 1, 2, 3, 4, 5], [.5, .6, .7, .8, .9, 1.]]
    handles = PlotLegend.construct([dict(marker='line', label=LABELS[scope], color=PALETTE[scope][0],
                                        linestyle='-', lw=1.8) for scope in SCOPES])
    leg = figure.axes.legend(handles=handles, loc='center', ncol=1,
        bbox_to_anchor=STYLE['legend_anchor'],
        frameon=False, fontsize=STYLE['legend_fontsize'], handlelength=2.3,
        handletextpad=.5, borderaxespad=0, labelspacing=.5)
    figure.prep_show()
    canvas = figure.figure.obj
    canvas.canvas.draw()
    box = leg.get_window_extent(canvas.canvas.get_renderer())
    require(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= canvas.bbox.width and box.y1 <= canvas.bbox.height,
            'Legend outside the canvas')
    # The legend occupies unused data space between the curves and guides.
    legend_data = box.transformed(figure.axes.obj.transData.inverted())
    legend_fraction = box.transformed(canvas.transFigure.inverted())
    legend_svg_bounds = [legend_fraction.x0 * 440, (1 - legend_fraction.y1) * 300,
                         legend_fraction.width * 440, legend_fraction.height * 300]
    for curve in curves:
        require(not legend_data.x0 <= curve['epoch100_time_hours_mean'] <= legend_data.x1,
                'Legend overlaps a completion guide')
        for xs, ys in zip(curve['time_hours_seed_values'], curve['accuracy_seed_values']):
            for x, y in zip(xs, ys):
                if legend_data.x0 <= x <= legend_data.x1:
                    require(y < legend_data.y0 or y > legend_data.y1, 'Legend overlaps a replicate curve')
    # Inspect the native lines: six completion guides and eighteen exact raw
    # replicate traces, with no mean learning curves or range collections.
    native_lines = figure.axes.obj.lines
    require(len(native_lines) == 32 and len(figure.axes.obj.collections) == 0,
            'Expected eight guides and twenty-four individual curves only')
    for curve, line in zip(curves, native_lines[:8]):
        require(np.array_equal(line.get_xdata(), [curve['epoch100_time_hours_mean']] * 2),
                'Completion guide does not match mean maximum elapsed time')
        require(line.get_linestyle() == ':', 'Completion guide must be dotted')
    require([text.get_text() for text in leg.get_texts()] == [LABELS[scope] for scope in SCOPES]
            and leg.get_title().get_text() == '', 'Expected four model labels without a legend title')
    expected = [(x, y) for curve in curves for x, y in
                zip(curve['time_hours_seed_values'], curve['accuracy_seed_values'])]
    for (x, y), line in zip(expected, native_lines[8:]):
        require(np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y),
                'Rendered replicate differs from recorded values')
    figures = ROOT / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    vector = figures / (STEM + '.svg')
    figure.figure.savefig(str(vector), format='svg', facecolor=figure.background)
    viewbox = [float(v) for v in ET.fromstring(vector.read_bytes()).get('viewBox').split()]
    require(viewbox == [0, 0, 440, 300], 'Export differs from original paper figure dimensions')
    cairosvg.svg2png(url=str(vector), write_to=str(vector.with_suffix('.png')),
                    output_width=round(STYLE['image_size'][0] * STYLE['dpi'] / 72))
    cairosvg.svg2pdf(url=str(vector), write_to=str(vector.with_suffix('.pdf')))
    figure.close()
    with (ROOT / 'merged-validation-values.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exports[0]))
        writer.writeheader()
        writer.writerows(exports)
    (ROOT / 'merged-validation-values.json').write_text(json.dumps(curves, indent=2, allow_nan=False) + '\n')
    catalog = [dict(stem=STEM, title='Combined-readout validation · individual replicates', kind='merged_validation',
                    uncertainty='All three individual replicates shown; no statistical bands')]
    (ROOT / 'merged-validation-catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    receipt = dict(status='passed', renderer_sha256=sha(__file__), original_input_hashes=manifest, zmatrix_input_hashes=zmanifest,
        original_18_replicate_curves_unchanged=True, original_6_completion_guides_unchanged=True,
        style=STYLE, condition_count=8, run_count=24, epochs_per_run=100,
        readout='both', number_of_mean_curves=0, number_of_replicate_curves=24,
        number_of_vertical_epoch100_guides=8, rendered_accuracy_range_bands=0,
        native_replicate_points_verified=2400, native_completion_guides_verified=8,
        legend_inside_free_space=True, legend_clear_of_curves_and_completion_guides=True,
        original_paper_size_points=[440, 300], exported_viewbox=viewbox,
        axes_labels=['Time (hours)', 'Accuracy'],
        legend_bounds_in_data=[legend_data.x0, legend_data.y0, legend_data.x1, legend_data.y1],
        legend_bounds_in_svg_points=legend_svg_bounds, legend_labels=[LABELS[scope] for scope in SCOPES],
        legend_line_styles=['-', '-', '-', '-'], epoch100_guide_style=':',
        x_basis='Each replicate own cumulative measured train-and-validation hours',
        epoch100_line_basis='Mean of three replicate maximum elapsed times after 100 epochs',
        y_basis='Each replicate observed validation accuracy',
        no_smoothing=True, no_interpolation=True, no_sd_or_confidence_intervals=True,
        no_checkpoint_selection_markers=True, no_discrete_error_bars=True, filled_range_bands=False,
        outputs={STEM + '.' + ext: sha(figures / (STEM + '.' + ext)) for ext in ('svg', 'pdf', 'png')})
    (ROOT / 'merged-validation-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('condition_count', 'number_of_replicate_curves',
        'number_of_vertical_epoch100_guides', 'rendered_accuracy_range_bands')}))


if __name__ == '__main__':
    main()
