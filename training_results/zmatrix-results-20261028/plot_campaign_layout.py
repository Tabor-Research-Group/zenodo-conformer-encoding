"""Plot the completed pure-Z-matrix R/S campaign through native McUtils.

No training package is imported. Incomplete/failed campaigns produce no plots;
completed near-chance or collapsed trajectories remain included.
"""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

FACTORS = ('scope', 'precision', 'lr_profile', 'feature_mode', 'readout')
LEVELS = (('zmatrix',), ('decoded', 'psi8'), ('source', 'source_lr10'),
          ('full', 'reduced52'), ('both', 'z_alpha'))
BASE = dict(scope='zmatrix', precision='decoded', lr_profile='source', feature_mode='full', readout='both')
METRICS = ('train_accuracy', 'validation_accuracy', 'train_bce', 'validation_bce')
PURPLE, ALTERNATIVE = '#663399', '#B775B8'
COMPARISONS = (
    ('precision', 'precision', ('decoded', 'psi8'), ('Reconstructed', 'Extra 8-bit pass')),
    ('lr_profile', 'learning_rate', ('source', 'source_lr10'), ('Original learning rate', 'Learning rate / 10')),
    ('feature_mode', 'chemical_features', ('full', 'reduced52'), ('Full descriptors', 'Reduced descriptors')),
    ('readout', 'readout', ('both', 'z_alpha'), ('Combined readout', 'Torsion latent readout')),
)
DEFAULT_MCUTILS = '/scratch/user/maboyer/llm/encoding/chiro-full-factorial-20261024/native-current-b243e775'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def completed_snapshot(root):
    pointer = root / 'analysis/current.json'
    if not pointer.exists():
        return None, dict(status='incomplete', reason='No pooled snapshot exists', plots_written=False)
    current = json.loads(pointer.read_text())
    path = (root / current['snapshot']).resolve()
    require(root in path.parents and path.parent == root / 'analysis/snapshots', 'Snapshot escapes analysis directory')
    summary = json.loads((path / 'summary.json').read_text())
    if (summary.get('status_counts') != {'complete': 48}
            or summary.get('training_endpoints') != 48 or summary.get('test_endpoints') != 48):
        return None, dict(status='incomplete', reason='All 48 completed training and test results are required',
                          status_counts=summary.get('status_counts'), plots_written=False)
    require(summary.get('planned_tasks') == 48 and summary.get('planned_conditions') == 16,
            'Campaign grid differs from 48 runs/16 conditions')
    require(not summary['artifact_errors'] and not summary['sampler_mismatches']
            and not summary['initialization_mismatches'], 'Pooled integrity findings require review')
    require(current['result_fingerprint'] == summary['result_fingerprint'], 'Snapshot pointer and summary differ')
    manifest = json.loads((path / 'files-sha256.json').read_text())
    for name, expected in manifest.items():
        require(Path(name).name == name and sha(path / name) == expected, 'Immutable pooled file changed: ' + name)
    return (path, current, summary, manifest), None


def load_histories(path, np):
    with (path / 'long-history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in FACTORS)].append(row)
    require(set(groups) == set(itertools.product(*LEVELS)) and len(rows) == 4800,
            'Every condition, seed, and epoch must remain included')
    data = {}
    expected = list(itertools.product((1, 2, 3), range(1, 101)))
    for key, records in groups.items():
        ordered = sorted(records, key=lambda row: (int(row['seed']), int(row['epoch'])))
        require([(int(row['seed']), int(row['epoch'])) for row in ordered] == expected, 'Missing or duplicate epochs')
        require(all(int(row['optimizer_updates']) == 3442 * int(row['epoch']) for row in ordered), 'Optimizer exposure differs')
        values = {metric: np.asarray([float(row[metric]) for row in ordered]).reshape(3, 100) for metric in METRICS}
        seconds = np.asarray([float(row['epoch_seconds']) for row in ordered]).reshape(3, 100)
        require(all(np.isfinite(value).all() for value in values.values()) and np.isfinite(seconds).all()
                and (seconds > 0).all(), 'Nonfinite metrics or invalid observed timing')
        require(all(((values[m] >= 0) & (values[m] <= 1)).all() for m in METRICS if m.endswith('accuracy')),
                'Accuracy outside its physical domain')
        require((values['train_bce'] >= 0).all() and (values['validation_bce'] >= 0).all(), 'Negative BCE')
        values['time_hours'] = np.cumsum(seconds, axis=1) / 3600
        data[key] = values
    return data


def render(root, source, output):
    snapshot, incomplete = completed_snapshot(root)
    if incomplete is not None:
        print(json.dumps(incomplete), flush=True)
        return 0
    path, current, summary, manifest = snapshot
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    for variable, leaf in [('MPLCONFIGDIR', 'mpl-modern'), ('NUMBA_CACHE_DIR', 'numba-modern'), ('XDG_CACHE_HOME', 'xdg-modern')]:
        cache = root / 'runtime-cache' / leaf
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    if sys.platform == 'darwin':
        os.environ.setdefault('DYLD_FALLBACK_LIBRARY_PATH', '/opt/homebrew/lib')
    sys.path.insert(0, str(source))
    import numpy as np
    import McUtils
    from McUtils.Plots import Graphics, GraphicsGrid, Plot, PlotLegend, Styled
    import cairosvg
    require(source in Path(McUtils.__file__).resolve().parents, 'Unexpected McUtils import')
    data = load_histories(path, np)
    output.mkdir(parents=True, exist_ok=True)
    final = output / ('snapshot-' + current['result_fingerprint'][:24])
    with (output / '.plot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.exists():
            receipt = json.loads((final / 'render-receipt.json').read_text())
            require(receipt['result_fingerprint'] == current['result_fingerprint'], 'Plot fingerprint collision')
            require(all(sha(final / name) == expected for name, expected in receipt['output_hashes'].items()), 'Existing plot output changed')
            print(json.dumps(dict(status='already_rendered', output=str(final), figures=13)), flush=True)
            return 0
        staging = Path(tempfile.mkdtemp(prefix='.render-', dir=str(output)))
        try:
            figures = staging / 'figures'
            figures.mkdir()
            theme = ('mccoy', {'font': {'family': 'DejaVu Sans', 'size': 10}, 'svg': {'fonttype': 'path'}})
            x = np.arange(1, 101) * 3.442
            all_accuracy = np.concatenate([values[name].ravel() for values in data.values() for name in ('train_accuracy', 'validation_accuracy')])
            accuracy_limits = (max(0., min(.475, float(all_accuracy.min()) - .01)), 1.01)
            bce_max = max(float(values[name].max()) for values in data.values() for name in ('train_bce', 'validation_bce'))
            bce_top = max(.1, float(np.ceil(bce_max * 1.05 * 10) / 10))
            records, checks, plotted = [], [], {}
            accuracy_coverage, bce_coverage = Counter(), Counter()

            def condition(**changes):
                values = dict(BASE, **changes)
                return tuple(values[name] for name in FACTORS)

            def line(panel, key, metric, pattern, color=PURPLE, times=False, expected=None):
                times_array = data[key]['time_hours'] if times else [x] * 3
                for xs, ys in zip(times_array, data[key][metric]):
                    Plot(xs, ys, figure=panel, color=color, linestyle=pattern, linewidth=1.0,
                         plot_style={'alpha': .86, 'zorder': 3})
                    expected.append((xs, ys))
                plotted['__'.join(key + (metric,))] = dict(condition=dict(zip(FACTORS, key)), metric=metric,
                    seeds=[1, 2, 3], epochs=list(range(1, 101)), optimizer_updates=(np.arange(1, 101) * 3442).tolist(),
                    values=data[key][metric].tolist(), time_hours=data[key]['time_hours'].tolist())

            def axes(panel, metric='accuracy'):
                panel.aspect_ratio = 'auto'
                panel.axes_labels = ['Optimizer updates (×10³)', 'Accuracy' if metric == 'accuracy' else 'Binary cross-entropy']
                panel.plot_range = [(0, 350), accuracy_limits if metric == 'accuracy' else (0, bce_top)]
                panel.ticks = [[0, 100, 200, 300], [.5, .75, 1.] if metric == 'accuracy' else np.linspace(0, bce_top, 5).tolist()]

            def legend(fig, panel, items, ncols):
                return panel.axes.legend(handles=PlotLegend.construct(items), loc='upper center',
                    bbox_to_anchor=(.52, .982), bbox_transform=fig.figure.obj.transFigure, ncol=ncols,
                    fontsize=9.3, frameon=False, borderaxespad=0, handlelength=2.1,
                    columnspacing=1.2, handletextpad=.4)

            def save(fig, stem, content, leg, expected, grid=False):
                fig.prep_show()
                if grid:
                    fig.figure.set_figure_spacings((.38, .80))
                native = fig.figure.obj
                native.canvas.draw()
                renderer = native.canvas.get_renderer()
                box = leg.get_window_extent(renderer)
                require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= native.bbox.width + 1
                        and box.y1 <= native.bbox.height + 1, 'Legend outside figure: ' + stem)
                lines = [item for ax in native.axes for item in ax.lines]
                require(len(lines) == len(expected) and all(len(ax.collections) == 0 for ax in native.axes), 'Unexpected means/bands: ' + stem)
                for item, (xs, ys) in zip(lines, expected):
                    require(np.array_equal(item.get_xdata(), xs) and np.array_equal(item.get_ydata(), ys), 'Rendered values differ from raw observations')
                for ax in native.axes:
                    for text in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
                        if text.get_visible() and text.get_text():
                            bound = text.get_window_extent(renderer)
                            require(bound.x0 >= -2 and bound.y0 >= -2 and bound.x1 <= native.bbox.width + 2
                                    and bound.y1 <= native.bbox.height + 2, 'Label outside canvas: ' + stem + ': ' + text.get_text())
                    if ax.title.get_text():
                        require(not box.overlaps(ax.title.get_window_extent(renderer)), 'Legend overlaps title')
                vector = figures / (stem + '.svg')
                fig.figure.savefig(str(vector), format='svg', facecolor=fig.background)
                cairosvg.svg2pdf(url=str(vector), write_to=str(vector.with_suffix('.pdf')))
                cairosvg.svg2png(url=str(vector), write_to=str(vector.with_suffix('.png')), scale=2)
                records.append(dict(stem=stem, **content))
                checks.append(dict(stem=stem, native_lines=len(lines), raw_native_values_verified=True, no_bands=True,
                                   labels_inside_canvas=True, legend_inside_canvas=True))
                fig.close()
                print(stem, flush=True)

            # The only averaged quantity is each condition's mean completion time.
            fig = Graphics(image_size=(440, 300), padding=((70, 20), (62, 72)), theme=theme)
            fig.image_size = (440, 300)
            expected, items = [], []
            main_keys = []
            palette = {('decoded', 'source'): '#663399', ('decoded', 'source_lr10'): '#9272AD',
                       ('psi8', 'source'): '#B14D8E', ('psi8', 'source_lr10'): '#D18CAF'}
            for precision, lr in itertools.product(('decoded', 'psi8'), ('source', 'source_lr10')):
                key = condition(precision=precision, lr_profile=lr)
                main_keys.append(key)
                color = palette[(precision, lr)]
                finish = float(data[key]['time_hours'][:, -1].mean())
                Plot([finish, finish], accuracy_limits, figure=fig, color=color, linestyle=':', linewidth=2,
                     plot_style={'alpha': .75, 'zorder': 0, 'dash_capstyle': 'round'})
                expected.append(([finish, finish], accuracy_limits))
                label = ('Reconstructed' if precision == 'decoded' else 'Extra 8-bit pass') + (', LR' if lr == 'source' else ', LR/10')
                items.append(dict(marker='line', label=label, color=color, linestyle='-' if lr == 'source' else '--', lw=1.6))
            for key in main_keys:
                line(fig, key, 'validation_accuracy', '-' if key[2] == 'source' else '--', palette[(key[1], key[2])], True, expected)
            fig.aspect_ratio = 'auto'
            fig.axes_labels = ['Time (hours)', 'Accuracy']
            fig.plot_range = [(0, max(float(data[key]['time_hours'].max()) for key in main_keys) * 1.045), accuracy_limits]
            leg = legend(fig, fig, items, 2)
            save(fig, 'zmatrix_validation_time', dict(kind='validation_time', conditions=4, runs=12,
                completion_guides='Mean of three replicate maximum measured training-and-validation times at epoch 100',
                timing_scope='Measured loops only; excludes setup, feature preparation, checkpoint writes, and test inference'), leg, expected)

            for factor, suffix, levels, labels in COMPARISONS:
                fig = GraphicsGrid(nrows=1, ncols=2, theme=theme, subimage_size=(220, 155),
                                   padding=((80, 25), (62, 75)), spacings=(60, 65))
                expected = []
                for j, split in enumerate(('train', 'validation')):
                    panel = fig[0, j]
                    for level, pattern, color in zip(levels, ('-', '--'), (PURPLE, ALTERNATIVE)):
                        line(panel, condition(**{factor: level}), split + '_accuracy', pattern, color, expected=expected)
                    axes(panel)
                    panel.plot_label = Styled(split.title(), fontsize=10)
                items = [dict(marker='line', label=label, color=color, lw=1.4, linestyle=pattern)
                         for label, pattern, color in zip(labels, ('-', '--'), (PURPLE, ALTERNATIVE))]
                leg = legend(fig, fig[0, 0], items, 2)
                save(fig, 'zmatrix_' + suffix, dict(kind='matched', factor=factor, levels=levels, fixed=BASE, conditions=2, runs=6), leg, expected, True)

            for feature, readout, metric in itertools.product(('full', 'reduced52'), ('both', 'z_alpha'), ('accuracy', 'bce')):
                fig = GraphicsGrid(nrows=2, ncols=2, theme=theme, subimage_size=(220, 145),
                                   padding=((80, 25), (62, 79)), spacings=(60, 65))
                expected = []
                for i, precision in enumerate(('decoded', 'psi8')):
                    for j, lr in enumerate(('source', 'source_lr10')):
                        panel = fig[i, j]
                        key = condition(precision=precision, lr_profile=lr, feature_mode=feature, readout=readout)
                        for split, pattern in [('train', '-'), ('validation', '--')]:
                            line(panel, key, split + '_' + metric, pattern, expected=expected)
                        (accuracy_coverage if metric == 'accuracy' else bce_coverage)[key] += 1
                        axes(panel, metric)
                        panel.plot_label = Styled(('Reconstructed' if precision == 'decoded' else 'Extra 8-bit pass') + '\n'
                            + ('Original learning rate' if lr == 'source' else 'Learning rate / 10'), fontsize=9.5)
                items = [dict(marker='line', label=label, color=PURPLE, linestyle=pattern, lw=1.4)
                         for label, pattern in [('Training', '-'), ('Validation', '--')]]
                leg = legend(fig, fig[0, 0], items, 2)
                stem = 'zmatrix_all_' + feature + '_' + readout + '_' + metric
                save(fig, stem, dict(kind='exhaustive', feature_mode=feature, readout=readout, metric=metric, conditions=4, runs=12), leg, expected, True)
            require(len(records) == 13 and set(accuracy_coverage) == set(data) and set(accuracy_coverage.values()) == {1}
                    and set(bce_coverage) == set(data) and set(bce_coverage.values()) == {1}, 'Incomplete figure coverage')
            save_json(staging / 'figure-catalog.json', records)
            save_json(staging / 'plotted-values.json', plotted)
            output_hashes = {str(item.relative_to(staging)): sha(item) for item in sorted(staging.rglob('*')) if item.is_file()}
            receipt = dict(status='passed', result_fingerprint=current['result_fingerprint'], snapshot=str(path),
                source_snapshot_hashes=manifest, renderer_sha256=sha(__file__), mcutils_root=str(source),
                raw_runs=48, conditions=16, epochs_per_run=100, no_smoothing=True, no_bands=True,
                all_completed_runs_included=True, test_trajectories_plotted=False,
                accuracy_limits=accuracy_limits, bce_max=bce_max, bce_display_max=bce_top,
                precision_caveat='Both arms use already discrete stored-Z-matrix geometry; an extra eight-bit pass may change only Cartesian reevaluation roundoff.',
                checks=checks, output_hashes=output_hashes)
            save_json(staging / 'render-receipt.json', receipt)
            staging.rename(final)
            save_json(output / 'current.json', dict(status='rendered', figures=13, path=str(final), result_fingerprint=current['result_fingerprint']))
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    print(json.dumps(dict(status='rendered', figures=13, output=str(final))), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mcutils-root', type=Path, default=Path(DEFAULT_MCUTILS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    return render(root, args.mcutils_root.resolve(), (args.output or root / 'analysis/figures-zmatrix').resolve())


if __name__ == '__main__':
    raise SystemExit(main())
