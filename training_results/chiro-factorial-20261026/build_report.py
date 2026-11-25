"""Assemble the figure index and readable analysis from verified numeric artifacts."""
from pathlib import Path
import csv
import html
import json
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'analysis/snapshots/snapshot-709d09ec5eb0812baed37c2b'
FIGURES = ROOT / json.loads((ROOT / 'analysis/latest-figures.json').read_text())['directory']


def esc(value):
    return html.escape(str(value), quote=True)


def rel(path):
    return str(Path(path).relative_to(ROOT))


def read_csv(name):
    with (SNAPSHOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def figure(path, title, caption):
    path = Path(path)
    links = ' · '.join('<a href="' + esc(rel(path.with_suffix('.' + suffix))) + '">' + suffix.upper() + '</a>'
                       for suffix in ('png', 'svg', 'pdf'))
    return ('<figure><h3>' + esc(title) + '</h3><a href="' + esc(rel(path.with_suffix('.svg'))) + '">'
            '<img loading="lazy" src="' + esc(rel(path)) + '" alt="' + esc(title) + '"></a>'
            '<figcaption>' + esc(caption) + '<br>' + links + '</figcaption></figure>')


def main():
    summary = json.loads((SNAPSHOT / 'summary.json').read_text())
    assert summary['experiment_outcome'] == 'all_succeeded'
    assert summary['training_endpoints'] == summary['test_endpoints'] == 144
    rows = read_csv('condition-means.csv')
    lookup = {(r['condition'], r['metric']): r for r in rows}
    tests = [r for r in rows if r['metric'] == 'test_accuracy']
    with (ROOT / 'independent-paired-effects.csv').open(newline='') as stream:
        effects = list(csv.DictReader(stream))
    labels = {('scope','minimal','augmented'): 'Augmented − Minimal',
              ('scope','minimal','complete'): 'Complete − Minimal',
              ('scope','augmented','complete'): 'Complete − Augmented',
              ('precision','decoded','psi8'): 'Additional 8-bit torsions − tag-decoded',
              ('lr_profile','source','source_lr10'): 'Learning rate / 10 − original rate',
              ('feature_mode','full','reduced52'): 'Reduced − full chemical descriptors',
              ('readout','both','z_alpha'): 'Torsion readout − combined readout'}
    effect_html = []
    for r in effects:
        name = labels[(r['factor'], r['level_a'], r['level_b'])]
        effect_html.append('<tr><td>' + esc(name) + '</td><td>' + str(r['fixed_conditions']) + '</td><td>'
                           + f"{float(r['mean_effect_pp']):+.3f} ± {float(r['seed_sd_of_design_average_pp']):.3f}"
                           + '</td><td>' + f"{float(r['min_cell_mean_pp']):+.3f} to {float(r['max_cell_mean_pp']):+.3f}"
                           + '</td></tr>')
    endpoint_html = []
    endpoint_md = []
    scope_order = {'minimal':0, 'augmented':1, 'complete':2}
    tests.sort(key=lambda r:(scope_order[r['scope']], r['precision'], r['lr_profile'], r['feature_mode'], r['readout']))
    for r in tests:
        val = lookup[(r['condition'], 'best_validation_accuracy')]
        seed = json.loads(r['seed_values'])
        values = '; '.join('Seed ' + s + ': ' + f'{100*v:.3f}%' for s,v in seed.items())
        values_test = f"{100*float(r['mean']):.2f} ± {100*float(r['sample_sd']):.2f}"
        values_val = f"{100*float(val['mean']):.2f} ± {100*float(val['sample_sd']):.2f}"
        names = [r['scope'].title(), 'Tag-decoded' if r['precision']=='decoded' else '+ 8-bit torsions',
                 'Original' if r['lr_profile']=='source' else 'Original / 10',
                 'Full' if r['feature_mode']=='full' else 'Reduced',
                 'Combined' if r['readout']=='both' else 'Torsion output', values_val, values_test]
        attrs = ' '.join('data-' + k + '="' + esc(r[k]) + '"' for k in
                         ('scope','precision','lr_profile','feature_mode','readout'))
        endpoint_html.append('<tr class="condition" ' + attrs + ' title="' + esc(values) + '">'
                             + ''.join('<td>' + esc(x) + '</td>' for x in names) + '</tr>')
        endpoint_md.append('| ' + ' | '.join(names) + ' |')
    controls = ''
    for key, title, options in [
        ('scope','Torsions',[('minimal','Minimal'),('augmented','Augmented'),('complete','Complete')]),
        ('precision','Geometry',[('decoded','Tag-decoded'),('psi8','Additional 8-bit torsions')]),
        ('lr_profile','Learning rate',[('source','Original'),('source_lr10','Original / 10')]),
        ('feature_mode','Chemistry',[('full','Full'),('reduced52','Reduced')]),
        ('readout','Readout',[('both','Combined'),('z_alpha','Torsion output')])]:
        controls += '<label>' + title + '<select class="filter" data-factor="' + key + '"><option value="">All</option>'
        controls += ''.join('<option value="' + a + '">' + b + '</option>' for a,b in options) + '</select></label>'
    curve_base = rel(FIGURES)
    curves_html = ''
    for metric,title in [('accuracy','Training and validation accuracy'),('bce','Training and validation BCE')]:
        path = FIGURES / ('pooled_full_decoded_' + metric + '.png')
        curves_html += '<figure><h3>' + title + '</h3><a id="curve-open-' + metric + '" href="' + esc(rel(path.with_suffix('.svg'))) + '">'
        curves_html += '<img id="curve-' + metric + '" src="' + esc(rel(path)) + '" alt="' + title + '"></a><figcaption>'
        curves_html += 'Solid: training. Dashed: resampled validation. Colors: torsion scope. Shading: sample SD across three seeds. '
        curves_html += 'Panels hold chemistry and precision fixed and cross learning rate with readout. '
        curves_html += ' · '.join('<a id="curve-' + metric + '-' + ext + '" href="' + esc(rel(path.with_suffix('.' + ext))) + '">' + ext.upper() + '</a>' for ext in ('png','svg','pdf'))
        curves_html += '</figcaption></figure>'
    endpoint_figure = figure(ROOT/'factor-figures/all48_test_accuracy_seed_sd.png', 'Every tested condition',
        'All 48 conditions, three seeds each. Small dots are individual seeds; larger markers and bars are mean ± sample SD. Circles use the original learning rate; squares use the tenfold smaller rate. Test checkpoints are selected by validation accuracy.')
    binary_figure = figure(ROOT/'factor-figures/binary_factor_test_effects_paired_seed_sd.png', 'Matched effects of the four binary factors',
        'Each row holds every other factor fixed. Small dots are the three same-seed differences; bars show their mean ± sample SD. The vertical zero line indicates no measured change. Units are percentage points.')
    scope_figure = figure(ROOT/'factor-figures/scope_test_effects_paired_seed_sd.png', 'Matched torsion-scope comparisons',
        'The three scope pairs are compared separately at every matched precision, learning rate, descriptor set, and readout. Mixed signs show why one global ranking would be misleading.')
    diagnostics = ''
    specs = [
        ('learning_rate_convergence_example', 'Training fit and validation generalization',
         'Labelled example: Complete, tag-decoded geometry, full descriptors, combined readout. Mean ± sample SD across three seeds. Accuracy is zoomed to 90–100%; the full learning range is available in the curve grids. Both learning rates fit training well, while the smaller rate has a larger validation gap.'),
        ('learning_onset_individual_seeds', 'Delayed learning is visible at the individual-seed level',
         'Full chemical descriptors, tag-decoded geometry, original learning rate. Individual validation trajectories and means preserve all 100 epochs without smoothing. Complete with the torsion readout first sustains 95% validation accuracy at epochs 55, 65, and 47; Minimal does so at 5, 6, and 6. Here “sustained” means five consecutive logged epochs, a descriptive diagnostic rather than a training gate.'),
        ('retained_loss_outlier_seed_traces', 'High accuracy can coexist with unstable probability loss',
         'Complete, additional 8-bit torsions, reduced chemical descriptors, combined readout, original learning rate. Every seed is retained. Seed 3 ends at 98.025% validation accuracy with BCE 0.575; its highest-accuracy checkpoint has BCE 0.866. The earlier minimum BCE was 0.056. Selection remains highest validation accuracy, with no post-hoc loss-based reselection. BCE alone does not identify a calibration mechanism, particularly with resampled validation.')]
    for stem,title,caption in specs:
        diagnostics += figure(ROOT/'convergence-figures'/(stem+'.png'),title,caption)
    txt = (ROOT/'independent-analysis.txt').read_text()
    html_text = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChIRo R/S factorial training analysis</title><style>
body{font:17px/1.55 system-ui,-apple-system,sans-serif;color:#152033;background:#fff;margin:0}main{max-width:1380px;margin:auto;padding:36px 32px 80px}h1{font-size:34px;line-height:1.2}h2{margin-top:52px;border-top:1px solid #dde3eb;padding-top:24px}h3{font-size:21px}a{color:#174c96}nav{display:flex;gap:20px;flex-wrap:wrap;margin:25px 0}p,li{max-width:1050px}.meta{color:#526074}.cards{display:flex;gap:20px;flex-wrap:wrap}.card{padding:15px 24px;background:#f3f6fa;border-radius:8px}.card strong{font-size:25px;display:block}figure{margin:32px 0 45px;padding:18px 20px;border:1px solid #dbe2ec;border-radius:8px}figure img{display:block;width:100%;height:auto;background:white}figcaption{color:#465469;font-size:15px;margin-top:12px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border-bottom:1px solid #dde3eb;padding:10px 12px;text-align:left;vertical-align:top}th{background:#f3f6fa}td:not(:first-child){font-variant-numeric:tabular-nums}.scroll{overflow:auto}label{display:inline-block;font-size:14px;margin:0 16px 16px 0}select{display:block;font:inherit;padding:7px 9px;min-width:130px;border:1px solid #aebccc;border-radius:5px;background:white}details{margin:20px 0}summary{cursor:pointer;font-weight:600}pre{font:13px/1.6 ui-monospace,monospace;background:#f3f6fa;padding:18px;white-space:pre-wrap;overflow:auto}.note{padding:14px 20px;background:#f4f7fb;border-left:4px solid #47679e}code{font-size:.9em;overflow-wrap:anywhere}.findings li{margin:12px 0}.download{display:inline-block;background:#0b367b;color:white;padding:10px 17px;border-radius:6px;text-decoration:none}@media(max-width:800px){main{padding:22px 14px}h1{font-size:27px}figure{padding:10px}.card{padding:12px}table{min-width:790px}}@media print{nav,select,.download{display:none}figure{break-inside:avoid}main{padding:0}h2{break-before:auto}}
</style><main><h1>ChIRo R/S training across 48 conditions</h1><p class="meta">October 26, 2026 · 144 training runs · all runs completed</p>
<div class="cards"><div class="card"><strong>48</strong>factor settings</div><div class="card"><strong>3</strong>seeds per setting</div><div class="card"><strong>100</strong>epochs per run</div><div class="card"><strong>69,719</strong>test conformers per run</div></div>
<nav><a href="#findings">Findings</a><a href="#curves">Learning curves</a><a href="#effects">Paired effects</a><a href="#diagnostics">Learning behavior</a><a href="#conditions">All conditions</a><a href="#methods">Interpretation</a><a href="#reproduce">Data and code</a></nav>
<a class="download" href="chiro-factorial-results.zip">Download figures, analysis, data and code</a>
<h2 id="findings">What the experiment shows</h2><ul class="findings">
<li><strong>All three torsion scopes learn successfully by the end of these runs.</strong> There are no near-chance final endpoints or observed-collapse endpoint flags. Learning speed can differ substantially: Complete can remain near chance for many epochs, especially with the torsion readout.</li>
<li><strong>The learning-rate change has the largest consistent test effect.</strong> Reducing the original rate tenfold lowers held-out accuracy in all 24 matched conditions and all 72 seed pairs, by 1.30–2.27 percentage points in condition means. Both rates still fit training at about 99%; the smaller rate leaves a larger validation/test gap at the tested exposure.</li>
<li><strong>Additional 8-bit torsion quantization has a mixed, small design-average effect.</strong> Twelve matched condition means improve and twelve worsen. The average is +0.019 percentage points, with conditional effects from −0.537 to +0.695. This is neither a universal benefit nor an equivalence claim.</li>
<li><strong>There is no universal Minimal, Augmented, Complete ranking.</strong> Chemistry and readout alter the scope comparison. Reducing chemical descriptors shifts Complete−Minimal downward in every one of the eight matched remaining-factor condition means, while other contrasts have mixed directions.</li>
<li><strong>Accuracy and BCE provide different information.</strong> A high-accuracy Complete run has unstable validation BCE and remains visible in all summaries. These curves show finite-budget outcomes; 100 epochs do not establish eventual convergence.</li></ul>
<h3>Balanced descriptive effects on test accuracy</h3><p>Each average first combines matched backgrounds within a seed, then summarizes the three seed averages. Its SD is across those three seed averages, not across the 16 or 24 correlated factorial backgrounds. The rightmost column preserves the conditional variation that an overall average can hide.</p>
<div class="scroll"><table><thead><tr><th>Contrast</th><th>Matched backgrounds</th><th>Average ± seed SD (percentage points)</th><th>Range of condition-mean changes</th></tr></thead><tbody>''' + ''.join(effect_html) + '''</tbody></table></div>
<h2 id="curves">Training and validation curves</h2><p>All 48 conditions are covered by eight grids: chemical descriptors × geometry precision × accuracy/BCE. In each grid, rows change the readout and columns change the learning rate. There is no smoothing. Epoch 100 is 344,200 optimizer updates.</p>
<div><label>Chemical descriptors<select id="curve-chemistry"><option value="full">Full</option><option value="reduced52">Reduced</option></select></label><label>Geometry precision<select id="curve-precision"><option value="decoded">Tag-decoded geometry</option><option value="psi8">Additional 8-bit torsions</option></select></label></div>''' + curves_html + '''
<h2 id="effects">Held-out results and paired effects</h2>''' + endpoint_figure + binary_figure + scope_figure + '''
<h2 id="diagnostics">Learning behavior beyond the final accuracy</h2>''' + diagnostics + '''
<h2 id="conditions">All 48 conditions</h2><p>Accuracy values are percentages, mean ± sample SD across three seeds. “Selected validation” is the maximum resampled-validation accuracy for each seed; its checkpoint alone supplies that seed’s held-out test result. Hover over a row to see the three test accuracies.</p>''' + controls + '''<p id="row-count" class="meta">48 conditions</p><div class="scroll"><table><thead><tr><th>Scope</th><th>Geometry</th><th>LR</th><th>Chemistry</th><th>Readout</th><th>Selected validation (%)</th><th>Held-out test (%)</th></tr></thead><tbody>''' + ''.join(endpoint_html) + '''</tbody></table></div>
<h2 id="methods">What the comparisons and error bars mean</h2>
<p>The crossed design is 3 torsion selections × 2 precisions × 2 learning rates × 2 descriptor sets × 2 readouts × 3 seeds. The original population is retained: 326,865 eligible training records, 70,099 validation records and 69,719 held-out test records. The original paired sampler draws 16 anchors/32 graphs per batch, giving 3,442 optimizer updates per epoch. Checkpoints are chosen by maximum validation accuracy, first tie; test results do not choose the checkpoint or the conditions shown.</p>
<div class="note"><strong>Uncertainty unit: the training seed.</strong> Curve bands and endpoint bars are sample SD across three training replicates. Paired bars are SD across the three same-seed differences at fixed background factors. Individual seed points or trajectories remain visible. These are not SEMs, confidence intervals, significance tests or molecular sampling errors. Three seeds provide limited evidence about optimization variability.</div>
<ul><li><strong>Tag-decoded geometry:</strong> every arm begins with the compressed source-tag reconstruction. There is no untouched Cartesian-geometry arm in this factorial.</li><li><strong>Additional 8-bit torsions:</strong> only the selected model torsions receive extra uniform quantization. Distance and bond-angle arrays remain fixed. The experiment measures the incremental torsion-discretization effect, not the total effect of compact storage.</li><li><strong>Full/reduced chemistry:</strong> the reduced set keeps element, total degree, total H count and aromaticity, and zeroes the other atom fields and all bond descriptors at unchanged native 52/14 widths. Connectivity and the graph architecture remain present.</li><li><strong>Combined/torsion readout:</strong> the combined 256-component readout contains graph, distance, angle and torsion representations. The 64-component torsion output is still graph/chemistry conditioned; it does not represent four isolated geometric numbers without structural context.</li><li><strong>Coverage:</strong> all runs, all seeds, and the audited rare reconstruction cases are retained. No unfavorable run is removed. No universal ordering, statistical equivalence, or asymptotic-convergence conclusion is asserted.</li></ul>
<details><summary>Full numerical interpretation and qualifications</summary><pre>''' + esc(txt) + '''</pre></details>
<h2 id="reproduce">Data, vector figures, and reproducible code</h2><p>Every scientific figure uses native McUtils plotting and is exported as PNG, SVG and PDF. Links below each image open full-resolution/vector versions. The local snapshot matches all nine original SHA-256 entries. The numerical audit independently reconstructs all 432 condition means and 3,888 seed-level metric contrasts.</p>
<ul><li><a href="independent-analysis.txt">Readable analysis</a> · <a href="independent-analysis.json">Full numerical analysis</a> · <a href="independent-paired-effects.csv">Balanced effect table</a></li><li><a href="''' + esc(rel(SNAPSHOT/'condition-means.csv')) + '''">All condition means</a> · <a href="''' + esc(rel(SNAPSHOT/'paired-seed-differences.csv')) + '''">Same-seed differences</a> · <a href="''' + esc(rel(SNAPSHOT/'long-history.csv')) + '''">Every logged epoch</a></li><li><a href="REPRODUCE.txt">Reproduction instructions</a> · <a href="plot_pool.py">Original curve renderer</a> · <a href="factor_plots.py">Endpoint and contrast renderer</a> · <a href="convergence_plots.py">Diagnostic renderer</a> · <a href="analyze_conditions.py">Numerical analysis</a></li><li><a href="transport-verification.json">Exact input verification</a> · <a href="''' + esc(rel(SNAPSHOT/'files-sha256.json')) + '''">Original snapshot hashes</a> · <a href="bundle-manifest.json">Artifact manifest</a></li></ul>
<details><summary>Plotting issue recorded for review</summary><p>A native McUtils log-axis accessor issue was encountered. The delivered diagnostics use linear axes, and no package source change or adapter was applied. <a href="issues/mcutils-mpl-axis-scale.txt">Issue description</a> · <a href="issues/mcutils-mpl-axis-scale.patch">Unapplied targeted draft patch</a>.</p></details>
<p class="meta">Source snapshot: snapshot-709d09ec5eb0812baed37c2b. These are held-out results from fixed validation-selected checkpoints; the 69,719 conformers are not independent training replicates. Native receipts and bindings were checked; large model/prediction binaries were not freshly rehashed in this analysis.</p>
<script>
const curveBase = ''' + json.dumps(curve_base) + ''';
function curves(){const chem=document.getElementById('curve-chemistry').value,precision=document.getElementById('curve-precision').value;for(const m of ['accuracy','bce']){const stem=curveBase+'/pooled_'+chem+'_'+precision+'_'+m;document.getElementById('curve-'+m).src=stem+'.png';document.getElementById('curve-open-'+m).href=stem+'.svg';for(const e of ['png','svg','pdf'])document.getElementById('curve-'+m+'-'+e).href=stem+'.'+e;}}
document.getElementById('curve-chemistry').addEventListener('change',curves);document.getElementById('curve-precision').addEventListener('change',curves);
function filterRows(){const fs=[...document.querySelectorAll('.filter')];let n=0;document.querySelectorAll('tr.condition').forEach(row=>{const show=fs.every(f=>!f.value||row.getAttribute('data-'+f.dataset.factor)===f.value);row.hidden=!show;if(show)n++;});document.getElementById('row-count').textContent=n+' of 48 conditions';}
document.querySelectorAll('.filter').forEach(e=>e.addEventListener('change',filterRows));
</script></main></html>'''
    (ROOT/'report.html').write_text(html_text)
    md = '# ChIRo R/S factorial analysis\n\n144 runs; 48 conditions; three seeds each; 100 epochs and full held-out testing.\n\n'
    md += txt + '\n\n## All condition endpoints\n\n'
    md += '| Scope | Geometry | LR | Chemistry | Readout | Selected validation (%) | Held-out test (%) |\n'
    md += '|---|---|---|---|---|---:|---:|\n' + '\n'.join(endpoint_md) + '\n'
    (ROOT/'analysis.md').write_text(md)
    reproduce = '''Reproducing the figures and analysis

All scientific plots use native McUtils. No model fitting or checkpoint reselection occurs here.
Use a Python environment containing NumPy, matplotlib and native McUtils. On this Mac:

PYTHON=/Users/Mark/Documents/Postdoc/Development/.codex/envs/python3.11-codex/bin/python
MCUTILS=/Users/Mark/Documents/UW/Research/Development/McUtils
cd <this artifact directory>

/usr/bin/arch -arm64 "$PYTHON" -B analyze_conditions.py --root .
/usr/bin/arch -arm64 "$PYTHON" -B plot_pool.py --root . --snapshot analysis/snapshots/snapshot-709d09ec5eb0812baed37c2b --mcutils-source "$MCUTILS"
/usr/bin/arch -arm64 "$PYTHON" -B factor_plots.py --inputs analysis/snapshots/snapshot-709d09ec5eb0812baed37c2b --output factor-figures --mcutils-source "$MCUTILS"
/usr/bin/arch -arm64 "$PYTHON" -B convergence_plots.py --inputs analysis/snapshots/snapshot-709d09ec5eb0812baed37c2b --output convergence-figures --mcutils-source "$MCUTILS"
/usr/bin/arch -arm64 "$PYTHON" -B build_report.py

The scripts put all plotting caches in this output directory. Scientific package sources,
training files, model weights and source data are not edited. The full nine-file snapshot
is included and verified by its original SHA-256 manifest. Supplemental plotted-value JSONs
preserve every mean, SD and individual-seed value behind the error bars.

Uncertainty: sample SD across three training seeds, or across three paired seed differences.
The balanced design-effect table first averages backgrounds within seed; its SD has n=3,
not 16/24 factorial cells and not 69,719 conformers. No formal significance/equivalence test.
'''
    (ROOT/'REPRODUCE.txt').write_text(reproduce)
    files = [ROOT/name for name in ('report.html','analysis.md','REPRODUCE.txt','REPRODUCE_ANALYSIS.txt',
             'build_report.py','plot_pool.py','factor_plots.py','convergence_plots.py','analyze_conditions.py',
             'independent-analysis.json','independent-analysis.txt','independent-paired-effects.csv',
             'transport-verification.json','analysis-reproduction-check.json') if (ROOT/name).exists()]
    files += [ROOT/'analysis/current.json',ROOT/'analysis/latest-figures.json']
    for folder in (SNAPSHOT,FIGURES,ROOT/'factor-figures',ROOT/'convergence-figures',ROOT/'issues'):
        if folder.exists():
            files += [p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.png','.svg','.pdf','.txt','.patch')]
    files = sorted(set(files))
    manifest = {'status':'complete','snapshot':rel(SNAPSHOT),'result_fingerprint':summary['result_fingerprint'],
        'replication_unit':'training seed; n=3','error_bars':'sample SD, paired sample SD for matched effects',
        'files_sha256':{rel(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(ROOT/'chiro-factorial-results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[ROOT/'bundle-manifest.json']:
            archive.write(path,rel(path))
    print(json.dumps({'status':'complete','report':str(ROOT/'report.html'),'bundled_files':len(files)+1,
                      'archive_bytes':(ROOT/'chiro-factorial-results.zip').stat().st_size},indent=2))


if __name__ == '__main__':
    main()
