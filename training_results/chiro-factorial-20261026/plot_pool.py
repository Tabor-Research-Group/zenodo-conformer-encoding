"""Native McUtils figures from a verified immutable seed-level pooling snapshot.

Run with a modern Python/McUtils environment, separately from legacy training.
No model fitting, prediction aggregation, or experiment selection occurs here.
"""
import argparse
import csv
import fcntl
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

STYLE = {
    'colors': {'complete': '#071F83', 'minimal': '#008000', 'augmented': '#D7478A'},
    'font_family': 'DejaVu Sans', 'font_size': 13, 'linewidth': 2.0,
    'subimage_size': (570, 325), 'spacings': (150, 190),
    'padding': ((90, 40), (85, 165)), 'panel_gaps': (.20, .27),
    'formats': ('png', 'svg', 'pdf'), 'dpi': 180,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name): os.unlink(name)


def load_snapshot(root, snapshot):
    snapshot = (root / snapshot).resolve()
    if root / 'analysis/snapshots' not in snapshot.parents:
        raise ValueError('Input must be an immutable analysis/snapshots directory beneath this root')
    manifest = json.loads((snapshot/'files-sha256.json').read_text())
    for name, expected in manifest.items():
        path = (snapshot/name).resolve()
        if path.parent != snapshot or sha(path) != expected:
            raise ValueError('Published pooled input differs: '+name)
    summary = json.loads((snapshot/'summary.json').read_text())
    with (snapshot/'learning-curve-seed-summary.csv').open(newline='') as stream:
        curves = list(csv.DictReader(stream))
    for row in curves:
        for key in ('epoch','optimizer_updates','n'): row[key]=int(row[key])
        for key in ('mean','sample_sd'): row[key]=float(row[key]) if row[key] else None
    conditions = json.loads((snapshot/'condition-means.json').read_text())
    contrasts = json.loads((snapshot/'paired-contrasts.json').read_text())
    return snapshot, summary, curves, conditions, contrasts, manifest


def matching(row, **condition):
    return all(row[k] == v for k,v in condition.items())


def configure(root, source):
    for variable, name in (('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')):
        directory=root/'runtime-cache'/'pool-plot'/name
        directory.mkdir(parents=True,exist_ok=True)
        os.environ[variable]=str(directory)
    os.environ['MPLBACKEND']='Agg'
    if source is not None: sys.path.insert(0,str(source.resolve()))
    import McUtils
    from McUtils.Plots import GraphicsGrid, Plot, FilledPlot, ErrorBarPlot, PlotLegend, Styled
    if source is not None and source.resolve() not in Path(McUtils.__file__).resolve().parents:
        raise RuntimeError('McUtils was imported from outside the requested source directory')
    return McUtils, GraphicsGrid, Plot, FilledPlot, ErrorBarPlot, PlotLegend, Styled


def render(root, snapshot, source=None):
    root=root.resolve()
    snapshot, summary, curves, endpoints, contrasts, manifest=load_snapshot(root,snapshot)
    if not curves:
        return dict(status='no_observed_training_curves', snapshot=str(snapshot.relative_to(root)), figures=[])
    McUtils, GraphicsGrid, Plot, FilledPlot, ErrorBarPlot, PlotLegend, Styled=configure(root,source)
    analysis=root/'analysis'
    with (analysis/'.plot.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        native_files={str(Path(sys.modules[obj.__module__].__file__).resolve()):
                      sha(sys.modules[obj.__module__].__file__) for obj in
                      (GraphicsGrid, Plot, FilledPlot, ErrorBarPlot, PlotLegend, Styled)}
        identity=dict(result_fingerprint=summary['result_fingerprint'], renderer_sha256=sha(__file__),
                      mcutils_module_sha256=sha(McUtils.__file__), mcutils_plot_source_sha256=native_files,
                      plotting_versions={name:metadata.version(name) for name in ('numpy','matplotlib')},style=STYLE)
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        destination=analysis/'figures'/('figures-'+fingerprint[:24])
        receipt_path=destination/'figure-manifest.json'
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt['identity'] != json.loads(json.dumps(identity)):
                raise RuntimeError('Plot output prefix collision')
            for name, expected in receipt['outputs_sha256'].items():
                if sha(destination/name) != expected: raise RuntimeError('Existing figure changed: '+name)
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=Path(tempfile.mkdtemp(prefix='.render-',dir=str(destination.parent)))
            outputs=[]
            try:
                theme=('mccoy',{'font':{'family':STYLE['font_family'],'size':STYLE['font_size']},
                                'svg':{'fonttype':'path'}})
                warning='\nINPUTS REQUIRE REVIEW' if summary['status']=='requires_review' else ''
                if summary.get('synthetic_fixture'): warning += '\nSYNTHETIC RENDER CHECK'
                def grid(title):
                    return GraphicsGrid(nrows=2,ncols=2,theme=theme,subimage_size=STYLE['subimage_size'],
                        spacings=STYLE['spacings'],padding=STYLE['padding'],
                        figure_label=Styled(title+'\nMean ± sample SD across available seeds; planned n = 3'+warning,fontsize=16))
                def legend(panel):
                    items=[{'marker':'line','label':scope.title(),'color':STYLE['colors'][scope],'lw':2.6}
                           for scope in ('complete','minimal','augmented')]
                    items += [{'marker':'line','label':'Training','color':'#333333','lw':2,'linestyle':'-'},
                              {'marker':'line','label':'Validation','color':'#333333','lw':2,'linestyle':'--'}]
                    panel.axes.legend(handles=PlotLegend.construct(items),loc='upper center',
                                      bbox_to_anchor=(1.10,1.27),ncol=5,fontsize=11,frameon=False,
                                      handlelength=2.4,columnspacing=1.3)
                def save(figure, stem):
                    figure.prep_show()
                    figure.figure.set_figure_spacings(STYLE['panel_gaps'])
                    for extension in STYLE['formats']:
                        name=stem+'.'+extension
                        figure.figure.savefig(str(temporary/name),format=extension,dpi=STYLE['dpi'],
                                              facecolor=figure.background)
                        outputs.append(name)
                    figure.close()
                for feature_mode in ('full','reduced52'):
                    for precision in ('decoded','psi8'):
                        subset=[r for r in curves if matching(r,feature_mode=feature_mode,precision=precision)]
                        if not subset: continue
                        chemistry='Full chemical descriptors' if feature_mode=='full' else 'Reduced chemical descriptors'
                        geometry='Decoded geometry' if precision=='decoded' else 'Decoded geometry + 8-bit torsions'
                        for metric in ('accuracy','bce'):
                            figure=grid(chemistry+' · '+geometry+' · '+('Accuracy' if metric=='accuracy' else 'BCE'))
                            scale=100 if metric=='accuracy' else 1
                            if metric=='accuracy':
                                minimum=min(r['mean']-(r['sample_sd'] or 0) for r in curves if r['metric'].endswith('_accuracy'))
                                limits=(40 if minimum>=.4 else 0,102)
                            else:
                                ymax=max(r['mean']+(r['sample_sd'] or 0) for r in subset if r['metric'].endswith('_bce'))
                                limits=(0,max(.75,1.06*ymax))
                            for i,readout in enumerate(('both','z_alpha')):
                                for j,lr in enumerate(('source','source_lr10')):
                                    panel=figure[i,j]
                                    panel.aspect_ratio='auto'
                                    readout_label='Graph + distance + angle + torsion' if readout=='both' else 'Torsion latent only'
                                    lr_label='LR = 5.69 × 10⁻⁴' if lr=='source' else 'LR = 5.69 × 10⁻⁵'
                                    notes=[]
                                    for scope in ('complete','minimal','augmented'):
                                        rows=[r for r in subset if matching(r,scope=scope,readout=readout,lr_profile=lr)]
                                        if not rows:
                                            notes.append(scope.title()+': no committed epochs')
                                            continue
                                        counts=sorted(set(r['n'] for r in rows))
                                        notes.append(scope.title()+': n = '+str(counts[0])+('–'+str(counts[-1]) if len(counts)>1 else ''))
                                        for split,pattern in (('train','-'),('validation','--')):
                                            series=sorted([r for r in rows if r['metric']==split+'_'+metric],key=lambda r:r['epoch'])
                                            if not series: continue
                                            x=[r['optimizer_updates']/1000 for r in series]
                                            y=[r['mean']*scale for r in series]
                                            # SD is undefined for a single seed; omit rather than draw a zero-width SD band.
                                            bands=[r for r in series if r['n']>=2]
                                            if bands:
                                                FilledPlot([r['optimizer_updates']/1000 for r in bands],
                                                    [(r['mean']+r['sample_sd'])*scale for r in bands],
                                                    [(r['mean']-r['sample_sd'])*scale for r in bands],
                                                    figure=panel,color=STYLE['colors'][scope],linewidth=0,
                                                    plot_style={'alpha':.11 if split=='train' else .07})
                                            Plot(x,y,figure=panel,color=STYLE['colors'][scope],linestyle=pattern,
                                                 linewidth=STYLE['linewidth'],plot_style={'alpha':.95 if split=='train' else .8})
                                    panel.plot_label=Styled(readout_label+'\n'+lr_label+'\n'+'; '.join(notes),fontsize=11)
                                    panel.axes_labels=['Optimizer updates (thousands)','Accuracy (%)' if metric=='accuracy' else 'Binary cross-entropy']
                                    panel.plot_range=[(0,350),limits]
                                    panel.ticks=[[0,100,200,300],([0,20,40,60,80,100] if metric=='accuracy' else None)]
                            legend(figure[0,0])
                            save(figure,'pooled_'+feature_mode+'_'+precision+'_'+metric)
                # Precision contrasts keep all other factors fixed, pair identical seeds,
                # and describe each endpoint separately; no unbalanced marginal pooling.
                for endpoint, title in (('final_validation_accuracy','Final validation'),('test_accuracy','Held-out test (best validation checkpoint)')):
                    available=[r for r in contrasts if r['factor']=='precision' and r['metric']==endpoint and r['n']]
                    if not available: continue
                    figure=grid(title+' · 8-bit minus decoded accuracy (percentage points)')
                    for i,readout in enumerate(('both','z_alpha')):
                        for j,feature_mode in enumerate(('full','reduced52')):
                            panel=figure[i,j]
                            panel.aspect_ratio='auto'
                            rows=[r for r in available if matching(r,readout=readout,feature_mode=feature_mode)]
                            for k,scope in enumerate(('minimal','augmented','complete')):
                                for lr,offset,marker in (('source',-.12,'o'),('source_lr10',.12,'s')):
                                    selected=[r for r in rows if matching(r,scope=scope,lr_profile=lr)]
                                    if not selected: continue
                                    row=selected[0]
                                    options=dict(figure=panel,color=STYLE['colors'][scope],marker=marker,
                                                 markersize=7,linestyle='None')
                                    if row['n']>=2:
                                        ErrorBarPlot([k+offset],[row['mean_percentage_points']],
                                            yerr=[row['sample_sd_percentage_points']],capsize=4,**options)
                                    else:
                                        Plot([k+offset],[row['mean_percentage_points']],**options)
                            span=max([abs(r['mean_percentage_points'])+(r['sample_sd_percentage_points'] or 0) for r in available]+[1.])
                            panel.plot_range=[(-.5,2.5),(-span*1.2,span*1.2)]
                            Plot([-.5,2.5],[0,0],figure=panel,color='#999999',linewidth=1)
                            panel.axes_labels=['Torsion selection','Accuracy difference (percentage points)']
                            panel.ticks=[([0,1,2],['Minimal','Augmented','Complete']),None]
                            counts=sorted(set(r['n'] for r in rows))
                            count_text=('n = '+str(counts[0])+('–'+str(counts[-1]) if len(counts)>1 else '')) if counts else 'No paired endpoints'
                            panel.plot_label=Styled(('Full descriptors' if feature_mode=='full' else 'Reduced descriptors')+' · '+
                                ('Full readout' if readout=='both' else 'Torsion only')+'\n'+count_text,fontsize=12)
                    items=[PlotLegend.construct_line_marker(lw=0,marker='o',label='Source LR',color='#444444'),
                           PlotLegend.construct_line_marker(lw=0,marker='s',label='Source LR / 10',color='#444444')]
                    figure[0,0].axes.legend(handles=PlotLegend.construct(items),loc='upper center',
                        bbox_to_anchor=(1.1,1.22),ncol=2,frameon=False,fontsize=12)
                    save(figure,'paired_precision_'+endpoint)
                receipt=dict(status='rendered',identity=identity,snapshot=str(snapshot.relative_to(root)),
                    snapshot_files_sha256=manifest,python=sys.version,machine=platform.machine(),
                    mcutils_path=str(Path(McUtils.__file__).resolve()),
                    outputs_sha256={name:sha(temporary/name) for name in outputs},
                    interpretation='Training and resampled validation are separate curves. Bands are sample SD across available training seeds, not conformer-level uncertainty. No run is excluded for poor accuracy; partial seed counts are shown. Precision contrasts retain fixed scope/readout/learning-rate/descriptors and pair seeds. Test uses only the prespecified validation-selected checkpoint.')
                write_json(temporary/'figure-manifest.json',receipt)
                os.replace(str(temporary),str(destination))
            finally:
                if temporary.exists(): shutil.rmtree(str(temporary))
        current=json.loads((analysis/'current.json').read_text())
        if current['result_fingerprint']==summary['result_fingerprint']:
            write_json(analysis/'latest-figures.json',dict(status='rendered',directory=str(destination.relative_to(root)),
                       result_fingerprint=summary['result_fingerprint'],figure_manifest_sha256=sha(receipt_path)))
        return dict(status='rendered',directory=str(destination.relative_to(root)),
                    figures=list(receipt['outputs_sha256']),input_status=summary['status'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--mcutils-source',type=Path)
    args=parser.parse_args()
    print(json.dumps(render(args.root,args.snapshot,args.mcutils_source),indent=2,sort_keys=True))


if __name__=='__main__': main()
