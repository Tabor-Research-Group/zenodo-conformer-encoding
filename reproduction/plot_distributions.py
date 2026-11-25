"""Filled KDE comparisons sized for the user's portrait figure-layout deck."""
from pathlib import Path
import copy,hashlib,json,os,platform,sys,xml.etree.ElementTree as ET
ARCHIVE=Path(__file__).resolve().parents[1]
ROOT=ARCHIVE/'outputs/distributions';ROOT.mkdir(parents=True,exist_ok=True)
DRAFTS=ARCHIVE/'workflows'
os.environ['MPLBACKEND']='Agg'
for key in ('MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME'):os.environ[key]=str(ROOT/'cache'/key.lower())
sys.path.insert(0,str(DRAFTS/'geom-quantized-local-acceptance-20261123/candidate'))
import numpy as np
from McUtils.Plots import Graphics,GraphicsGrid,FilledPlot,Plot,ThemeManager
from McUtils.Plots.Backends import DPI_SCALING
COLORS={'Uniform':'#001c7f','Model':'#017517','Adaptive':'#cc457e'}
ALPHA={'Uniform':.16,'Model':.20,'Adaptive':.18}
FILL_ORDER=('Adaptive','Uniform','Model')
timing=json.loads((DRAFTS/'geom-interval-decoding-distributions-20261123/results/merged/plot_data.json').read_text())
energy=json.loads((DRAFTS/'geom-constrained-comparison-20261121/results/merged/plot_data.json').read_text())
tails=json.loads((DRAFTS/'geom-decode-tail-audit-20261123/survival.json').read_text())
out=ROOT/'figures';out.mkdir(exist_ok=True)
SVG='http://www.w3.org/2000/svg';ET.register_namespace('',SVG);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def theme(font,tick=None):
    return ('mccoy',{'font':{'family':'DejaVu Sans','size':font},'axes':{'labelsize':font},
        'xtick':{'labelsize':tick or font},'ytick':{'labelsize':tick or font},'svg':{'fonttype':'path'}})

def lookup(payload,stem):return copy.deepcopy(next(s for s in payload['plots'] if s['stem']==stem))

def curves(panel,spec,th,legend=False):
    selected='updated' if any(c['decode']=='updated' for c in spec['curves']) else 'constrained'
    for layer,name in enumerate(FILL_ORDER,1):
        for c in spec['curves']:
            if c['decode']!=selected or c['encoder']!=name:continue
            x,y=np.asarray(c['x']),np.asarray(c['y'])
            FilledPlot(x,y,figure=panel,theme=th,color=COLORS[c['encoder']],alpha=ALPHA[c['encoder']],
                       linewidth=0,zorder=layer,label='_nolegend_')
    for c in spec['curves']:
        Plot(c['x'],c['y'],figure=panel,theme=th,color=COLORS[c['encoder']],
            linewidth=2.0 if c['encoder']=='Adaptive' else 1.8,
            linestyle='-' if c['decode']==selected else '--',
            label=c['encoder'] if legend and c['decode']==selected else '_nolegend_',
            plot_style={'alpha':1.,'zorder':5})

def separate_legend(path):
    r=ET.parse(path).getroot();legends=[n for n in r.iter() if n.get('id','').startswith('legend_')]
    assert len(legends)==1,(path,len(legends))
    layer=ET.Element(r.tag,r.attrib);defs=ET.SubElement(layer,'{%s}defs'%SVG);seen=set()
    for group in r.iter('{%s}defs'%SVG):
        for node in group:
            ident=node.get('id')
            if ident is None or ident not in seen:
                defs.append(copy.deepcopy(node))
                if ident:seen.add(ident)
    clone=copy.deepcopy(legends[0]);parents={c:p for p in clone.iter() for c in p}
    for node in list(clone.iter('{%s}defs'%SVG)):parents[node].remove(node)
    layer.append(clone)
    parents={c:p for p in r.iter() for c in p};parents[legends[0]].remove(legends[0])
    files=[out/(path.stem+'_plot.svg'),out/(path.stem+'_legend.svg')]
    for root,file in zip((r,layer),files):
        ET.ElementTree(root).write(file,encoding='utf-8',xml_declaration=True)
        ids=[n.get('id') for n in root.iter() if n.get('id')];assert len(ids)==len(set(ids))
        for n in root.iter():
            h=n.get('{http://www.w3.org/1999/xlink}href','');assert not h.startswith('#') or h[1:] in ids
    return files,list(map(float,r.get('viewBox').split()))

records=[]
def finish(fig,stem,title,note,role,panels):
    fig.figure.obj.canvas.draw();canvas=fig.figure.obj.bbox;renderer=fig.figure.obj.canvas.get_renderer()
    for panel in panels:
        for obj in (panel.axes.obj.xaxis.label,panel.axes.obj.yaxis.label,panel.axes.obj.title):
            b=obj.get_window_extent(renderer)
            if b.width>0 and b.height>0:assert canvas.contains(b.x0,b.y0) and canvas.contains(b.x1,b.y1),(stem,obj.get_text(),b.extents)
        legend=panel.axes.obj.get_legend()
        if legend is not None:
            b=legend.get_window_extent(renderer);assert canvas.contains(b.x0,b.y0) and canvas.contains(b.x1,b.y1)
            assert [t.get_text() for t in legend.get_texts()] in (['Uniform','Model','Adaptive'],['Uniform','Model'])
    for ext in ('svg','png','pdf'):fig.figure.savefig(str(out/(stem+'.'+ext)),dpi=180,facecolor='white')
    (plot,legend),view=separate_legend(out/(stem+'.svg'))
    records.append(dict(stem=stem,title=title,note=note,role=role,plot_svg=str(plot),legend_svg=str(legend),
                        view_box=view,png=str(out/(stem+'.png'))))
    fig.close();print(json.dumps({'rendered':stem}),flush=True)

for full in (False,True):
    stem='decode_time_updated'+('_full_tails' if full else '')
    spec=lookup(timing,stem);th=theme(18,16)
    with ThemeManager.from_spec(th):
        f=Graphics(image_size=(900,285),padding=((70,30),(60,45)),theme=th)
        curves(f,spec,th,legend=True)
        f.axes_labels=['Time (seconds)','Counts'];f.frame=((True,False),(True,False));f.ticks=[None,[]]
        f.axes_scales=['log','linear'];peak=max(max(c['y']) for c in spec['curves'])
        f.plot_range=[spec['limits'],(0,1.06*peak)]
        f.plot_legend=True;f.legend_style={'loc':'lower right','bbox_to_anchor':(1.,1.02),'ncol':3,
            'frameon':False,'fontsize':14,'handlelength':1.5,'columnspacing':1.,'borderaxespad':0}
        f.prep_show()
        finish(f,stem,'Decode timing'+(' — full tails' if full else ''),
            'All 99,990 source-available records per encoder, including startup. Dashed unconstrained, solid constrained. '
            'Filled curves: constrained Adaptive behind Uniform and Model. Adaptive uses pink fill and an opaque pink KDE. Same stored model pointer in both timing controls.',
            'timing',[f])

th=theme(18,16)
with ThemeManager.from_spec(th):
    f=Graphics(image_size=(900,285),padding=((90,10),(60,45)),theme=th)
    for c in tails['curves']:
        y=np.array(c['y']);y=np.where(y>0,y,np.nan)
        Plot(c['x'],y,figure=f,theme=th,color=COLORS[c['encoder']],linewidth=1.8,
            linestyle='--' if c['condition']=='all' else '-',
            label=c['encoder'] if c['condition']=='all' else '_nolegend_')
    f.axes_labels=['Time (seconds)','Fraction above'];f.axes_scales=['log','log'];f.frame=((True,False),(True,False))
    f.plot_range=[(.008,5.5),(7e-6,1.2)];f.plot_legend=True
    f.legend_style={'loc':'lower right','bbox_to_anchor':(1.,1.02),'ncol':2,'fontsize':14,'frameon':False,'borderaxespad':0}
    f.prep_show()
    finish(f,'decode_startup_survival','Timing tails and initialization',
        'Dashed all calls, solid after independently identified startup events. Original n=99,990 each; '
        'initialized n=99,783 Uniform and 99,790 Model. No timing cutoff defines the diagnostic subset.', 'timing',[f])

for full in (False,True):
    stem='rmsd_unconstrained_vs_constrained'+('_full_tails' if full else '')
    spec=lookup(timing,stem);th=theme(24)
    # The saved KDE ordinates are counts per bin: density * n * bin width.
    width=(spec['limits'][1]-spec['limits'][0])/100
    for c in spec['curves']:
        c['y']=(np.asarray(c['y'])/(c['n']*width)).tolist()
    spec['ylabel']='Density'
    with ThemeManager.from_spec(th):
        f=Graphics(image_size=(650,316),padding=((40,40),(74,10)),theme=th)
        curves(f,spec,th,legend=True);f.axes_labels=[r'RMSD ($\mathrm{\AA}$)','Density']
        f.frame=((True,False),(True,False));step=.04 if full else .02
        f.ticks=[np.arange(0,spec['limits'][1]+1e-8,step).tolist(),[]]
        f.plot_range=[(0,spec['limits'][1]),(0,1.08*max(max(c['y']) for c in spec['curves']))]
        f.plot_legend=True;f.legend_style={'loc':'upper right','frameon':False,'fontsize':17,'labelspacing':.25,'handlelength':1.5}
        f.prep_show()
        finish(f,stem,'Reconstruction RMSD'+(' — full tails' if full else ''),
            'All-atom proper-rotation RMSD, 99,990 available conformers per curve, explicit hydrogens retained. '
            'Probability-density KDE ordinates (per angstrom), normalized over the full population rather than the displayed window. '
            'Dashed unconstrained, solid constrained. Adaptive chooses the lower-reference-RMSD packet. Ten unavailable inputs remain flagged.',
            'rmsd',[f])

potentials=[('NSE','NSE'),('AIMNet2_2025','B97-3c'),('AIMNet2_default','ωB97M-D3')]
for full in (False,True):
    for unit,tag in [('energy','total'),('energy_per_atom','per_atom')]:
        specs=[lookup(energy,p+'_'+unit+'_before_after'+('_full_tails' if full else '')) for p,title in potentials]
        limits=[min(s['limits'][0] for s in specs),max(s['limits'][1] for s in specs)]
        for s in specs:
            ratio=(limits[1]-limits[0])/(s['limits'][1]-s['limits'][0])
            for c in s['curves']:c['y']=(np.array(c['y'])*ratio).tolist()
        peak=max(max(c['y']) for s in specs for c in s['curves'])
        th=theme(18,16)
        with ThemeManager.from_spec(th):
            grid=GraphicsGrid(nrows=1,ncols=3,subimage_size=(300,245),padding=((65,15),(60,70)),spacings=(25,0),theme=th)
            for j,((potential,title),spec) in enumerate(zip(potentials,specs)):
                f=grid[0,j];curves(f,spec,th,legend=(j==2))
                f.axes_labels=[r'$\Delta E$ (kJ/mol)' if tag=='total' else r'$\Delta E/N$ (kJ/mol/atom)','Counts' if j==0 else '']
                f.frame=((True,False),(True,False));f.plot_range=[limits,(0,1.08*peak)]
                ticks=([0,100,200] if full else [0,4,8]) if tag=='total' else ([0,.5,1.,1.5] if full else [0,.05,.10,.15])
                f.ticks=[ticks,[]];f.plot_label=title
                if j==2:
                    f.plot_legend=True
                    f.legend_style={'loc':'upper right','bbox_to_anchor':(.98,.995),'bbox_transform':grid.figure.obj.transFigure,
                        'ncol':3,'fontsize':14,'frameon':False,'handlelength':1.5,'borderaxespad':0.,'columnspacing':1.}
            grid.prep_show()
            for j in range(3):
                f=grid[0,j];f.aspect_ratio='auto'
                f.axes_bbox=np.array([[.065+.325*j,.20],[.345+.325*j,.81]])*DPI_SCALING
            grid[0,2].axes.legend(**grid[0,2].legend_style)
            stem='energy_theory_'+tag+('_full_tails' if full else '_main')
            finish(grid,stem,'Energy changes'+(' per atom' if tag=='per_atom' else '')+(' — full tails' if full else ''),
                'Left to right: AIMNet2-NSE, AIMNet2-2025 B97-3c, default AIMNet2 ωB97M-D3. '
                'All six curves share 99,488 matched evaluated conformers, including stored hydrogens. '
                'Signed E(reconstructed)-E(reference), without geometry optimization. Dashed unconstrained, solid constrained. '
                'Common axes across the three potentials. KDE count normalization uses 100 common-width bins. '
                'Saved energy evaluations are reused; this is not new inference on newly accepted geometries.',
                'energy',[grid[0,j] for j in range(3)])

(ROOT/'manifest.json').write_text(json.dumps(dict(status='complete',figures=records,colors=COLORS,
    fill_alpha=ALPHA,fill_back_to_front=FILL_ORDER,legend_order=['Uniform','Model','Adaptive'],layout_template='/Users/Mark/Documents/Postdoc/Projects/Encoding/figs_Encoding.pptx',
    style='Filled KDE curves, compact encoder-only legends. No histogram artists. Shared legend per theory triptych.'),indent=2)+'\n')
