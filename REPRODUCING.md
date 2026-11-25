# Reproducing results

## Figures and tables from saved results

Create a Python 3.11 environment and install the figure dependencies and copied libraries:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python -m jupyter lab notebooks
```

CairoSVG also needs the Cairo system library. The notebooks load the copied
libraries directly, without installing source snapshots whose setup scripts
depend on Git version tags. Each portable notebook locates the
repository from its working directory and writes only to `outputs/`.

| Notebook | Reproduces |
|---|---|
| `01_reconstruction_distributions.ipynb` | RMSD, energy and timing KDEs, with separate SVG legend layers |
| `02_training_curves.ipynb` | Individual validation replicates for Minimal, Complete, Augmented and Z-matrix |
| `03_energy_statistics.ipynb` | Absolute-error MAE, P10, P90, maximum and tail counts, both total and per atom |

The portable plotting scripts change input/output locations and remove the
original machine-architecture guard. They retain the archived plotting calculations.
They regenerate the curves and legend layers, while the author's editable placement
is retained in `figures/editable/figs_Encoding.pptx`. Current narrow and `_wide`
exports in `figures/paper/` are preserved without regeneration or resizing.

## Original fitting and simulation workflows

The original notebooks and source scripts are preserved in dated `workflows/`
directories. Many contain their original absolute HPC paths. Run a copy after
setting its input/output paths to this archive or your compute allocation. Its
`pyproject.toml`, lockfile, source snapshot, manifests and submission receipts
describe the original environment and settings where available. The generic
figure environment above does not reproduce a CUDA training environment.

| Result | Original workflow |
|---|---|
| Coordinate distributions and Gaussian Base/likelihood Fit | `geom-distributions-20261007` |
| Periodic pre-shifted dihedral fit | `dihedral-shift-20261012` |
| Distinct eMolecules conformer population | `emols-distinct-025-20261019` |
| Both directional similarity scores | `emols-similarity-20261012` |
| Compact GEOM SMILESSupplier database | `geom-drugs-smidb-20261019` |
| Three-potential reference/reconstruction energies | `geom-reference-model-comparison-20261016`, `geom-reconstruction-models-20261019` |
| Adaptive encoder selection | `geom-adaptive-encoder-20261118` |
| Final quantization-interval constraint decoding | `geom-quantized-local-acceptance-20261123`, `geom-interval-decoding-distributions-20261123` |
| Main 144-run ChIRo-like factorial campaign | `chiro-full-factorial-20261024` |
| 48-run Z-matrix extension | `chiro-zmatrix-20261026` |
| Interactive molecule examples | `emols-near-linear-x3d-20261125`, `geom-energy-outliers-20261118` |

Software revisions and whether each original checkout contained local changes
are recorded in `provenance/software_revisions.tsv`. Historical campaign-specific
source snapshots remain beside the scripts that used them. Do not substitute
the current library snapshot for a historical patched version when reproducing
that experiment.

## Population and interpretation

The eMolecules deduplication keeps the same 10,000 molecules, each with up to 100
conformers separated by the saved 0.25 Angstrom RMSD criterion (747,974 retained
conformers). Similarity sampling keeps both directional scores and reproduces
the notebook's torsion condition. The similarity campaign predates that
deduplication and retains its own saved population; do not silently replace it.

The final GEOM RMSD/timing comparison contains 99,990 source-available records
from the frozen 100,000-conformer selection. The matched three-potential energy
comparison contains 99,488 records. Use each result's explicit masks and status
fields. Rejected, unsupported and unresolved records remain in the data and
must not be treated as successful evaluations.

Training curves show validation accuracy, not held-out test accuracy. Replicates
and collapsed runs are retained. Held-out test endpoints are supplied separately
in the campaign result collections. Solid/dashed curves and epoch-100 guides
retain the definitions in the copied plotting source.
