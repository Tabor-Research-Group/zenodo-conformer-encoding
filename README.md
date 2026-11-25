# Conformer-encoding reproduction archive

This repository collects the code, notebooks, fitted encoders, frozen input selections,
numerical results and figure materials used for the conformer-encoding paper.
The original project files have been copied, not moved or edited.

## Start here

1. Follow [REPRODUCING.md](REPRODUCING.md) to regenerate distributions, training
   curves and energy statistics from the saved results. The three notebooks in
   `notebooks/` are portable entry points. Original campaign notebooks remain in
   `workflows/` and `original_project/`.
2. Follow [DATASETS.md](DATASETS.md) to collect the remaining HPC data. Large files
   use a resumable SSH/rsync script, avoiding the file-transfer API.
3. Run `python scripts/check_archive.py inventory` after additional downloads.
   `provenance/files.tsv` records paths, sizes and SHA-256 checksums.

## Git and Zenodo

Large files are physically present in this working tree, while `.gitignore`
excludes data, arrays, JSON results, checkpoints and presentation files from Git.
The GitHub source ZIP and a Zenodo archive generated only from Git will therefore
omit these data. To package the full local collection, including ignored files:

```sh
python scripts/check_archive.py export --output ../conformer-encoding-zenodo.tar.gz
```

The export excludes Git history, environments and runtime caches. It does not
download missing files. Complete the transfers and check the transfer receipts
before publishing it. No commit, push or Zenodo upload has been performed.

## Contents

| Directory | Contents |
|---|---|
| `notebooks/`, `reproduction/` | Portable figure and statistics entry points |
| `workflows/` | Original fitting, encoding, sampling, energy, training and figure workflows |
| `software/` | Copied McUtils, Psience, ocen and original ChIRo source, with existing licenses |
| `data/original_project/` | Original SMILES, Coley inputs, arrays and checkpoints |
| `data/paper/` | Saved model grids and paper numerical data |
| `data/remote/` | Completed Entropy/Grace downloads |
| `training_results/` | Locally saved individual histories, pooled results and figures |
| `figures/` | Current author figure exports and the editable PowerPoint snapshot |
| `manuscript_snapshot/` | Manuscript and table context, not a second live editing target |
| `provenance/` | Source locations, software revisions, checksums and transfer plan |

Historical controls and exploratory difference encoding are retained for provenance.
They are distinguished from the final paper comparisons in DATASETS.md.
