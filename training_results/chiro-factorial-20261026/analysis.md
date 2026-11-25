# ChIRo R/S factorial analysis

144 runs; 48 conditions; three seeds each; 100 epochs and full held-out testing.

Independent numerical analysis of all 144 ChIRo factorial runs
2026-10-26; snapshot snapshot-709d09ec5eb0812baed37c2b.

All 144 runs completed 100 epochs and full 69,719-conformer test evaluations. The 48 conditions each have three seeds. I independently reconstructed all 432 condition means and 3,888 seed-level metric contrasts from the run ledger; no discrepancy was found. No runs were excluded or selected by accuracy for inclusion. The comparisons below do not choose a model using test accuracy.

For condition plots, use mean plus/minus sample SD across the three training seeds and show each seed. For paired effects, use the same three seed differences with every other factor fixed. The descriptive design averages in the accompanying CSV first average matched conditions within seed, then compute mean and SD over the three seed averages. Do not compute their uncertainty across correlated conformers or across the 16/24 factorial conditions.

Learning rate has the strongest consistent endpoint effect. Reducing the original ChIRo rate 5.694249946e-4 by ten lowers test accuracy in every one of 24 matched conditions and all 72 seed pairs. Condition mean losses range from 1.302 to 2.268 percentage points. The balanced seed-average effect is -1.786 plus/minus 0.057 percentage points. Final training accuracy is nearly unchanged across the balanced design (99.174% versus 99.069%), whereas final validation accuracy is 98.078% versus 96.240%. This is worse validation/test generalization at the tested exposure, rather than inability to fit training. It is not an asymptotic convergence result: median selected epochs are 89 and 96; 58 of 72 lower-rate runs choose epoch 90 or later.

Additional torsion quantization has no uniform endpoint direction. The psi8-minus-decoded test effect is positive in 12 of 24 condition means and negative in 12; the range is -0.537 to +0.695 percentage points. The balanced effect is +0.019 plus/minus 0.028 percentage points. This supports a small design-average incremental effect in this experiment, rather than strict equivalence or a universal benefit of discretization.

There is no universal torsion-scope ranking. Augmented-minus-Minimal test means range from -0.585 to +0.490 percentage points, with ten positive and six negative; Complete-minus-Minimal ranges from -0.667 to +0.600, with eight of each sign. Descriptor reduction changes this comparison: Complete-minus-Minimal shifts downward after reducing chemistry in all eight matched remaining-factor condition means (by 0.104 to 0.865 percentage points). The balanced interaction is -0.297 plus/minus 0.229 percentage points. This is a conditional interaction, not proof that adding torsions inherently helps or harms.

Reduced chemistry has a modest negative design average (-0.127 plus/minus 0.073 percentage points), but conditional effects range from -0.683 to +0.391. Removing graph/distance/angle output branches leaves the graph-conditioned alpha representation: z_alpha-minus-both averages -0.027 plus/minus 0.052 percentage points, with a conditional range of -0.706 to +0.396. Do not describe z_alpha as geometry with no chemical context, or reduced52 as deletion of connectivity or a narrower neural network.

Accuracy and BCE should both remain visible. One Complete/psi8/source-LR/reduced52/both seed-3 run has final validation accuracy 98.025% but BCE 0.575; its selected highest-accuracy checkpoint has BCE 0.866, and its earlier minimum BCE was 0.056. This run is complete and must remain in the seed variance. More generally, lower-rate balanced validation BCE increases from 0.166 at epoch 50 to 0.217 at epoch 100 while training BCE falls. High accuracy alone does not establish good probability estimates. Checkpoint selection remains maximum validation accuracy, not minimum BCE.

All cells begin with decoded source-tag geometries. psi8 adds quantization of model torsions while distance/angle arrays stay fixed. There is no untouched-raw-geometry arm here, so these curves quantify the incremental torsion discretization effect, not the entire effect of storing conformers in the compact format. The original population, labels, and audited rare reconstruction losses are retained.

Three seeds provide limited descriptive uncertainty. Avoid significance or noninferiority claims from overlapping SD bars. Test conformers are correlated within molecules and shared across runs; they are not independent experimental replicates. Large checkpoint/prediction binaries were not freshly rehashed in this read-only analysis; native receipts and pool bindings were checked.



## All condition endpoints

| Scope | Geometry | LR | Chemistry | Readout | Selected validation (%) | Held-out test (%) |
|---|---|---|---|---|---:|---:|
| Minimal | Tag-decoded | Original | Full | Combined | 98.11 ± 0.20 | 98.31 ± 0.42 |
| Minimal | Tag-decoded | Original | Full | Torsion output | 98.34 ± 0.16 | 98.53 ± 0.10 |
| Minimal | Tag-decoded | Original | Reduced | Combined | 98.28 ± 0.06 | 98.42 ± 0.13 |
| Minimal | Tag-decoded | Original | Reduced | Torsion output | 98.36 ± 0.09 | 98.55 ± 0.18 |
| Minimal | Tag-decoded | Original / 10 | Full | Combined | 96.29 ± 0.37 | 96.53 ± 0.20 |
| Minimal | Tag-decoded | Original / 10 | Full | Torsion output | 96.65 ± 0.15 | 96.89 ± 0.20 |
| Minimal | Tag-decoded | Original / 10 | Reduced | Combined | 96.81 ± 0.05 | 96.92 ± 0.21 |
| Minimal | Tag-decoded | Original / 10 | Reduced | Torsion output | 96.21 ± 0.29 | 96.62 ± 0.28 |
| Minimal | + 8-bit torsions | Original | Full | Combined | 98.25 ± 0.20 | 98.44 ± 0.11 |
| Minimal | + 8-bit torsions | Original | Full | Torsion output | 98.36 ± 0.24 | 98.43 ± 0.29 |
| Minimal | + 8-bit torsions | Original | Reduced | Combined | 98.23 ± 0.37 | 98.45 ± 0.33 |
| Minimal | + 8-bit torsions | Original | Reduced | Torsion output | 98.53 ± 0.11 | 98.63 ± 0.06 |
| Minimal | + 8-bit torsions | Original / 10 | Full | Combined | 96.43 ± 0.29 | 96.47 ± 0.30 |
| Minimal | + 8-bit torsions | Original / 10 | Full | Torsion output | 96.58 ± 0.24 | 96.75 ± 0.15 |
| Minimal | + 8-bit torsions | Original / 10 | Reduced | Combined | 96.55 ± 0.19 | 96.72 ± 0.18 |
| Minimal | + 8-bit torsions | Original / 10 | Reduced | Torsion output | 96.38 ± 0.14 | 96.65 ± 0.23 |
| Augmented | Tag-decoded | Original | Full | Combined | 98.61 ± 0.07 | 98.77 ± 0.27 |
| Augmented | Tag-decoded | Original | Full | Torsion output | 98.47 ± 0.12 | 98.64 ± 0.10 |
| Augmented | Tag-decoded | Original | Reduced | Combined | 98.29 ± 0.26 | 98.54 ± 0.13 |
| Augmented | Tag-decoded | Original | Reduced | Torsion output | 98.05 ± 0.46 | 98.38 ± 0.37 |
| Augmented | Tag-decoded | Original / 10 | Full | Combined | 96.66 ± 0.32 | 97.02 ± 0.11 |
| Augmented | Tag-decoded | Original / 10 | Full | Torsion output | 96.43 ± 0.40 | 96.66 ± 0.41 |
| Augmented | Tag-decoded | Original / 10 | Reduced | Combined | 96.03 ± 0.27 | 96.34 ± 0.13 |
| Augmented | Tag-decoded | Original / 10 | Reduced | Torsion output | 96.34 ± 0.35 | 96.64 ± 0.17 |
| Augmented | + 8-bit torsions | Original | Full | Combined | 98.27 ± 0.29 | 98.56 ± 0.29 |
| Augmented | + 8-bit torsions | Original | Full | Torsion output | 98.35 ± 0.24 | 98.64 ± 0.10 |
| Augmented | + 8-bit torsions | Original | Reduced | Combined | 98.13 ± 0.42 | 98.39 ± 0.35 |
| Augmented | + 8-bit torsions | Original | Reduced | Torsion output | 98.22 ± 0.26 | 98.33 ± 0.22 |
| Augmented | + 8-bit torsions | Original / 10 | Full | Combined | 96.20 ± 0.61 | 96.48 ± 0.35 |
| Augmented | + 8-bit torsions | Original / 10 | Full | Torsion output | 96.52 ± 0.39 | 96.86 ± 0.38 |
| Augmented | + 8-bit torsions | Original / 10 | Reduced | Combined | 96.43 ± 0.42 | 96.75 ± 0.34 |
| Augmented | + 8-bit torsions | Original / 10 | Reduced | Torsion output | 96.04 ± 0.23 | 96.42 ± 0.28 |
| Complete | Tag-decoded | Original | Full | Combined | 98.39 ± 0.18 | 98.43 ± 0.23 |
| Complete | Tag-decoded | Original | Full | Torsion output | 98.13 ± 0.13 | 98.31 ± 0.14 |
| Complete | Tag-decoded | Original | Reduced | Combined | 98.28 ± 0.05 | 98.43 ± 0.04 |
| Complete | Tag-decoded | Original | Reduced | Torsion output | 98.05 ± 0.27 | 98.22 ± 0.25 |
| Complete | Tag-decoded | Original / 10 | Full | Combined | 97.01 ± 0.26 | 97.13 ± 0.11 |
| Complete | Tag-decoded | Original / 10 | Full | Torsion output | 96.37 ± 0.29 | 96.62 ± 0.34 |
| Complete | Tag-decoded | Original / 10 | Reduced | Combined | 96.21 ± 0.17 | 96.66 ± 0.17 |
| Complete | Tag-decoded | Original / 10 | Reduced | Torsion output | 95.59 ± 0.63 | 95.95 ± 0.69 |
| Complete | + 8-bit torsions | Original | Full | Combined | 98.29 ± 0.08 | 98.51 ± 0.11 |
| Complete | + 8-bit torsions | Original | Full | Torsion output | 98.25 ± 0.29 | 98.49 ± 0.20 |
| Complete | + 8-bit torsions | Original | Reduced | Combined | 98.04 ± 0.21 | 98.27 ± 0.03 |
| Complete | + 8-bit torsions | Original | Reduced | Torsion output | 98.38 ± 0.10 | 98.48 ± 0.12 |
| Complete | + 8-bit torsions | Original / 10 | Full | Combined | 96.61 ± 0.40 | 96.68 ± 0.25 |
| Complete | + 8-bit torsions | Original / 10 | Full | Torsion output | 96.68 ± 0.21 | 97.07 ± 0.29 |
| Complete | + 8-bit torsions | Original / 10 | Reduced | Combined | 96.48 ± 0.62 | 96.83 ± 0.65 |
| Complete | + 8-bit torsions | Original / 10 | Reduced | Torsion output | 96.33 ± 0.25 | 96.65 ± 0.13 |
