# Risk concentration, computed from model predictions

Generated 2026-08-15 by `compute_concentration.py`.
scikit-learn 1.9.0, 517,754 training rows, model `accident_risk_model.pkl`.

Ranking every training row by the model's **predicted** risk, then
measuring how much total predicted risk falls in the highest-risk tiers.
Concentration ratio = share of total predicted risk / share of segments.

| Risk tier | Segments | Mean predicted risk | Share of risk mass | Concentration |
|---|---:|---:|---:|---:|
| Top 1% | 5,177 | 0.784 | 2.2% | **2.23x** |
| Top 5% | 25,887 | 0.698 | 9.9% | **1.98x** |
| Top 10% | 51,775 | 0.643 | 18.2% | **1.82x** |
| Top 25% | 129,438 | 0.554 | 39.3% | **1.57x** |
| (all rows) | 517,754 | 0.352 | 100% | 1.00x |

## What this does and does not say

It says the model concentrates predicted risk: the highest-risk tenth of
conditions carries roughly 1.8 times its proportional share. That is what
makes targeted intervention worth investigating rather than spreading
resources evenly.

It does not say crashes were prevented, that the concentration is causal,
or that intervening on these conditions will reduce harm by any particular
amount. The training data is synthetic (Kaggle Playground Series S5E10).

## Note on an earlier published version

Figures of 2.50x / 2.22x / 1.93x / 1.64x appeared in earlier drafts and in
the app. They could not be reproduced from model predictions. Their tier
means (0.669 at the top decile, 0.573 at the top quartile) match the
distribution of the **actual target values** (0.670, 0.574) rather than of
predictions (0.643, 0.554), so they appear to have been computed by ranking
on the target while being described as ranking on predictions. The table
above replaces them and is reproducible by running this script.
