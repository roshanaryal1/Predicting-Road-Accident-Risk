# Model change evaluated and declined, 2026-08-15

Recorded so the decision is auditable rather than implicit.

## What was tested

Three configurations on the identical split (`test_size=0.2`,
`random_state=42`, 517,754 rows, same engineered features and encoding):

| Model | R² | MAE | RMSE | fit time |
|---|---:|---:|---:|---:|
| **Random Forest (current, shipped)** | 0.88028 | 0.04486 | 0.05750 | 16.2 s |
| Random Forest, max_depth=20 | 0.88330 | 0.04407 | 0.05676 | 25.4 s |
| HistGradientBoostingRegressor | **0.88454** | **0.04385** | **0.05646** | **4.3 s** |

HistGradientBoosting is genuinely better: +0.0043 R², 2.3% lower MAE, and
roughly 4x faster to fit.

## Why the model was not changed

The improvement is real and too small to matter for this purpose. The showcase
is judged on tangible outcomes, not on the fourth decimal place of R², and no
assessor distinguishes 0.880 from 0.885.

Against that, switching would invalidate work that is already verified:

- the risk-concentration figure in `IMPACT.md` (1.82x at the top decile),
  which is computed from model predictions and would change
- the subgroup performance table in `MODEL_VALIDATION.md`
- the independent reproduction check of 11 Aug 2026 (holdout R² 0.8835 against
  recorded 0.8803), which is the evidence that the reported metric is real
  rather than stale
- the committed screenshots showing current in-app numbers

Trading verified evidence for a 0.4% metric gain, days before a deadline, is a
poor trade. The faster fit time was noted and judged not to matter, since
deployment retraining has not caused an operational problem.

**Decision: keep the Random Forest exactly as shipped.** Revisit only if
deployment time becomes a real constraint, in which case the switch requires
retraining, recomputing the concentration table, updating both validation
documents and refreshing screenshots, not a drop-in swap.
