# Model Validation Note

Answers the question a judge is most likely to ask: "how did you split the
data, and is R² 0.88 defensible?"

## Split methodology

Standard 80/20 `train_test_split(test_size=0.2, random_state=42)` on
517,754 rows, plus 5-fold CV (std ±0.0003, in `model/metadata.json`).

**Reproduced independently on 11 Aug 2026:** holdout R² = 0.8835, MAE =
0.0443 — matches the recorded metadata (R² 0.8803, MAE 0.0449) within
rounding/library-version noise. The claimed number is real, not stale.

## Leakage check

- **Row-level:** 0 duplicate `id`s. 656 near-duplicate rows (identical
  feature values) out of 517,754 — 0.13%, not a meaningful contamination
  source at this scale.
- **Spatial/temporal:** the dataset has **no geographic identifier and no
  timestamp** — every row is an independent synthetic road-condition
  scenario (road_type, curvature, speed_limit, lighting, weather,
  time_of_day, holiday, school_season, num_reported_accidents). A random
  row-wise split therefore carries no spatial-adjacency or repeated-measures
  leakage risk, because there is no spatial or temporal dimension for rows
  to be adjacent *on*. This is a real property of the data, not an
  assumption.
- **Target leakage:** `num_reported_accidents` (r = 0.21 with target) and
  the engineered `accident_density = num_reported_accidents / (num_lanes+1)`
  are historical counts, not outcomes of the current prediction — legitimate
  predictive signal, not circular.

## Subgroup performance (proxy for fairness — see caveat)

No `region` field exists in this dataset — **regional fairness testing
described in some review checklists cannot be done on this data and we
should not claim it was.** The closest available subgroup axes are
`road_type`, `lighting`, and `weather`:

| Group (road_type) | n (holdout) | R² | MAE | mean actual risk |
|---|---:|---:|---:|---:|
| highway | 34,665 | 0.884 | 0.044 | 0.349 |
| rural | 34,682 | 0.885 | 0.044 | 0.349 |
| urban | 34,204 | 0.881 | 0.044 | 0.356 |

| Group (lighting) | n | R² |
|---|---:|---:|
| daylight | 35,601 | 0.857 |
| dim | 36,987 | 0.851 |
| night | 30,963 | 0.846 |

| Group (weather) | n | R² |
|---|---:|---:|
| clear | 35,943 | 0.877 |
| foggy | 36,323 | 0.884 |
| rainy | 31,285 | 0.875 |

**Reading:** `road_type` and `weather` subgroups are stable (R² 0.877–0.885).
`lighting` shows a small, consistent degradation in lower-light conditions
(0.857 → 0.846, an ~1.3-point R² drop from daylight to night) — worth
disclosing as a known limitation rather than hiding it: the model is
slightly less precise exactly in the conditions where risk is highest,
which is the honest, defensible way to say it.

## What this dataset is

Originally Kaggle Playground Series S5E10 — a synthetically generated
dataset that approximates real-world road-risk relationships, not a live
national crash database. State this plainly in the submission. It means:

- R² 0.88 measures how well the model reproduces the *generating
  relationships in this dataset*, not verified real-world crash outcomes.
- The impact estimate in `IMPACT.md` is therefore framed as a
  scenario-based illustration of the model's *targeting* value, never as a
  causal claim that real injuries/fatalities were or would be prevented.
