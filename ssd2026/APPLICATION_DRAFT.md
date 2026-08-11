# Application Form — Draft Content

Copy-paste blocks for the official SSD2026 "Call for Projects" Application
Form once downloaded. Written to be trimmed to fit whatever field/character
limits the real form imposes — each block is ordered longest-to-essential
so you can cut from the bottom.

**Before using this:** download the actual Application Form from the SSD2026
site's Call for Projects page and check its field structure — this draft
guesses at typical categories (problem, solution, SDG, evidence, demo)
since the real form isn't in hand yet. Reconcile before submitting, and
re-verify the 1 Sep deadline against it (see `PROGRESS.md` deadline flag).

---

## Project title

Road Accident Risk Predictor — Explainable Risk Scoring for Infrastructure
Safety Prioritization

## One-line summary (for a summary/tagline field)

Machine learning identifies which road conditions carry disproportionate
accident risk, so limited infrastructure safety spending can be targeted
rather than spread evenly — with SHAP explanations for every prediction.

## Problem statement

Road safety agencies cannot treat every road segment equally with finite
inspection and engineering resources. Road trauma remains concentrated in
particular conditions rather than spread evenly across a network, and the
challenge is identifying which conditions carry disproportionate risk so
intervention (signage, lighting, speed-limit review) can be prioritized
where it has the most potential benefit.

## Solution

A Random Forest model trained on 517,754 road-condition scenarios (15
features, engineered interaction terms) predicts a risk score for a given
road/environment combination and explains *why* via SHAP feature
attribution — not just a number, but which factors are driving it (speed,
curvature, lighting, weather). Live, publicly deployed:
**https://roadaccident-roshanar-aryal.streamlit.app/**

## Evidence of impact / outcome

Ranking all training rows by predicted risk, the **top 10% of road
conditions carry 1.93× their proportional share of total predicted risk**
(19.3% of total risk mass concentrated in 10% of segments) — demonstrated
concentration, not an assumed one. Under scenario-based intervention
effectiveness assumptions (15–30%, sourced from FHWA's Crash Modification
Factors Clearinghouse), this represents a targeting mechanism with
measurable potential benefit. Full methodology, including what this claim
does *not* assert, is documented at `ssd2026/IMPACT.md` in the project
repository — we lead with what's measured, not a causal claim.

## SDG alignment

**Primary: SDG 3.6** — halve global deaths and injuries from road traffic
accidents. **Secondary: SDG 11.2** — safe, affordable, sustainable
transport systems. Mechanism: risk scoring → targeted infrastructure
spend → fewer serious injuries.

## Innovation

Three things distinguish this from a standard prediction demo: (1)
**predictive** — moves from historical reporting toward forward-looking
risk prioritization; (2) **explainable** — SHAP translates every
prediction into interpretable contributing factors, not a black box; (3)
**action-oriented** — connects risk scores to infrastructure intervention
prioritization rather than stopping at a number.

## Responsible AI / ethics

The model uses road and environmental conditions only — no driver
identity, ethnicity, or demographic data. Intended for infrastructure
planning and safety prioritization, explicitly **not** for individual
driver liability, profiling, or enforcement. Deployment is honestly framed
as a live decision-support prototype, not a production system with an
operator or SLA. Full limitations documented at
`ssd2026/RESPONSIBLE_AI.md`, including a disclosed subgroup-performance
gap in low-light conditions and the honest boundary of what this synthetic
Kaggle-derived dataset can and cannot support.

## Live demo

**https://roadaccident-roshanar-aryal.streamlit.app/**

Note: hosted on Streamlit Community Cloud's free tier, which sleeps after
inactivity — cold start measured at ~95 seconds. **If the link shows
"Zzzz... this app has gone to sleep," click "Yes, get this app back up!"
and wait roughly a minute.** A screenshot/recording of the working app is
included in the supporting materials as a fallback.

## Technical evidence (for a "methodology" or "technical approach" field)

Random Forest Regressor, 200 trees, 517,754 training rows, validation R²
0.88 (independently reproduced, not just claimed — see
`ssd2026/MODEL_VALIDATION.md`), 5-fold CV std ±0.0003. No data leakage: no
duplicate IDs, no geographic/temporal columns for a random split to leak
across. Subgroup performance checked by road type and weather (stable,
R² 0.877–0.885) and by lighting (small honest degradation toward night,
R² 0.857→0.846, disclosed rather than hidden).

## Links

- Live demo: https://roadaccident-roshanar-aryal.streamlit.app/
- GitHub: https://github.com/roshanaryal1/Predicting-Road-Accident-Risk
- Full evidence packet: `ssd2026/` folder in the repository (IMPACT.md,
  RESPONSIBLE_AI.md, MODEL_VALIDATION.md, screenshots, this draft)
