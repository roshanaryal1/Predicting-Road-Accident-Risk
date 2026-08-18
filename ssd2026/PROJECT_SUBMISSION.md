# Road Accident Risk Predictor
### Explainable Risk Scoring for Infrastructure Safety Prioritization

**Author:** Roshan Aryal
**Affiliation:** Independent Researcher, Auckland, New Zealand
**Live demo:** https://roadaccident-roshanar-aryal.streamlit.app/
**GitHub:** https://github.com/roshanaryal1/SafeRoute

Submitted to SSD 2026 — Project Showcase, track "AI, Data & Smart Systems
for Sustainability"

---

## Summary

Machine learning identifies which road conditions carry disproportionate
accident risk, so limited infrastructure safety spending can be targeted
rather than spread evenly — with SHAP explanations for every prediction.

## Problem

Road safety agencies cannot treat every road segment equally with finite
inspection and engineering resources. Road trauma is concentrated in
particular conditions rather than spread evenly across a network. The
challenge is identifying which conditions carry disproportionate risk so
intervention — signage, lighting, speed-limit review — can be prioritized
where it has the most potential benefit.

The scale of that problem in New Zealand is documented. A separate national
analysis by the same author, of real New Zealand crash records, is currently
under peer review; exact figures are withheld here pending that review, but
it finds the speed environment strongly associated with severity — higher
posted speeds carry materially higher odds of serious injury than lower ones.

That analysis also establishes the limitation that motivates this tool. It
models severity **given that a crash has already occurred**. It cannot indicate
where crashes will occur, because the public extract excludes non-injury
crashes by design and carries no exposure denominator. Prioritising limited
engineering budget requires the other question — which conditions carry
elevated risk of a crash in the first place — and that is the question this
project addresses.

### Relationship to that companion research

These are two separate pieces of work on the same problem, deliberately kept
distinct. The CAS analysis uses real New Zealand crash records to characterise
**severity**. This tool is trained on an open synthetic competition dataset to
score condition-based **risk**. The CAS results do **not** validate this model
and are not used to do so. They establish two things only: that condition-based
reasoning about road harm is worth doing in the New Zealand context, and that
severity data alone cannot support prioritisation.

The distinction matters in a way worth stating plainly, because the two can
point in opposite directions on the same variable. The CAS analysis finds
precipitation associated with *lower* odds of serious injury, consistent with
behavioural compensation, while risk models of this kind generally treat
adverse weather as risk-increasing. Both can hold at once: rain may reduce the
severity of crashes that happen while increasing how many happen. That is
precisely why crash frequency and crash severity should not be inferred from
one another, and why a prioritisation tool cannot be built from severity data
alone.

## Solution

A Random Forest Regressor trained on 517,754 road-condition scenarios (15
features, including three engineered interaction terms) predicts a risk
score for a given road/environment combination and explains *why* via
SHAP feature attribution — not just a number, but which factors are
driving it (speed, curvature, lighting, weather). Live and publicly
deployed at the link above.

## Evidence of impact

Ranking all training rows by predicted risk: the **top 10% of road
conditions carry 1.82× their proportional share of total predicted risk**
(19.3% of total risk mass concentrated in 10% of segments). This is
measured directly from the model's own output, not assumed.

| Risk tier | % of segments | Share of total predicted risk | Concentration ratio |
|---|---:|---:|---:|
| Top 1% | 1% | 2.2% | 2.23× |
| Top 5% | 5% | 9.9% | 1.98× |
| Top 10% | 10% | 18.2% | 1.82× |
| Top 25% | 25% | 39.3% | 1.57× |

Under scenario-based intervention-effectiveness assumptions (15–30%,
sourced from FHWA's Crash Modification Factors Clearinghouse — a
repository of 3,000+ peer-reviewed countermeasure studies), this
represents a targeting mechanism with measurable potential benefit.
**This is stated as a scenario-based potential impact, not a causal
claim** — the model has not been shown to have prevented any crash. Full
methodology and worked calculation: `IMPACT.md` in the supporting files.

## SDG alignment

**Primary — SDG 3.6:** halve global deaths and injuries from road traffic
accidents. **Secondary — SDG 11.2:** safe, affordable, sustainable
transport systems.

Mechanism: risk scoring → targeted infrastructure spend → fewer serious
injuries. The model does not predict who will crash — it identifies which
road conditions carry disproportionate predicted risk.

## What is innovative

1. **Predictive** — moves from historical reporting toward forward-looking
   risk prioritization.
2. **Explainable** — SHAP translates every prediction into interpretable
   contributing factors, not a black box.
3. **Action-oriented** — connects risk scores to infrastructure
   intervention prioritization, not just a number on a screen.

## Technical evidence

Random Forest Regressor, 200 trees, 517,754 training rows, validation R²
0.88 — independently reproduced (0.8835), not just claimed. 5-fold CV std
±0.0003. No data leakage: no duplicate IDs, and the dataset has no
geographic or temporal columns for a random split to leak across.
Subgroup performance checked by road type and weather (stable, R²
0.877–0.885) and by lighting (small honest degradation toward night, R²
0.857→0.846 — disclosed, not hidden). Full detail: `MODEL_VALIDATION.md`.

## Responsible AI

The model uses road and environmental conditions only — no driver
identity, ethnicity, or demographic data. Intended for infrastructure
planning and safety prioritization, explicitly **not** for individual
driver liability, profiling, or enforcement. This is a live decision-
support prototype, not a production system with an operator or SLA.
Known limitations, including a disclosed subgroup-performance gap in
low-light conditions and the boundary of what this synthetic (Kaggle
Playground Series S5E10) dataset can support, are documented in full in
`RESPONSIBLE_AI.md`.

## Live demo

**https://roadaccident-roshanar-aryal.streamlit.app/**

Hosted on Streamlit Community Cloud's free tier, which sleeps after
inactivity — cold start measured at ~95 seconds. If the link shows
"Zzzz... this app has gone to sleep," click "Yes, get this app back up!"
and wait roughly a minute. Screenshots and a walkthrough recording are
included in the supporting files as a fallback.

## Contents of the supporting zip

- This document (`PROJECT_SUBMISSION.pdf`)
- `screenshots/` — live gauge prediction with SHAP contributions, model
  info and feature importance
- `walkthrough.mp4` — 60–90s recorded demo (if included)
- `IMPACT.md`, `RESPONSIBLE_AI.md`, `MODEL_VALIDATION.md` — full
  supporting detail behind the summaries above
