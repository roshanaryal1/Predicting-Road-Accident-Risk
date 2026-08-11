# Responsible AI & Deployment Boundaries

## What the model uses

Road and environmental characteristics only: road type, lane count,
curvature, speed limit, lighting, weather, signage presence, public-road
status, time of day, holiday/school-season flags, and historical reported
accident count (plus three engineered interaction features derived from
those).

## What it deliberately does not use

No driver identity, ethnicity, income, age, or any individual demographic
or behavioural data. The system evaluates **road and environmental risk**,
not personal risk assigned to a driver or vehicle. This is a design
choice, not a gap — stated explicitly so it reads as a strength, not an
omission.

## Known limitations (measured, not asserted)

- **No geographic field.** The dataset (Kaggle Playground Series S5E10)
  has no region/location column at all, so regional bias cannot be
  measured on this data — full stop. Don't claim geographic fairness
  testing was done; it wasn't possible.
- **Subgroup performance measured on the closest available axes** —
  `road_type` and `weather` are stable (R² 0.877–0.885 across groups);
  `lighting` shows a small consistent drop toward low-light conditions
  (R² 0.857 daylight → 0.846 night, see `MODEL_VALIDATION.md`). Disclosed
  rather than hidden: the model is marginally less precise in exactly the
  conditions most associated with elevated risk.
- **Synthetic training data.** Approximates real-world relationships (per
  Kaggle's playground-series methodology) but is not drawn from an actual
  crash register — predictions reflect learned patterns in this dataset,
  not verified real-world outcomes.
- **No uncertainty quantification yet.** The current UI shows a point risk
  score; it doesn't yet surface a confidence interval or prediction
  interval. Documented here as a roadmap item (see PROGRESS.md), not
  papered over.

## A note on estimand, for anyone checking this against related work

This author has a concurrent peer-reviewed analysis of the NZ national
crash record (Crash Analysis System, 2010–2024) under submission to the
same conference's Research track. That analysis finds precipitation
associated with **lower** odds of death-or-serious-injury given a crash
occurred (adjusted OR 0.835 for light rain), while this model's training
data associates rainy weather with **higher** predicted risk (mean
accident_risk 0.362 vs. 0.310 for clear). Darkness agrees in direction
across both (higher risk/severity at night in both).

This is not a contradiction to paper over — it's two different estimands,
and the difference is the point that other manuscript makes explicitly:
severity conditional on a crash occurring is not the same quantity as
crash occurrence risk, and open/synthetic road-condition data structurally
cannot separate them without external reporting-probability information
(Hauer's frequency-severity indeterminacy). This model's `accident_risk`
target is a synthetic Kaggle competition label of ambiguous estimand — it
was not constructed to isolate severity-given-crash the way the CAS
analysis was. Stated here so the two pieces of work are consistent in
what they claim, not because either is wrong on its own terms.

## Appropriate use

**Intended:** infrastructure planning and safety-intervention
prioritization — helping a roading authority decide where limited
inspection/engineering resources might have the most concentrated effect
(see `IMPACT.md` §1 for the risk-concentration evidence behind this).

**Not intended for:** individual driver liability or fault determination,
driver profiling, policing or enforcement targeting, insurance pricing, or
any punitive or individual-level decision. The model has no input that
identifies a person, and using its output as if it did would be a misuse
of a road-condition model, not a supported use case.

## Deployment maturity — how to describe it honestly

Call this a **"live deployed decision-support prototype"** or **"live
demonstrator with a production-oriented architecture,"** not "in
production." It's a public Streamlit deployment with no operator SLA, no
on-call, and no organization currently relying on it for real decisions —
saying "in production" invites exactly the question a judge should ask
("who operates it, who's accountable for predictions?") and the honest
answer is: nobody yet, it's a demonstrator. Overclaiming maturity here is a
bigger risk to the submission's credibility than underclaiming it.
