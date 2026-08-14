# Impact Analysis

**Framing (read this first):** this model does not claim to have prevented
any crashes. It claims to **concentrate limited safety-intervention
resources onto the road conditions the data identifies as highest-risk,
more efficiently than treating all segments equally.** Everything below is
scenario-based potential impact under stated assumptions, not a causal or
retrospective claim. See `MODEL_VALIDATION.md` for why this dataset
(Kaggle Playground Series S5E10, synthetic) can't support a stronger claim
than that.

## 1. Risk concentration — the primary evidence

Ranking all 517,754 training rows by predicted risk score:

| Risk tier | % of segments | Mean predicted risk | Share of total risk mass | Concentration ratio |
|---|---:|---:|---:|---:|
| Top 1% | 1% | 0.784 | 2.2% | **2.23×** |
| Top 5% | 5% | 0.698 | 9.9% | **1.98×** |
| Top 10% | 10% | 0.643 | 18.2% | **1.82×** |
| Top 25% | 25% | 0.554 | 39.3% | **1.57×** |
| (overall mean) | 100% | 0.352 | 100% | 1.00× |

*Reproduce with `python3 compute_concentration.py`. An earlier version of this
table (2.50/2.22/1.93/1.64) could not be reproduced from model predictions: its
tier means matched the distribution of the actual target values rather than of
predictions, while being described as ranking on predictions. See
`CONCENTRATION.md`.*

Concentration ratio = share of total risk mass ÷ share of segments. A ratio
of 1.82 for the top 10% means: **the highest-risk tenth of road conditions
in this dataset carries about 1.8× its proportional share of total
predicted risk.** That's the number that says "this model finds usable
concentration, not noise" — stronger evidence for judges than R² alone,
because it demonstrates decision usefulness, not just fit.

(Note: this dataset has one continuous `accident_risk` target, not a
separate severity/injury label — so this is risk-mass concentration, not
"% of severe crashes," and is described that way throughout. Don't let the
submission text drift into claiming the latter.)

## 2. Scenario-based potential benefit

Chain: **model → risk ranking → high-risk segment identification → targeted
intervention → assumed effectiveness → scenario range.** Each link is
labeled; nothing here is asserted as measured fact.

Published engineering countermeasures (signage, lighting upgrades, curve
treatment) have documented crash-reduction effects in the 15–30% range at
treated sites (FHWA Crash Modification Factors Clearinghouse — a repository
of 3,000+ peer-reviewed countermeasure studies). We use three effectiveness
scenarios against the risk mass held by the top-10% tier (19.3% of total):

| Scenario | Assumed effectiveness | Potential reduction in top-10%-tier risk mass | As share of total network risk mass |
|---|---:|---:|---:|
| Conservative | 15% | 2.9% | 0.6% |
| Moderate | 20% | 3.9% | 0.8% |
| Strong | 30% | 5.8% | 1.2% |

**Translating to New Zealand scale, illustratively:** NZ's provisional road
toll was 289 deaths in 2024 and 340 in 2023 (NZTA/Police, via 1News). If a
regional roading authority used this kind of targeting to prioritize a
comparable share of its serious-crash risk exposure, the moderate scenario
(0.8% of total network risk mass) applied to ~300 annual deaths is an
order-of-magnitude illustration of ~2–3 fewer fatalities/year — **stated as
a scenario range under the assumptions above, not a prediction or a
guarantee.** Waka Kotahi values the social cost of a single road fatality
at NZD $14.98M (2024 prices, land-value + medical + legal + vehicle-damage
components) — useful for framing scale, kept as a tertiary, clearly-labeled
figure rather than the headline number, since monetary valuation adds
assumptions (value-of-life methodology) on top of the crash-count chain
already built above.

## 3. What this section deliberately does not claim

- Does not claim the model has prevented, or will provably prevent, any
  specific crash.
- Does not claim the R² of 0.88 is itself the impact — it's the
  precondition for the ranking in §1 being trustworthy.
- Does not use "cost per fatality" as the primary metric — crash-risk
  concentration (§1) is primary because it's directly computable from this
  model; the dollar figure is tertiary framing only.

Sources: WHO Global Status Report on Road Safety 2023
(who.int/publications/i/item/9789240086517); FHWA CMF Clearinghouse
(cmfclearinghouse.fhwa.dot.gov); NZTA/Police provisional road toll figures
via 1News (1news.co.nz, Jan 2024 and Jan 2025 reports); Waka Kotahi social
cost of a fatal crash (NZTA benefits-management guidance, June 2024 prices).
