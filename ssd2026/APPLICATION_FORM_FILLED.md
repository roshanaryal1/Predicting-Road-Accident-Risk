# 4th International Conference on Smart Sustainable Development
## SDG Projects — Application Form

---

**1. Title of Project**

Road Accident Risk Predictor: Explainable Risk Scoring for Infrastructure Safety Prioritization

---

**2. Team members**

Roshan Aryal — Independent Researcher, Auckland, New Zealand

---

**3. Contact person's name**

Roshan Aryal

---

**4. Contact person's email address**

roshanaryaal@gmail.com

---

**5. Summary (400–500 words)**

Road safety agencies have finite resources for inspection, signage,
lighting and speed-limit review, but road trauma is not spread evenly
across a network. It concentrates in particular combinations of road and
environmental conditions. The Road Accident Risk Predictor addresses one
practical question: given limited resources, which road conditions should
be prioritized first?

The project is a machine learning system built on a Random Forest
Regressor trained on 517,754 road-condition scenarios, using 15 features
including three engineered interaction terms (speed times curvature,
accidents per lane, and a night-plus-bad-weather flag). The model
predicts a continuous risk score for a given combination of road type,
curvature, speed limit, lighting, weather, signage, and traffic context,
and explains every prediction using SHAP feature attribution, so the
output is not a black-box number but an interpretable breakdown of which
factors drive that specific score.

The system is live and publicly deployed as an interactive web
application, where a user can adjust road and environmental conditions
and see the predicted risk update in real time, alongside the SHAP
explanation and context-aware safety recommendations. Validation
performance is R-squared 0.88, independently reproduced rather than only
self-reported, with 5-fold cross-validation stability of plus-or-minus
0.0003. The model was checked for data leakage (no duplicate identifiers,
and the dataset has no geographic or temporal fields that a random
train/test split could leak across) and for subgroup performance
stability across road type, weather, and lighting conditions, with
results and limitations documented transparently rather than only
headline metrics reported.

The core evidence behind the project's value is a risk-concentration
result computed directly from the model's own output: ranking all
517,754 training scenarios by predicted risk, the highest-risk 10 percent
of conditions carry 1.82 times their proportional share of total
predicted risk. This demonstrates that the model finds genuine, usable
concentration rather than noise, which is the precondition for any
targeting strategy to be worth pursuing. Building on this, the project
sets out a scenario-based estimate of potential benefit under stated
intervention-effectiveness assumptions drawn from published
road-engineering literature, explicit about what it does and does not
claim: the model has not been shown to prevent any specific crash, and
the estimate is a targeting-value illustration, not a causal claim.

The project also documents its own boundaries directly: it uses road and
environmental data only, with no driver identity or demographic
information, and is intended for infrastructure planning and safety
prioritization, not individual driver liability or profiling. It is
described honestly as a live decision-support prototype rather than a
production system with an operator, and known limitations, including
reduced precision in low-light conditions and the synthetic nature of the
training data, are disclosed rather than omitted. The result is a system
whose primary contribution is not a single accuracy number but a
demonstrated, explainable, and honestly bounded way of turning road
condition data into prioritization decisions.

---

**6. Impact (100–150 words)**

The project's primary impact evidence is measured directly from the
model, not assumed: the highest-risk 10 percent of road conditions in the
training data carry 1.82 times their proportional share of total
predicted risk, meaning limited safety-intervention resources can be
concentrated on a small share of conditions for disproportionate expected
benefit, rather than spread evenly across a network. Under scenario-based
intervention-effectiveness assumptions (15 to 30 percent, sourced from
the FHWA Crash Modification Factors Clearinghouse), this targeting
approach represents a measurable potential benefit for road safety
agencies, insurers, and the public who share the road. The project is
explicit that this is a scenario-based potential impact and not a causal
claim of prevented harm, since demonstrating that would require a
deployed intervention study this project has not conducted.

---

**7. Alignment with the UN's Sustainable Development Goals — SDGs (200 words)**

The project targets SDG 3.6 as its primary goal: halve global deaths and
injuries from road traffic accidents. The mechanism is direct: the
model's risk-concentration result shows that predicted risk is not evenly
distributed across road conditions, which means targeted infrastructure
spending, rather than evenly spread investment, can plausibly achieve a
larger reduction in serious injuries for the same resource outlay. The
secondary goal is SDG 11.2, safe, affordable, and sustainable transport
systems, since the model's explicit purpose is supporting infrastructure
planning decisions such as signage, lighting, and speed-limit review,
which are the concrete interventions that make transport systems safer
for all road users.

Two supporting targets were deliberately not claimed. An earlier draft of
this project included SDG 9.c, but this was removed after review because
9.c concerns ICT and internet access in least-developed countries
specifically, which does not fit this project's mechanism. The project
favors precision over a broad SDG list, on the view that a small number
of directly justified goals is stronger evidence than a longer, weaker
list.

---

**8. Other Comments**

The live demo (https://roadaccident-roshanar-aryal.streamlit.app/) is
hosted on Streamlit Community Cloud's free tier, which sleeps after a
period of inactivity. Cold start was measured directly during preparation
of this submission at approximately 95 seconds from click to fully
rendered app. If the link shows "this app has gone to sleep," clicking
"Yes, get this app back up!" and waiting roughly a minute will resolve
it. Screenshots of the live application and a walkthrough recording are
included with this submission as a fallback in case the reviewer
encounters the app mid-wake.

**Companion research, kept deliberately separate.** The author has also
completed an independent national analysis of real New Zealand crash records,
currently under peer review, modelling injury **severity given a crash
occurred**. Dataset scale and DOI are withheld here pending that review. This
project scores condition-based **crash risk**, using an open synthetic
dataset — a different question, on different data.

They are not combined and that other work is **not** used to validate this
model. It is mentioned only because it demonstrates the specific limitation
this tool addresses: severity data cannot indicate where crashes will occur,
so it cannot support prioritisation on its own — and because, on the same
variable, the two lines of reasoning can point in different directions
(severity research sometimes finding adverse weather associated with lower
injury severity, while risk models of this kind generally treat adverse
weather as risk-increasing). Both can be true simultaneously, and that is
exactly why frequency and severity should not be inferred from one another.

Full supporting technical detail beyond what fits in the word limits
above — the complete model validation and leakage audit, the full
risk-concentration table, and the complete responsible-AI and limitations
documentation — is included in the supporting materials.

---

**9. A URL / web link for a shared folder of supporting documents**

GitHub repository: https://github.com/roshanaryal1/SafeRoute

(Full evidence packet — screenshots, walkthrough video, model validation,
impact analysis, responsible AI documentation — submitted as a zip file
alongside this form via both the SSD2026 Google Form portal and as an
email attachment to info@smartsust.org, per the conference's stated
submission instructions.)
