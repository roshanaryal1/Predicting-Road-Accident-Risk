# Walkthrough Video Script (~75 seconds)

Target: 60-90s, per the application form's supporting-file allowance.
SDG/Responsible AI section is confirmed live on the app (verified
11 Aug 2026) — clear to record. Practice once before the real take; the
cold-start (~95s measured, though it's often warm) means **start
recording only after the app is already awake** — open the link a minute
before you hit record.

---

**0:00-0:10 — Problem**
(Screen: live app homepage, Predict Risk page)

> "Road safety agencies can't treat every road segment equally with
> limited resources. The question is where targeted intervention — such
> as signage, lighting, or speed review — may have the greatest
> potential benefit."

**0:10-0:25 — Input → prediction**
(Screen: adjust road type, curvature, weather, lighting, speed → result
updates live)

> "The model takes road and environmental conditions — including road
> type, curvature, weather, lighting, and speed — and produces a risk
> score in real time."

**0:25-0:40 — High-risk result**
(Screen: high-risk scenario: wet road, darkness, sharp curve, higher
speed. Rehearse these exact slider values beforehand — don't improvise
on camera.)

> "Here's a high-risk scenario: a sharp curve, wet conditions, poor
> lighting, and higher speed. The model flags the combination clearly —
> and we can see what's driving the prediction."

**0:40-0:52 — SHAP explanation**
(Screen: scroll to SHAP contribution chart)

> "The explanation is shown here. SHAP identifies which factors are
> pushing this individual prediction higher or lower, so it's not a
> black box."

**0:52-1:05 — Risk concentration / targeting value**
(Screen: Model Info page, risk-concentration metrics row)

> "Ranking all 517,754 training scenarios by predicted risk, the highest
> 10 percent carry nearly twice their proportional share of total
> predicted risk. That's predicted-risk concentration — it's what makes
> targeted intervention worth investigating, not proof that intervention
> reduces crashes."

**1:05-1:22 — New Zealand context**
(Screen: About page, scroll to "New Zealand context, and a companion study")

> "For the New Zealand picture I ran a separate national analysis of the
> Crash Analysis System: 153,000 reported injury crashes over fifteen
> years, one in five killing or seriously injuring someone. That study
> measures how severe a crash is once it happens. This tool asks a
> different question — where risk is elevated in the first place. I've
> kept them separate deliberately: the crash data doesn't validate this
> model, it shows why a tool like this is needed, because severity data
> alone can't tell you where to intervene."

**1:22-1:32 — SDG + responsible AI, closing**
(Screen: About page, SDG and Responsible AI section)

> "This supports UN SDG 3.6 and 11.2, and it's designed to support
> infrastructure decisions — not individual driver judgment. The goal
> isn't to predict who will crash. It's to help identify where safer
> infrastructure may matter most."

---

Visual sequence: **Problem → Inputs → Prediction → Explanation →
Targeting evidence → NZ context → SDG/Responsible AI.**

**Note on the NZ beat:** say "separate", "doesn't validate", and "different
question" out loud. If a judge later reads both projects and finds the CAS
analysis reporting rain as associated with *lower* severity while this tool
treats bad weather as risk-increasing, the video should already have told them
these measure different things. Claiming the crash data backs the model would
be false and is the one thing that could actively lose the entry.

## Shot list checklist

- [ ] App already awake before recording starts (avoid the cold-start
      screen on camera)
- [ ] One clean scenario walkthrough, no stumbling on slider inputs
      (rehearse the exact values beforehand — looks deliberate, not like
      testing your own app live)
- [ ] SHAP chart fully rendered and legible on screen, not mid-load
- [ ] About page shows the New Zealand context / companion study section
- [ ] About page shows the live SDG/Responsible AI section (confirmed
      live 11 Aug 2026)
- [ ] Audio: no dead air longer than ~2s, no filler words
- [ ] Export at 1080p, under the form's file-size limit if one is stated
