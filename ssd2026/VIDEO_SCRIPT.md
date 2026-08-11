# Walkthrough Video Script (~75 seconds)

Target: 60–90s, per the application form's supporting-file allowance.
Record after the SDG/Responsible AI changes are pushed and live (see
PROGRESS.md — they aren't live yet as of 11 Aug 2026). Practice once
before the real take; the cold-start (~95s measured) means **start
recording only after the app is already awake** — open the link a minute
before you hit record, or splice the wake wait out in editing.

---

**0:00–0:10 — Problem**
(Screen: live app homepage, Predict Risk page)

> "Road safety agencies can't treat every road segment equally with
> limited resources. The question is where targeted intervention —
> signage, lighting, speed review — has the greatest potential benefit."

**0:10–0:25 — Input → prediction**
(Screen: adjust road type, curvature, weather sliders; click through to
result)

> "The model takes road and environmental conditions — road type,
> curvature, weather, lighting — and predicts a risk score in real time."

**0:25–0:40 — High-risk result**
(Screen: gauge showing a high-risk scenario, e.g. rainy + night + sharp
curve + high speed)

> "Here's a high-risk scenario: sharp curve, wet road, poor lighting, high
> speed. The model flags it clearly — this isn't a black box."

**0:40–0:52 — SHAP explanation**
(Screen: scroll to SHAP contribution chart)

> "And it explains why — this chart shows exactly which factors are
> pushing the score up or down for this specific prediction."

**0:52–1:05 — Risk concentration / impact**
(Screen: Model Info page, or a slide with the concentration table if not
built into the UI yet)

> "Ranking all 517,000 training scenarios by predicted risk, the highest
> 10 percent carry nearly twice their proportional share of total risk —
> that's the targeting value: concentrated, not spread evenly."

**1:05–1:15 — SDG + responsible AI, closing line**
(Screen: About page SDG/Responsible AI section)

> "This targets UN SDG 3.6 and 11.2 — and it's built to support
> infrastructure decisions, not individual driver judgment. The goal isn't
> to predict who will crash. It's to help identify where safer
> infrastructure may matter most."

---

## Shot list checklist

- [ ] App already awake before recording starts (avoid the ~95s cold-start
      screen on camera)
- [ ] One clean scenario walkthrough, no stumbling on slider inputs
      (rehearse the exact values beforehand)
- [ ] SHAP chart fully rendered and legible on screen, not mid-load
- [ ] About page shows the live SDG/Responsible AI section (only true
      after the push — see PROGRESS.md)
- [ ] Audio: no dead air longer than ~2s, no filler words
- [ ] Export at 1080p, under the form's file-size limit if one is stated
