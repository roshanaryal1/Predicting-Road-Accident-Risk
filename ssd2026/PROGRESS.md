# SSD 2026 Submission — Progress Tracker

Conference: Smart Sustainable Development 2026, Auckland (26-27 Nov 2026)
Category: Project Showcase (cash prize), SDG-aligned
Track: "AI, Data & Smart Systems for Sustainability"
**Deadline: 01 Sep 2026** — confirmed by user 11 Aug 2026. Earlier
LinkedIn-post conflict (1 Aug) resolved in favor of the site date.
(internal target: submit 29-30 Aug)

Full plan: `ssd2026/submission-plan.html` — open in browser. Also mirrored as a
Claude artifact: https://claude.ai/code/artifact/22b2bf8a-c124-4212-b4a9-7ecf6574f347

## Status (as of 11 Aug 2026)

Plan reviewed against real SSD2026 site — verified accurate. SDG targets
corrected to 2 (3.6 primary, 11.2 secondary; dropped weak 9.c). Timeline
extended through acceptance/registration dates.

## Checklist — Week 1 (11-17 Aug)

- [x] A. Impact quantification — reframed per review: risk-concentration
      (top 10% of segments = 1.93x their proportional risk share) as
      primary evidence, scenario-based potential benefit as secondary,
      cost-per-fatality as tertiary framing only. Not a causal claim.
      See `ssd2026/IMPACT.md`.
- [x] B. Responsible AI & deployment boundaries (renamed from "ethics") —
      what the model uses/doesn't use, real subgroup performance table,
      appropriate-use boundary, honest "prototype not production" framing.
      See `ssd2026/RESPONSIBLE_AI.md`.
- [x] A2 (added, not in original plan). Model validation / leakage audit —
      reproduced R²=0.8835 independently (matches claimed 0.8803), no id
      leakage, no geographic/temporal columns to leak across a random
      split, subgroup R² stable except a small honest dip in low-light
      conditions. See `ssd2026/MODEL_VALIDATION.md`.
- [x] D. Demo consolidation — confirm Streamlit deployment as sole canonical
      demo link for judges; React/FastAPI stack framed as roadmap only

## Checklist — Week 2 (18-24 Aug)

- [ ] C. Add SDG section to README + in-app (3.6 + 11.2, mechanism sentence,
      §03A impact number)
- [x] E. Screenshot set: gauge prediction + SHAP contributions
      (`ssd2026/screenshots/prediction_gauge.png`), model info + feature
      importance (`ssd2026/screenshots/model_info.png`) — captured live
      11 Aug 2026 via automated browser. **Measured cold-start: ~95
      seconds** from click to fully rendered (site was asleep). This is
      real evidence for the demo-fallback risk, not a guess — use it to
      justify the recording fallback line in the application form.
- [ ] E. 60-90s screen-recording walkthrough (script drafted, see
      `ssd2026/VIDEO_SCRIPT.md` — recording itself is a manual step)
- [ ] Draft application form content (state demo URL + cold-start fallback
      line explicitly in the form text)

## Checklist — Week 3 (25-28 Aug)

- [ ] Fill official Application Form (download from SSD2026 site "Call for
      Projects" page)
- [ ] Convert to PDF
- [ ] Assemble supporting-docs zip
- [ ] Full dry-run of live demo link

## Later

- [ ] 29 Aug: fresh-eyes final review, fix broken links/typos
- [ ] Submit by 30 Aug (buffer before 1 Sep deadline)
- [ ] 23 Oct: acceptance notification (no action, just watch for it)
- [ ] 30 Oct: registration + camera-ready if accepted

## Open questions / decisions still needed

- Whether to build the uncertainty/confidence-interval UI improvement
  (`RESPONSIBLE_AI.md` flags this as not-yet-built) — backlog, not blocking
  submission; the doc discloses its absence honestly either way.

## Decision log

**§03D demo consolidation — DECIDED 11 Aug 2026:** Canonical judge-facing
demo is the Streamlit deployment: https://roadaccident-roshanar-aryal.streamlit.app/
Already the only deployed surface (README's stated "Live Demo"). React
(`frontend/`) + FastAPI (`backend/`) exist as source but are not deployed
anywhere — no live URL to confuse judges with. State them in the
application form as "in-progress production architecture" roadmap item,
not a second demo link. No code change needed — this was already the
de facto state, just confirming it explicitly for the submission packet.

**Demo cold-start — MEASURED 11 Aug 2026:** live-tested the Streamlit URL
cold (it had gone to sleep). Wake-up to fully rendered app took ~95
seconds. Confirms the risk register's "cold demo" entry is real, not
hypothetical. If a judge clicks cold during the judging window, plan for
~90+ seconds of "Zzzz" screen before anything shows — the fallback
recording/screenshot is not optional, it's load-bearing.

**In-app SDG/Responsible AI content — NOT YET LIVE:** added to
`streamlit_app.py`'s About page locally (11 Aug 2026) but the live
Streamlit Cloud deployment auto-builds from git — it won't show until
these changes are committed and pushed. The screenshots taken this
session are of the *current live* (pre-SDG-update) version. Re-screenshot
the About page after pushing, before finalizing the evidence packet.
