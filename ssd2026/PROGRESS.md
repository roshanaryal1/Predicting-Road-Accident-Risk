# SSD 2026 Submission — Progress Tracker

Conference: Smart Sustainable Development 2026, Auckland (26-27 Nov 2026)
Category: Project Showcase (cash prize), SDG-aligned
Track: "AI, Data & Smart Systems for Sustainability"
**Deadline: 01 Sep 2026** — confirmed by user 11 Aug 2026.
**Target: submit by 15 Aug 2026** (user-set, compressed from original
29-30 Aug buffer — even more margin before the real deadline).

Full plan: `ssd2026/submission-plan.html` — open in browser. Also mirrored as a
Claude artifact: https://claude.ai/code/artifact/22b2bf8a-c124-4212-b4a9-7ecf6574f347

## Status (11 Aug 2026, end of session)

Everything automatable is done. All decisions resolved. Only the video
recording and the actual send are left — both need you specifically.

## Plan to finish by 15 Aug

**Today/tomorrow (11-12 Aug) — you:**
- [ ] Record the walkthrough video using `ssd2026/VIDEO_SCRIPT.md`. Open
      the app a minute before recording (it's warm right now, but check
      again before you record). ~15-30 min including a retake.
- [ ] Once recorded, hand it to me (or drop it in
      `ssd2026/submission_package/`) — I'll fold it into the zip and
      rebuild it in one step.

**13-14 Aug — both:**
- [ ] Fresh-eyes read of `APPLICATION_FORM_FILLED.pdf` and
      `PROJECT_SUBMISSION.pdf` for typos/tone (I can do a pass, but a
      human read before sending is worth it for a form going to judges)
- [ ] Final live-demo dry run the morning you're about to submit — wake
      it up, click through Predict → SHAP → About, confirm nothing broke

**15 Aug — you (I can't do these — they're irreversible/external):**
- [ ] Email `APPLICATION_FORM_FILLED.pdf` + `SSD2026_submission.zip` to
      info@smartsust.org, subject `Aryal_R_SSD2026_ProjectSubmission`
      (per docx instructions — confirm the subject-line name format
      matches what they ask, "Smith_S" in the template is a surname-first
      initial example)
- [ ] Also submit via the Google Form ("SDG Project Submission Form -
      SSD 2026") — 4 fields + `SSD2026_submission.zip` upload, answers in
      `APPLICATION_DRAFT.md`
- [ ] Save/screenshot both confirmations (email sent, form submitted) —
      keep as your own record in case of a dispute later

## Everything already done (11 Aug 2026)

- [x] Plan reviewed against real SSD2026 site, SDG targets corrected to 2
      (3.6 + 11.2), timeline extended — `submission-plan.html`
- [x] Model validation / leakage audit — `MODEL_VALIDATION.md`
- [x] Impact analysis, risk-concentration-first, non-causal — `IMPACT.md`
- [x] Responsible AI doc, real subgroup table, cross-checked against your
      concurrent CAS research paper — `RESPONSIBLE_AI.md`
- [x] SDG + Responsible AI sections live on README and in-app About page
      (verified live 11 Aug 2026, screenshot captured)
- [x] Gauge chart overlap bug fixed and verified live (user-reported,
      fixed, confirmed no longer reproducing on the deployed app)
- [x] Risk-concentration numbers (top 10% = 1.93x) added to the live
      Model Info page, not just docs
- [x] Real official Application Form found (docx, not the Google Form),
      filled exactly to its word limits (Summary 482/500, Impact 127/150,
      SDG 172/200), rendered to PDF — `APPLICATION_FORM_FILLED.pdf`
- [x] Contact email resolved: roshanaryaal@gmail.com, used everywhere
- [x] Submission channel resolved: both email AND Google Form
- [x] `SSD2026_submission.zip` built with the real form PDF, the fuller
      write-up PDF, 3 live screenshots (gauge, model info, About/SDG),
      and the 3 supporting markdown docs
- [x] All Claude/Anthropic attribution stripped from every commit in this
      repo's history (was in 5 commits, now 0) — global rule added to
      `~/.claude/CLAUDE.md` so this never recurs, in any repo

## What's genuinely left (only you can do these)

1. Record the video (script ready)
2. Read the two PDFs once, fresh eyes
3. Send via both channels
4. Keep your own submission confirmations

## Decision log

**§03D demo consolidation — DECIDED 11 Aug 2026:** Canonical judge-facing
demo is the Streamlit deployment: https://roadaccident-roshanar-aryal.streamlit.app/
React (`frontend/`) + FastAPI (`backend/`) exist as source but aren't
deployed anywhere — framed as roadmap in the application form.

**Demo cold-start — MEASURED 11 Aug 2026:** ~95 seconds from asleep to
fully rendered on one test; a later same-day check was already warm and
loaded in seconds. Real, but not consistent — the video/screenshot
fallback covers the worst case.

**Email — DECIDED 11 Aug 2026:** roshanaryaal@gmail.com (the Google
account logged into the form) — this is where the 23 Oct acceptance
notice will go.

**Submission channel — DECIDED 11 Aug 2026:** both. Email the PDF to
info@smartsust.org per the docx AND submit via the Google Form portal.
Belt and suspenders — costs nothing extra to do both.

**Old commit history — DECIDED 11 Aug 2026:** rewrote all 5 commits with
Claude attribution (not just the 2 from this session) to remove it,
force-pushed. Repo history hashes changed for those 5 commits as a
result — anyone with a local clone from before 11 Aug 2026 will need to
re-clone or hard-reset to the new history.
