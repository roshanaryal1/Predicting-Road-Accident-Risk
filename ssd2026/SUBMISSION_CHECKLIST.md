# Final Submission Checklist

Everything that's automatable is done. What's left needs you specifically
— a signature, a download, a recording, a click — nothing here can be
delegated further. Read this top to bottom before submitting.

## Done (11 Aug 2026)

- [x] Plan reviewed against real SSD2026 site + corrected (SDG targets,
      timeline) — `submission-plan.html`, `PROGRESS.md`
- [x] Demo consolidation decided — Streamlit is canonical, confirmed live
- [x] Model validation / leakage audit — `MODEL_VALIDATION.md`
- [x] Impact analysis, risk-concentration-first framing — `IMPACT.md`
- [x] Responsible AI doc, real subgroup table — `RESPONSIBLE_AI.md`
- [x] Cross-checked against your concurrent CAS research paper (rain-
      direction caveat added to `RESPONSIBLE_AI.md`)
- [x] SDG section added to `README.md`
- [x] SDG + Responsible AI section added to `streamlit_app.py` About page
- [x] Live demo tested end-to-end, cold-start measured (~95s)
- [x] Screenshots captured: `screenshots/prediction_gauge.png`,
      `screenshots/model_info.png`
- [x] **Deadline confirmed: 1 Sep 2026** (user-confirmed 11 Aug 2026)
- [x] Real form structure discovered (Google Form, 4 text fields + 1 zip
      upload, no long-form fields) — `APPLICATION_DRAFT.md` rewritten with
      exact field answers
- [x] `PROJECT_SUBMISSION.pdf` written and generated — the actual document
      a judge reads, since the form itself has no room for content
- [x] `SSD2026_submission.zip` built — PDF + screenshots + IMPACT.md +
      RESPONSIBLE_AI.md + MODEL_VALIDATION.md

## Left — needs you (in order)

0. **Resolve the submission channel conflict.** The real official form
   (`/Users/roshanaryal/Downloads/SASD 2026 Project Application Form.docx`,
   found 11 Aug 2026) says to email the filled PDF to
   info@smartsust.org, subject `Smith_S_SSD2026_ProjectSubmission` — this
   is a different channel than the Google Form ("SDG Project Submission
   Form - SSD 2026") you showed earlier. Both could be legitimate (the
   docx may be the source-of-truth content form, the Google Form the
   actual upload portal — or the docx instructions may be outdated). I'm
   not guessing on this — confirm which channel(s) to actually use, maybe
   by checking the site's Call for Projects page again or emailing the
   organisers if unclear, before sending anything.
1. **Resolve the email field.** Two addresses have shown up in this
   project (see `APPLICATION_DRAFT.md`) — pick the one you actually check,
   don't let the Google account default silently.
1a. **Real form filled** — `APPLICATION_FORM_FILLED.pdf` matches the
   official docx exactly (Title, Team members, Contact, Summary 400-500w,
   Impact 100-150w, SDG 200w, Other Comments, shared-folder URL). Sent to
   you for review 11 Aug 2026 — read it before it goes anywhere.
2. [x] **Pushed** — `e995c01` pushed to origin/main 11 Aug 2026. Streamlit
   Cloud will auto-redeploy shortly; the in-app SDG/Responsible AI section
   isn't visible in the screenshots taken earlier this session (those
   predate the push) — re-screenshot below once the redeploy lands.
3. **Re-screenshot the About page** once the push is live, showing the
   new SDG/Responsible AI section, and add it into
   `submission_package/screenshots/` (rebuild the zip after).
4. **Record the walkthrough video** using `VIDEO_SCRIPT.md`. Open the app
   a minute before recording so the cold-start isn't on camera. Add it to
   `submission_package/` and rebuild the zip (command in
   `APPLICATION_DRAFT.md`).
5. **Fill the 4-field Google Form** using the exact answers in
   `APPLICATION_DRAFT.md`, upload `SSD2026_submission.zip`.
6. **Dry-run the live demo link** the morning you submit — wake it up
   first so it's warm if a judge clicks during a low-traffic window.
7. **Submit by 29–30 Aug**, not on the 1 Sep deadline itself.

## What I did not build (deliberately, per review — don't add these)

- No second ML model, no chasing R² higher
- No new frontend polish beyond what exists
- No causal crash-prevention claims anywhere in the docs
- No claim of "in production" — everywhere says prototype/demonstrator
- No regional fairness claim — the dataset has no region field, so none
  was fabricated
