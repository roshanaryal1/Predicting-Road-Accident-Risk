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
- [x] Application form content drafted — `APPLICATION_DRAFT.md`
- [x] Video script written — `VIDEO_SCRIPT.md`

## Left — needs you (in order)

1. **Verify the deadline.** Download the actual official Application Form
   PDF from the SSD2026 site and confirm 1 Sep, not 1 Aug (see conflict
   note in `PROGRESS.md`). Do this first — it could change everything
   downstream.
2. **Review and commit the code changes.** `README.md` and
   `streamlit_app.py` have local edits not yet committed. Check the diff,
   then commit and push — Streamlit Cloud auto-deploys from git, so the
   in-app SDG/Responsible AI section won't be live until you do.
3. **Re-screenshot the About page** once the push is live, showing the
   new SDG/Responsible AI section — add it to `screenshots/`.
4. **Record the walkthrough video** using `VIDEO_SCRIPT.md`. Open the app
   a minute before recording so the cold-start isn't on camera.
5. **Fill the real Application Form** using `APPLICATION_DRAFT.md` as
   source copy, trimmed to whatever field limits the real form has.
6. **Convert to PDF**, assemble the supporting-docs zip (screenshots +
   video + this `ssd2026/` folder or relevant excerpts).
7. **Dry-run the live demo link** the morning you submit — wake it up
   first so it's warm if a judge clicks during a low-traffic window.
8. **Submit by 29–30 Aug**, not on the 1 Sep deadline itself.

## What I did not build (deliberately, per review — don't add these)

- No second ML model, no chasing R² higher
- No new frontend polish beyond what exists
- No causal crash-prevention claims anywhere in the docs
- No claim of "in production" — everywhere says prototype/demonstrator
- No regional fairness claim — the dataset has no region field, so none
  was fabricated
