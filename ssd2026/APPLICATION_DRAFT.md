# Application Form — Exact Field Answers

Confirmed 11 Aug 2026: the real form is a **Google Form**, "SDG Project
Submission Form - SSD 2026," with only 4 text fields plus one file upload.
No long-form fields — all substantive content (problem, solution, SDG,
evidence) lives in the uploaded document, not typed into the form.

## Form fields

**Author(s)' Name/s** *
```
Roshan Aryal
```

**Author(s)' Affiliation/s** *
```
Otago Polytechnic Auckland International Campus, Auckland, New Zealand
```
(Same affiliation used in the concurrent CAS research paper submission —
keep consistent across both SSD2026 submissions.)

**Title of the project** *
```
Road Accident Risk Predictor: Explainable Risk Scoring for Infrastructure
Safety Prioritization
```

**Email** *
```
CHECK BEFORE SUBMITTING — two addresses have shown up in this project:
manaratharyal@gmail.com (this session's context) and roshanaryaal@gmail.com
(logged into the Google Form). Use whichever you actually check for the
acceptance notification on 23 Oct 2026 — don't let this default silently.
```

**Upload your files here** * (max 10GB, zip if multiple files)
```
ssd2026/SSD2026_submission.zip
```

## What's in the zip

Built at `ssd2026/SSD2026_submission.zip` (11 Aug 2026), contains:
- `PROJECT_SUBMISSION.pdf` — the actual write-up: problem, solution,
  evidence, SDG alignment, innovation, technical validation, responsible
  AI, live demo link + cold-start warning. This is the document a judge
  will actually read.
- `screenshots/prediction_gauge.png`, `screenshots/model_info.png` — live
  app evidence
- `IMPACT.md`, `RESPONSIBLE_AI.md`, `MODEL_VALIDATION.md` — full backing
  detail behind the PDF's summaries

**Not yet in the zip:** the walkthrough video (not recorded yet — see
`VIDEO_SCRIPT.md`) and updated screenshots showing the SDG/Responsible AI
section once the code is pushed live (see `SUBMISSION_CHECKLIST.md`).
Rebuild the zip after adding those, before final submission.

## Rebuilding the zip after adding the video

```bash
cd ssd2026/submission_package
cp /path/to/walkthrough.mp4 .
cp screenshots_updated/*.png screenshots/   # after re-screenshotting About page
rm -f ../SSD2026_submission.zip
zip -r ../SSD2026_submission.zip . -x ".*"
```
