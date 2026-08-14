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
Independent Researcher, Auckland, New Zealand
```
(This work was researched and developed independently. It was not a
campus, coursework or institutionally affiliated project: no supervision,
funding, resources or endorsement from any institution. The concurrent CAS
research is likewise independent and is being submitted to a journal, not to
SSD 2026.)

**Title of the project** *
```
Road Accident Risk Predictor: Explainable Risk Scoring for Infrastructure
Safety Prioritization
```

**Email** *
```
roshanaryaal@gmail.com
```
(Confirmed 11 Aug 2026 — this is the address acceptance notifications go
to on 23 Oct 2026.)

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
