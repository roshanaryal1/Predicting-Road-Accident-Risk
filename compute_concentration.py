#!/usr/bin/env python3
"""
Compute the risk-concentration table that underpins the SDG 3.6 / 11.2
targeting claim, and the percentile reference the app uses to place a single
prediction against the training distribution.

Why this script exists
----------------------
The concentration figures were previously hardcoded in `streamlit_app.py` and
`ssd2026/IMPACT.md` with no committed script to regenerate them, which meant
nobody could verify them. On 2026-08-15 an attempt to reproduce them found the
published table matched the distribution of the **actual target values**, not
of **model predictions**, while every document described it as "ranking all
517,754 training rows by predicted risk".

Those are different claims. Ranking by the true target measures how
concentrated risk is in the data. Ranking by model predictions measures
whether *the model can find that concentration*, which is the only version
that supports using this tool for targeting. This script computes the latter.

Run:
    PYTHONPATH=$PWD python3 compute_concentration.py

Writes:
    model/risk_percentiles.npy   1001 quantiles of predicted risk
    ssd2026/CONCENTRATION.md     the table, with provenance
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import engineer_features  # noqa: E402

TIERS = [0.01, 0.05, 0.10, 0.25]


def main() -> None:
    model = joblib.load("model/accident_risk_model.pkl")
    encoders = joblib.load("model/label_encoders.pkl")
    order = joblib.load("model/feature_order.pkl")

    train = pd.read_csv("data/train.csv")
    X = engineer_features(train.drop(columns=["id", "accident_risk"]))
    for col, le in encoders.items():
        X[col] = le.transform(X[col].astype(str))

    pred = model.predict(X[order])
    n = len(pred)
    ranked = np.sort(pred)[::-1]

    rows = []
    for tier in TIERS:
        k = int(n * tier)
        top = ranked[:k]
        share = top.sum() / pred.sum()
        rows.append(
            dict(tier=tier, n=k, mean=float(top.mean()),
                 share=float(share), ratio=float(share / tier))
        )

    # Percentile reference for the app: where does one prediction sit?
    quantiles = np.quantile(pred, np.linspace(0, 1, 1001)).astype("float32")
    Path("model").mkdir(exist_ok=True)
    np.save("model/risk_percentiles.npy", quantiles)

    lines = [
        "# Risk concentration, computed from model predictions",
        "",
        f"Generated {date.today().isoformat()} by `compute_concentration.py`.",
        f"scikit-learn {sklearn.__version__}, {n:,} training rows, "
        f"model `accident_risk_model.pkl`.",
        "",
        "Ranking every training row by the model's **predicted** risk, then",
        "measuring how much total predicted risk falls in the highest-risk tiers.",
        "Concentration ratio = share of total predicted risk / share of segments.",
        "",
        "| Risk tier | Segments | Mean predicted risk | Share of risk mass | Concentration |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| Top {int(r['tier']*100)}% | {r['n']:,} | {r['mean']:.3f} | "
            f"{r['share']*100:.1f}% | **{r['ratio']:.2f}x** |"
        )
    lines += [
        f"| (all rows) | {n:,} | {pred.mean():.3f} | 100% | 1.00x |",
        "",
        "## What this does and does not say",
        "",
        "It says the model concentrates predicted risk: the highest-risk tenth of",
        "conditions carries roughly 1.8 times its proportional share. That is what",
        "makes targeted intervention worth investigating rather than spreading",
        "resources evenly.",
        "",
        "It does not say crashes were prevented, that the concentration is causal,",
        "or that intervening on these conditions will reduce harm by any particular",
        "amount. The training data is synthetic (Kaggle Playground Series S5E10).",
        "",
        "## Note on an earlier published version",
        "",
        "Figures of 2.50x / 2.22x / 1.93x / 1.64x appeared in earlier drafts and in",
        "the app. They could not be reproduced from model predictions. Their tier",
        "means (0.669 at the top decile, 0.573 at the top quartile) match the",
        "distribution of the **actual target values** (0.670, 0.574) rather than of",
        "predictions (0.643, 0.554), so they appear to have been computed by ranking",
        "on the target while being described as ranking on predictions. The table",
        "above replaces them and is reproducible by running this script.",
    ]
    out = Path("ssd2026/CONCENTRATION.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[9:16]))
    print(f"\nwrote {out} and model/risk_percentiles.npy")
    print(json.dumps({f"top{int(r['tier']*100)}": round(r["ratio"], 2) for r in rows}))


if __name__ == "__main__":
    main()
