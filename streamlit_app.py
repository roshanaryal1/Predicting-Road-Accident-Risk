"""
Road Accident Risk Predictor — Streamlit Web App
All-in-one file. Logic lives in src/; this file handles only UI routing.

Pages:
  1. Predict Risk     — live prediction, SHAP, confidence interval, recommendations
  2. Compare          — side-by-side scenario comparison
  3. Batch Predict    — CSV upload + download
  4. History          — session prediction log
  5. Model Info       — honest metrics, feature importance
  6. About            — project details
"""

import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Road Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: glassmorphism + Streamlit native theme ───────────────────────────────
st.markdown("""
<style>
/* Glassmorphism cards */
.glass {
    background: rgba(38, 43, 56, 0.65);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* Result card */
.result-card {
    background: rgba(38, 43, 56, 0.85);
    border-radius: 14px;
    border-left: 5px solid;
    padding: 28px 24px;
    text-align: center;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(22, 27, 38, 0.85);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Gradient button */
.stButton > button {
    background: linear-gradient(90deg, #6A82FB, #FC5C7D);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    width: 100%;
    font-size: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    background: linear-gradient(90deg, #FC5C7D, #6A82FB);
}

/* Metric delta hide arrow on risk display */
[data-testid="stMetricDelta"] svg { display: none; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Imports that depend on src/ ───────────────────────────────────────────────
from src.model_utils    import load_model, predict_with_confidence, compute_shap, SHAP_AVAILABLE
from src.preprocessing  import (
    encode_input, encode_batch, validate_input,
    WEATHER_OPTIONS, LIGHTING_OPTIONS, ROAD_TYPE_OPTIONS, TIME_OPTIONS,
    DEMO_SCENARIOS, FEATURE_DISPLAY_NAMES,
)
from src.recommendations import generate_recommendations
from src.visualization   import (
    create_gauge, create_shap_chart, create_importance_chart,
    create_comparison_gauges, create_history_chart, risk_label,
)


# ── Session state initialisation ──────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

if 'demo_idx' not in st.session_state:
    st.session_state.demo_idx = 0


# ── Model loading (with auto-train fallback) ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        with st.spinner("🤖 First run — training model (this takes ~3 min)…"):
            result = subprocess.run(
                [sys.executable, 'train_and_save_model.py'],
                capture_output=True, text=True, timeout=600,
            )
        if result.returncode != 0:
            st.error(f"Model training failed:\n{result.stderr}")
            st.stop()
        st.rerun()


model, encoders, feature_order, metadata = get_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_prediction(raw: dict) -> tuple[float, float, float, np.ndarray | None, float | None]:
    """
    Encode input, predict, compute CI and SHAP.
    Returns (prediction, ci_low, ci_high, shap_values, base_value).
    """
    X = encode_input(raw, encoders, feature_order)
    pred, ci_low, ci_high = predict_with_confidence(model, X)
    pred    = float(np.clip(pred, 0.0, 1.0))
    ci_low  = float(np.clip(ci_low, 0.0, 1.0))
    ci_high = float(np.clip(ci_high, 0.0, 1.0))
    shap_vals, base_val = compute_shap(model, X, feature_order)
    return pred, ci_low, ci_high, shap_vals, base_val


def input_widgets(prefix: str, defaults: dict | None = None) -> dict:
    """
    Render all 12 input widgets and return raw values dict.
    `prefix` makes widget keys unique (needed for comparison page).
    `defaults` overrides default widget values (for demo scenarios).
    """
    d = defaults or {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("##### 🛣️ Road Characteristics")
        road_type = st.selectbox(
            "Road Type", ROAD_TYPE_OPTIONS,
            index=ROAD_TYPE_OPTIONS.index(d.get('road_type', 'urban')),
            key=f"{prefix}_road_type",
            help="Type of road: highway, rural, or urban street.",
        )
        num_lanes = st.slider(
            "Number of Lanes", 1, 4,
            value=d.get('num_lanes', 2),
            key=f"{prefix}_num_lanes",
            help="Total lanes on this road (training data range: 1–4).",
        )
        curvature = st.slider(
            "Road Curvature", 0.0, 1.0,
            value=float(d.get('curvature', 0.1)),
            step=0.05,
            key=f"{prefix}_curvature",
            help="0.0 = perfectly straight road  |  1.0 = hairpin turn.",
        )
        road_signs = st.checkbox(
            "Road Signs Present",
            value=bool(d.get('road_signs_present', True)),
            key=f"{prefix}_road_signs",
            help="Warning or informational road signs visible.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("##### 🌤️ Environmental")
        weather = st.selectbox(
            "Weather", WEATHER_OPTIONS,
            index=WEATHER_OPTIONS.index(d.get('weather', 'clear')),
            key=f"{prefix}_weather",
            help="Current weather: clear, rainy, or foggy.",
        )
        lighting = st.selectbox(
            "Lighting", LIGHTING_OPTIONS,
            index=LIGHTING_OPTIONS.index(d.get('lighting', 'daylight')),
            key=f"{prefix}_lighting",
            help="Ambient lighting: daylight, dim (dusk/dawn), or night.",
        )
        time_of_day = st.selectbox(
            "Time of Day", TIME_OPTIONS,
            index=TIME_OPTIONS.index(d.get('time_of_day', 'afternoon')),
            key=f"{prefix}_time_of_day",
            help="General time period of travel.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("##### 🚦 Traffic & Context")
        speed_limit = st.slider(
            "Speed Limit (km/h)", 25, 70,
            value=d.get('speed_limit', 50), step=5,
            key=f"{prefix}_speed",
            help="Posted speed limit (training range: 25–70 km/h).",
        )
        num_accidents = st.slider(
            "Historical Accidents", 0, 7,
            value=d.get('num_reported_accidents', 1),
            key=f"{prefix}_accidents",
            help="Number of previously reported accidents at this location (max 7 in data).",
        )
        public_road = st.checkbox(
            "Public Road",
            value=bool(d.get('public_road', True)),
            key=f"{prefix}_public_road",
            help="Publicly maintained road (vs private).",
        )
        holiday = st.checkbox(
            "Holiday",
            value=bool(d.get('holiday', False)),
            key=f"{prefix}_holiday",
            help="Today is a public holiday.",
        )
        school_season = st.checkbox(
            "School Season",
            value=bool(d.get('school_season', True)),
            key=f"{prefix}_school",
            help="Schools are currently in session.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return {
        'road_type':              road_type,
        'num_lanes':              num_lanes,
        'curvature':              curvature,
        'speed_limit':            speed_limit,
        'lighting':               lighting,
        'weather':                weather,
        'road_signs_present':     road_signs,
        'public_road':            public_road,
        'time_of_day':            time_of_day,
        'holiday':                holiday,
        'school_season':          school_season,
        'num_reported_accidents': num_accidents,
    }


def show_result_card(pred: float, ci_low: float, ci_high: float):
    """Render the risk score card HTML."""
    level, emoji, color = risk_label(pred)
    st.markdown(f"""
    <div class="result-card" style="border-left-color:{color}">
        <p style="color:{color};font-size:1rem;font-weight:700;margin:0">{emoji} {level} RISK</p>
        <p style="font-size:3.5rem;font-weight:900;margin:4px 0;color:{color}">{pred*100:.1f}%</p>
        <p style="color:#999;font-size:0.9rem;margin:0">Predicted accident probability</p>
        <p style="color:#666;font-size:0.82rem;margin-top:6px">
            90% CI: {ci_low*100:.1f}% – {ci_high*100:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Road Risk AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Predict Risk", "⚖️ Compare Scenarios", "📦 Batch Predict",
         "📜 History", "📈 Model Info", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    val_r2 = metadata.get('val_r2')
    r2_str = f"{val_r2*100:.1f}%" if val_r2 else "N/A"
    st.markdown(f"""
    <div class="glass" style="padding:14px">
        <p style="margin:2px 0"><b>Model:</b> <span style="color:#6A82FB">{metadata.get('algorithm','RF').split()[0]} Forest</span></p>
        <p style="margin:2px 0"><b>Val R²:</b> <span style="color:#6A82FB">{r2_str}</span></p>
        <p style="margin:2px 0"><b>Features:</b> <span style="color:#6A82FB">{metadata.get('feature_count', len(feature_order))}</span></p>
        <p style="margin:2px 0"><b>Trained:</b> <span style="color:#6A82FB">{metadata.get('trained_at','—')}</span></p>
        <p style="margin:2px 0"><b>SHAP:</b> <span style="color:{'#4CAF50' if SHAP_AVAILABLE else '#888'}">{'✓ enabled' if SHAP_AVAILABLE else '✗ install shap'}</span></p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Always obey local traffic laws.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Risk":
    st.markdown("# 🔮 Predict Accident Risk")
    st.markdown(
        "<p style='color:#999'>Adjust the conditions below — risk updates instantly.</p>",
        unsafe_allow_html=True,
    )

    # Demo scenario button
    demo_col, _ = st.columns([1, 4])
    with demo_col:
        if st.button("🎲 Load Demo Scenario"):
            st.session_state.demo_idx = (st.session_state.demo_idx + 1) % len(DEMO_SCENARIOS)

    demo_defaults = DEMO_SCENARIOS[st.session_state.demo_idx]
    raw = input_widgets("pred", defaults=demo_defaults)

    # Validation warnings
    _, warnings = validate_input(raw)
    for w in warnings:
        st.warning(w)

    # ── Live prediction (no button needed) ────────────────────────────────────
    pred, ci_low, ci_high, shap_vals, base_val = run_prediction(raw)

    st.markdown("---")
    st.markdown("### 📊 Result")

    col_card, col_gauge = st.columns([1, 1.4])
    with col_card:
        show_result_card(pred, ci_low, ci_high)

        # Add to history
        entry = {
            'time':  datetime.now().strftime("%H:%M:%S"),
            'risk':  pred,
            'label': f"{raw['road_type'].title()} | {raw['weather']} | {raw['lighting']}",
            **raw,
        }
        # Update or append (deduplicate on same conditions)
        if (not st.session_state.history or
                st.session_state.history[-1].get('label') != entry['label'] or
                abs(st.session_state.history[-1]['risk'] - pred) > 0.001):
            st.session_state.history.append(entry)
            if len(st.session_state.history) > 50:
                st.session_state.history = st.session_state.history[-50:]

    with col_gauge:
        fig_gauge = create_gauge(pred, ci_low, ci_high)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SHAP explanation ──────────────────────────────────────────────────────
    if SHAP_AVAILABLE and shap_vals is not None:
        st.markdown("---")
        st.markdown("### 🧠 Why this score? (SHAP Feature Contributions)")
        st.caption(
            "Bars show each feature's push on the prediction. "
            "🔴 Red = increases risk · 🟢 Green = decreases risk"
        )
        fig_shap = create_shap_chart(shap_vals, feature_order, base_val, pred)
        st.plotly_chart(fig_shap, use_container_width=True)
    elif not SHAP_AVAILABLE:
        st.info("Install `shap` to see per-feature explanations: `pip install shap`")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Safety Recommendations")
    recs = generate_recommendations(raw, pred)
    for rec in recs:
        st.info(rec)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPARE SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Scenarios":
    st.markdown("# ⚖️ Compare Two Scenarios")
    st.markdown(
        "<p style='color:#999'>Configure Scenario A and B side-by-side to see which is riskier.</p>",
        unsafe_allow_html=True,
    )

    tab_a, tab_b = st.tabs(["🔵 Scenario A", "🔴 Scenario B"])

    with tab_a:
        raw_a = input_widgets("cmp_a", defaults=DEMO_SCENARIOS[0])

    with tab_b:
        raw_b = input_widgets("cmp_b", defaults=DEMO_SCENARIOS[1])

    # Predict both
    pred_a, ci_a_lo, ci_a_hi, _, _ = run_prediction(raw_a)
    pred_b, ci_b_lo, ci_b_hi, _, _ = run_prediction(raw_b)

    st.markdown("---")
    st.markdown("### 📊 Comparison")

    # Side-by-side gauges
    fig_cmp = create_comparison_gauges(pred_a, pred_b, "Scenario A", "Scenario B")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Delta and result cards
    col_a, col_mid, col_b = st.columns([1, 0.4, 1])
    with col_a:
        show_result_card(pred_a, ci_a_lo, ci_a_hi)
    with col_mid:
        delta = pred_b - pred_a
        arrow = "▲" if delta > 0 else "▼"
        color = "#F44336" if delta > 0 else "#4CAF50"
        st.markdown(f"""
        <div style="text-align:center;padding-top:40px">
            <p style="font-size:2rem;color:{color};margin:0">{arrow}</p>
            <p style="font-size:1.1rem;color:{color};font-weight:700;margin:0">
                {abs(delta)*100:.1f}%
            </p>
            <p style="color:#888;font-size:0.8rem">difference</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        show_result_card(pred_b, ci_b_lo, ci_b_hi)

    # Feature-by-feature comparison table
    st.markdown("---")
    st.markdown("### 📋 Condition Differences")
    diffs = []
    display = FEATURE_DISPLAY_NAMES
    for key in raw_a:
        val_a = raw_a[key]
        val_b = raw_b[key]
        if val_a != val_b:
            diffs.append({
                'Feature': display.get(key, key),
                'Scenario A': str(val_a),
                'Scenario B': str(val_b),
            })
    if diffs:
        st.dataframe(pd.DataFrame(diffs), use_container_width=True, hide_index=True)
    else:
        st.success("Both scenarios are identical.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Predict":
    st.markdown("# 📦 Batch Prediction")
    st.markdown(
        "<p style='color:#999'>Upload a CSV with road conditions to predict risk for multiple rows at once.</p>",
        unsafe_allow_html=True,
    )

    # Template download
    template_cols = [
        'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
        'weather', 'road_signs_present', 'public_road', 'time_of_day',
        'holiday', 'school_season', 'num_reported_accidents',
    ]
    template_row = {
        'road_type': 'urban', 'num_lanes': 2, 'curvature': 0.2, 'speed_limit': 50,
        'lighting': 'daylight', 'weather': 'clear', 'road_signs_present': True,
        'public_road': True, 'time_of_day': 'morning', 'holiday': False,
        'school_season': True, 'num_reported_accidents': 1,
    }
    template_df = pd.DataFrame([template_row])

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "📥 Download Template CSV",
            template_df.to_csv(index=False),
            file_name="batch_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"**{len(df_raw)} rows loaded.** Preview:")
            st.dataframe(df_raw.head(5), use_container_width=True)

            # Check required columns
            missing = [c for c in template_cols if c not in df_raw.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                with st.spinner("Predicting…"):
                    X_batch = encode_batch(df_raw[template_cols], encoders, feature_order)
                    preds   = model.predict(X_batch)
                    preds   = np.clip(preds, 0.0, 1.0)

                df_result = df_raw.copy()
                df_result['predicted_risk'] = preds
                df_result['risk_pct']       = (preds * 100).round(1)
                df_result['risk_level']     = [
                    risk_label(p)[0] for p in preds
                ]

                # Colour-coded display
                def colour_risk(val):
                    if val == 'HIGH':
                        return 'background-color: rgba(244,67,54,0.2)'
                    elif val == 'MEDIUM':
                        return 'background-color: rgba(255,152,0,0.15)'
                    return 'background-color: rgba(76,175,80,0.12)'

                st.markdown("### Results")
                st.dataframe(
                    df_result[['risk_pct', 'risk_level'] + template_cols]
                    .style.applymap(colour_risk, subset=['risk_level']),
                    use_container_width=True,
                    height=400,
                )

                # Summary stats
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Risk", f"{preds.mean()*100:.1f}%")
                c2.metric("High Risk Rows", int((preds >= 0.67).sum()))
                c3.metric("Low Risk Rows",  int((preds <  0.33).sum()))

                # Download results
                st.download_button(
                    "📤 Download Results CSV",
                    df_result.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.markdown("# 📜 Prediction History")
    st.markdown(
        "<p style='color:#999'>All predictions made this session.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No predictions yet — go to **🔮 Predict Risk** to get started.")
    else:
        col_btn, _ = st.columns([1, 5])
        with col_btn:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

        # Trend chart
        fig_hist = create_history_chart(st.session_state.history)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Table
        rows = []
        for i, h in enumerate(st.session_state.history, 1):
            level, emoji, _ = risk_label(h['risk'])
            rows.append({
                '#':        i,
                'Time':     h.get('time', '—'),
                'Risk':     f"{h['risk']*100:.1f}%",
                'Level':    f"{emoji} {level}",
                'Scenario': h.get('label', '—'),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        # Summary
        risks = [h['risk'] for h in st.session_state.history]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions", len(risks))
        c2.metric("Avg Risk", f"{np.mean(risks)*100:.1f}%")
        c3.metric("Highest", f"{max(risks)*100:.1f}%")
        c4.metric("Lowest",  f"{min(risks)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Info":
    st.markdown("# 📈 Model Information")
    st.markdown(
        "<p style='color:#999'>Training details and honest performance metrics.</p>",
        unsafe_allow_html=True,
    )

    # Metrics cards
    val_r2   = metadata.get('val_r2')
    val_mae  = metadata.get('val_mae')
    val_rmse = metadata.get('val_rmse')
    cv_mean  = metadata.get('cv_r2_mean')
    cv_std   = metadata.get('cv_r2_std')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation R²",  f"{val_r2*100:.1f}%"  if val_r2   else "N/A",
              help="R² on 20% held-out validation set (honest — not training data)")
    c2.metric("Val MAE",        f"{val_mae:.4f}"       if val_mae  else "N/A",
              help="Mean Absolute Error on validation set")
    c3.metric("Val RMSE",       f"{val_rmse:.4f}"      if val_rmse else "N/A",
              help="Root Mean Squared Error on validation set")
    c4.metric("CV R² (5-fold)", f"{cv_mean:.3f} ± {cv_std:.3f}" if cv_mean else "N/A",
              help="Cross-validated R² on training split")

    st.markdown("---")

    st.markdown("### 🎯 Risk Concentration — SDG 3.6 / 11.2 Evidence")
    st.markdown(
        "<p style='color:#999'>Ranking all 517,754 training rows by predicted risk: "
        "how much of total risk is concentrated in the highest-risk segments?</p>",
        unsafe_allow_html=True,
    )
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Top 1%", "2.50×", help="Top 1% of segments carry 2.5% of total risk mass — 2.50x their proportional share")
    rc2.metric("Top 5%", "2.22×", help="Top 5% of segments carry 11.1% of total risk mass")
    rc3.metric("Top 10%", "1.93×", help="Top 10% of segments carry 19.3% of total risk mass")
    rc4.metric("Top 25%", "1.64×", help="Top 25% of segments carry 41.0% of total risk mass")
    st.caption(
        "Concentration ratio = share of total predicted risk ÷ share of segments. "
        "This is the targeting evidence behind SDG 3.6 (halve road deaths/injuries) and "
        "11.2 (safe transport systems) — limited safety-intervention resources can be "
        "concentrated where predicted risk is highest, rather than spread evenly. "
        "See `ssd2026/IMPACT.md` for the full method and what this does *not* claim."
    )

    st.markdown("---")

    col_det, col_imp = st.columns([1, 1.5])

    with col_det:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🤖 Model Details")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Algorithm | {metadata.get('algorithm', 'Random Forest Regressor')} |
| Trees (n_estimators) | {metadata.get('n_estimators', '—')} |
| Max Depth | {metadata.get('max_depth', '—')} |
| Max Features | {metadata.get('max_features', '—')} |
| Training Samples | {metadata.get('training_samples', 0):,} |
| Total Features | {metadata.get('feature_count', len(feature_order))} |
| Trained On | {metadata.get('trained_at', '—')} |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        # Engineered features info
        eng = metadata.get('engineered_features', [])
        if eng:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Engineered Features")
            for f in eng:
                st.markdown(f"- **{FEATURE_DISPLAY_NAMES.get(f, f)}**")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_imp:
        st.markdown("### 🎯 Feature Importance")
        if hasattr(model, 'feature_importances_'):
            fig_imp = create_importance_chart(model.feature_importances_, feature_order)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    # Note on metric honesty
    st.markdown("---")
    st.info(
        "ℹ️ **About these metrics**: R² and MAE are computed on a **held-out 20% validation set** "
        "that the model never saw during training. This gives an honest estimate of real-world performance. "
        "The original app mistakenly reported training-set R² (always inflated for Random Forests)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("# ℹ️ About This Project")

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("""
### 🚗 Road Accident Risk Predictor

An AI-powered web application that predicts road accident probability based on
12 environmental and road-condition features — with SHAP explanations, confidence
intervals, scenario comparison, and batch prediction.

#### 🎯 Purpose
- Identify high-risk road scenarios before travel
- Understand *which* factors drive risk (SHAP explanations)
- Compare alternative routes or conditions side-by-side
- Process bulk location data via CSV

#### 🛠️ Technology Stack
| Component | Technology |
|-----------|-----------|
| Web Framework | Streamlit |
| ML Model | Random Forest Regressor (sklearn) |
| Explainability | SHAP TreeExplainer |
| Visualisation | Plotly |
| Data | Kaggle Playground Series S5E10 |

#### 📊 Dataset
- **Source**: [Kaggle Playground Series S5E10](https://www.kaggle.com/competitions/playground-series-s5e10)
- **Training Data**: 517,754 samples
- **Features**: 12 raw + 3 engineered interaction features

#### 🔗 Links
- **GitHub**: [roshanaryal1/SafeRoute](https://github.com/roshanaryal1/SafeRoute)

---

### 🌍 SDG Alignment

Targets **UN SDG 3.6** (halve global deaths and injuries from road traffic
accidents) and **SDG 11.2** (safe, affordable, sustainable transport
systems). **Mechanism:** risk scoring → targeted infrastructure spend →
fewer serious injuries — the model identifies which road conditions carry
disproportionate predicted risk, so limited safety-engineering resources
can be prioritized rather than spread evenly.

Ranking all 517,754 training rows by predicted risk, the **top 10%
highest-risk conditions carry 1.93×** their proportional share of total
predicted risk. Full methodology and an explicit statement of what this
does *not* claim: see `ssd2026/IMPACT.md` in the repository.

### 🛡️ Responsible AI

Uses road and environmental conditions only — no driver identity or
demographic data. Intended for infrastructure planning and safety
prioritization, **not** individual driver liability, profiling, or
enforcement. This is a live deployed decision-support prototype, not a
production system with an operator or SLA. Full limitations (synthetic
training data, no geographic field, subgroup performance by lighting/
weather) documented in `ssd2026/RESPONSIBLE_AI.md`.

---
⚠️ **Disclaimer**: This is an educational ML project. Always follow local traffic laws and
exercise caution while driving regardless of any predicted risk score.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
