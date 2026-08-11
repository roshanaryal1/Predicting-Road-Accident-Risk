"""
Plotly chart builders for the Streamlit UI.
Each function returns a go.Figure — pass to st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.preprocessing import get_display_name


# ── Colour palette ────────────────────────────────────────────────────────────
LOW_COLOR    = '#4CAF50'   # green
MEDIUM_COLOR = '#FF9800'   # orange
HIGH_COLOR   = '#F44336'   # red
UP_COLOR     = '#FF6B6B'   # SHAP increase
DOWN_COLOR   = '#6BCB77'   # SHAP decrease
BRAND_BLUE   = '#6A82FB'
BRAND_PINK   = '#FC5C7D'


def risk_color(score: float) -> str:
    if score < 0.33:
        return LOW_COLOR
    elif score < 0.67:
        return MEDIUM_COLOR
    return HIGH_COLOR


def risk_label(score: float) -> tuple[str, str, str]:
    """Returns (level_str, emoji, hex_color)."""
    if score < 0.33:
        return "LOW", "🟢", LOW_COLOR
    elif score < 0.67:
        return "MEDIUM", "🟡", MEDIUM_COLOR
    return "HIGH", "🔴", HIGH_COLOR


# ── Gauge chart ───────────────────────────────────────────────────────────────

def create_gauge(risk_score: float, ci_low: float = None, ci_high: float = None) -> go.Figure:
    """
    Improved risk gauge with optional 90 % confidence band annotation.
    The needle/bar colour matches the risk zone.
    """
    pct    = risk_score * 100
    color  = risk_color(risk_score)
    label, _, _ = risk_label(risk_score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number" if ci_low is not None else "gauge+number+delta",
        value=pct,
        number={'suffix': '%', 'font': {'size': 36, 'color': color}},
        delta={'reference': 50, 'relative': False,
               'increasing': {'color': HIGH_COLOR},
               'decreasing': {'color': LOW_COLOR}} if ci_low is None else None,
        domain={'x': [0, 1], 'y': [0.16, 1] if ci_low is not None else [0, 1]},
        title={'text': f"<b>{label} RISK</b>", 'font': {'size': 18, 'color': color}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#888',
                'tickfont': {'color': '#ccc'},
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33],  'color': 'rgba(76,175,80,0.15)'},
                {'range': [33, 67], 'color': 'rgba(255,152,0,0.15)'},
                {'range': [67, 100],'color': 'rgba(244,67,54,0.15)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': pct,
            },
        },
    ))

    # Confidence interval annotation — pinned below the gauge's own domain
    # (which is shrunk to [0.16, 1] above) so it never collides with the
    # number/title Plotly places inside the indicator itself.
    if ci_low is not None and ci_high is not None:
        fig.add_annotation(
            text=f"90% CI: {ci_low*100:.1f}% – {ci_high*100:.1f}%",
            x=0.5, y=0.02,
            xref='paper', yref='paper',
            showarrow=False,
            font={'size': 12, 'color': '#aaa'},
        )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa'},
    )
    return fig


# ── SHAP waterfall ────────────────────────────────────────────────────────────

def create_shap_chart(
    shap_values: np.ndarray,
    feature_names: list[str],
    base_value: float,
    prediction: float,
    top_n: int = 10,
) -> go.Figure:
    """
    Horizontal waterfall-style bar chart showing per-feature SHAP contributions.
    Positive values (increase risk) shown in red; negative (decrease) in green.
    """
    # Sort by absolute importance, take top_n
    idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    names  = [get_display_name(feature_names[i]) for i in idx]
    values = [shap_values[i] for i in idx]

    # Reverse so largest bar is at top
    names  = names[::-1]
    values = values[::-1]

    colors = [UP_COLOR if v > 0 else DOWN_COLOR for v in values]
    text   = [f"+{v*100:.1f}%" if v > 0 else f"{v*100:.1f}%" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        text=text,
        textposition='outside',
        textfont={'size': 11, 'color': '#eee'},
        hovertemplate='%{y}: %{text}<extra></extra>',
    ))

    fig.add_vline(x=0, line_width=1, line_color='#666')

    fig.update_layout(
        title={
            'text': (
                f"Why {prediction*100:.1f}%?  "
                f"<span style='font-size:13px;color:#aaa'>"
                f"Base: {base_value*100:.1f}%</span>"
            ),
            'font': {'size': 15},
        },
        xaxis_title='Contribution to Risk Score',
        xaxis=dict(tickformat='.0%', tickfont={'color': '#ccc'}, gridcolor='#333'),
        yaxis=dict(tickfont={'size': 12, 'color': '#ddd'}),
        height=max(280, top_n * 30 + 80),
        margin=dict(l=10, r=60, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa'},
        showlegend=False,
    )
    return fig


# ── Feature importance ────────────────────────────────────────────────────────

def create_importance_chart(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 12,
) -> go.Figure:
    """
    Horizontal Plotly bar chart for global feature importance (colour-coded).
    """
    df = pd.DataFrame({
        'feature': [get_display_name(f) for f in feature_names],
        'importance': importances,
    }).nlargest(top_n, 'importance').sort_values('importance')

    # Gradient colour: low importance = blue, high = pink
    norm = (df['importance'] - df['importance'].min()) / (
        df['importance'].max() - df['importance'].min() + 1e-9
    )
    colors = [
        f"rgb({int(106 + (252-106)*v)},{int(130 + (92-130)*v)},{int(251 + (125-251)*v)})"
        for v in norm
    ]

    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in df['importance']],
        textposition='outside',
        textfont={'size': 10, 'color': '#ccc'},
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))

    fig.update_layout(
        title='Feature Importance (MDI)',
        xaxis_title='Mean Decrease in Impurity',
        xaxis=dict(tickfont={'color': '#ccc'}, gridcolor='#333'),
        yaxis=dict(tickfont={'size': 11, 'color': '#ddd'}),
        height=max(320, top_n * 28 + 80),
        margin=dict(l=10, r=60, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa'},
        showlegend=False,
    )
    return fig


# ── Comparison chart ──────────────────────────────────────────────────────────

def create_comparison_gauges(score_a: float, score_b: float,
                              label_a: str = "Scenario A",
                              label_b: str = "Scenario B") -> go.Figure:
    """
    Side-by-side bullet gauges for scenario comparison.
    """
    color_a = risk_color(score_a)
    color_b = risk_color(score_b)

    fig = go.Figure()

    for i, (score, label, color) in enumerate([
        (score_a, label_a, color_a),
        (score_b, label_b, color_b),
    ]):
        x0, x1 = (0.0, 0.45) if i == 0 else (0.55, 1.0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={'suffix': '%', 'font': {'size': 28, 'color': color}},
            title={'text': f"<b>{label}</b>", 'font': {'size': 14, 'color': '#ddd'}},
            domain={'x': [x0, x1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': '#999'}},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 33],  'color': 'rgba(76,175,80,0.12)'},
                    {'range': [33, 67], 'color': 'rgba(255,152,0,0.12)'},
                    {'range': [67, 100],'color': 'rgba(244,67,54,0.12)'},
                ],
                'threshold': {
                    'line': {'color': color, 'width': 3},
                    'thickness': 0.8,
                    'value': score * 100,
                },
            },
        ))

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa'},
    )
    return fig


# ── History trend ─────────────────────────────────────────────────────────────

def create_history_chart(history: list[dict]) -> go.Figure:
    """Line chart of risk scores over prediction index."""
    if not history:
        return go.Figure()

    indices = list(range(1, len(history) + 1))
    scores  = [h['risk'] * 100 for h in history]
    colors  = [risk_color(h['risk']) for h in history]
    labels  = [h.get('label', f"#{i}") for i, h in enumerate(history, 1)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=indices,
        y=scores,
        mode='lines+markers',
        line={'color': BRAND_BLUE, 'width': 2},
        marker={'color': colors, 'size': 10, 'line': {'color': '#fff', 'width': 1}},
        text=labels,
        hovertemplate='Prediction #%{x}<br>Risk: %{y:.1f}%<br>%{text}<extra></extra>',
    ))

    fig.add_hrect(y0=0,  y1=33, fillcolor=LOW_COLOR,    opacity=0.05, line_width=0)
    fig.add_hrect(y0=33, y1=67, fillcolor=MEDIUM_COLOR, opacity=0.05, line_width=0)
    fig.add_hrect(y0=67, y1=100,fillcolor=HIGH_COLOR,   opacity=0.05, line_width=0)

    fig.update_layout(
        title='Risk Score History',
        xaxis_title='Prediction #',
        yaxis_title='Risk Score (%)',
        yaxis={'range': [0, 100], 'tickfont': {'color': '#ccc'}, 'gridcolor': '#333'},
        xaxis={'tickfont': {'color': '#ccc'}, 'gridcolor': '#333'},
        height=300,
        margin=dict(l=10, r=10, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa'},
    )
    return fig
