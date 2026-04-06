"""
Safety recommendation engine.
All conditions match the actual label values used in training data.
"""


def generate_recommendations(raw_input: dict, risk_score: float) -> list[str]:
    """
    Generate context-aware safety tips based on actual road conditions.

    Args:
        raw_input: dict of raw (pre-encoding) input values
        risk_score: predicted accident risk probability (0.0–1.0)

    Returns:
        List of recommendation strings (with emoji prefix)
    """
    recs = []

    # ── Overall risk level ────────────────────────────────────────────────────
    if risk_score >= 0.67:
        recs.append("🚨 HIGH RISK detected — extreme caution required. Consider delaying travel.")
    elif risk_score >= 0.33:
        recs.append("⚠️ MEDIUM RISK — drive attentively and be prepared for hazards.")

    # ── Weather conditions ────────────────────────────────────────────────────
    weather = raw_input.get('weather', 'clear')
    if weather == 'rainy':
        recs.append("🌧️ Wet road surface — reduce speed by 20–30% and increase following distance.")
    elif weather == 'foggy':
        recs.append("🌫️ Fog reduces visibility — use fog lights and drive significantly slower.")

    # ── Lighting conditions ───────────────────────────────────────────────────
    lighting = raw_input.get('lighting', 'daylight')
    if lighting == 'night':
        recs.append("🌙 Night driving — ensure headlights are on and watch for pedestrians.")
    elif lighting == 'dim':
        recs.append("🔦 Dim lighting — switch on headlights and reduce speed.")

    # ── Combined night + bad weather (highest visibility risk) ────────────────
    if lighting in ('night', 'dim') and weather in ('rainy', 'foggy'):
        recs.append("👁️ Night + poor weather — visibility is severely reduced. Drive with extreme care.")

    # ── Speed ─────────────────────────────────────────────────────────────────
    speed = raw_input.get('speed_limit', 0)
    if speed >= 60:
        recs.append(f"⚡ High speed zone ({speed} km/h) — maintain 3-second following distance.")

    # ── Road curvature (0.0–1.0) ──────────────────────────────────────────────
    curvature = raw_input.get('curvature', 0.0)
    if curvature >= 0.7:
        recs.append("🔄 Sharp curves ahead — slow down well before the bend, not during it.")
    elif curvature >= 0.4:
        recs.append("↩️ Moderate curves — stay in your lane and avoid sudden steering corrections.")

    # ── Accident history ──────────────────────────────────────────────────────
    accidents = raw_input.get('num_reported_accidents', 0)
    if accidents >= 5:
        recs.append(f"⚠️ High-risk location: {accidents} prior accidents recorded — stay alert.")
    elif accidents >= 2:
        recs.append(f"📋 {accidents} prior accidents at this location — heightened caution advised.")

    # ── Road signs ────────────────────────────────────────────────────────────
    if not raw_input.get('road_signs_present', True):
        recs.append("🚧 No road signs present — navigate cautiously and follow road markings.")

    # ── Contextual factors ────────────────────────────────────────────────────
    if raw_input.get('holiday', False):
        recs.append("🎉 Holiday — expect heavier traffic and potentially impaired drivers.")

    if raw_input.get('school_season', False) and raw_input.get('time_of_day') in ('morning', 'afternoon'):
        recs.append("🏫 School season + peak hours — watch for children near crossings.")

    road_type = raw_input.get('road_type', 'urban')
    if road_type == 'rural':
        recs.append("🌾 Rural road — watch for animals, agricultural vehicles and unmarked junctions.")

    # ── Fallback if everything looks fine ─────────────────────────────────────
    if not recs or (risk_score < 0.33 and len(recs) <= 1):
        recs.append("✅ Conditions are relatively safe — maintain standard driving practices.")

    return recs
