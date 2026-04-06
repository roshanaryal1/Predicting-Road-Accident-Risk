"""
Test suite for Road Accident Risk Predictor.
Run with: pytest test_app.py -v
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

# ── Make sure src/ is importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing  import (
    engineer_features, encode_input, validate_input,
    WEATHER_OPTIONS, LIGHTING_OPTIONS, ROAD_TYPE_OPTIONS, TIME_OPTIONS,
    CATEGORICAL_COLS, BOOL_COLS,
)
from src.recommendations import generate_recommendations


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAFE_INPUT = {
    'road_type': 'highway', 'num_lanes': 3, 'curvature': 0.05,
    'speed_limit': 60, 'lighting': 'daylight', 'weather': 'clear',
    'road_signs_present': True, 'public_road': True,
    'time_of_day': 'morning', 'holiday': False, 'school_season': True,
    'num_reported_accidents': 0,
}

RISKY_INPUT = {
    'road_type': 'rural', 'num_lanes': 1, 'curvature': 0.9,
    'speed_limit': 70, 'lighting': 'night', 'weather': 'foggy',
    'road_signs_present': False, 'public_road': True,
    'time_of_day': 'evening', 'holiday': True, 'school_season': False,
    'num_reported_accidents': 7,
}


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_adds_speed_curvature(self):
        df = pd.DataFrame([SAFE_INPUT])
        result = engineer_features(df)
        assert 'speed_curvature' in result.columns
        expected = SAFE_INPUT['speed_limit'] * SAFE_INPUT['curvature']
        assert abs(result['speed_curvature'].iloc[0] - expected) < 1e-9

    def test_adds_accident_density(self):
        df = pd.DataFrame([SAFE_INPUT])
        result = engineer_features(df)
        assert 'accident_density' in result.columns
        expected = SAFE_INPUT['num_reported_accidents'] / (SAFE_INPUT['num_lanes'] + 1)
        assert abs(result['accident_density'].iloc[0] - expected) < 1e-9

    def test_night_bad_weather_flag_true(self):
        risky = RISKY_INPUT.copy()
        risky['lighting'] = 'night'
        risky['weather']  = 'foggy'
        df = pd.DataFrame([risky])
        result = engineer_features(df)
        assert result['is_night_bad_weather'].iloc[0] == 1

    def test_night_bad_weather_flag_false(self):
        safe = SAFE_INPUT.copy()
        safe['lighting'] = 'daylight'
        safe['weather']  = 'clear'
        df = pd.DataFrame([safe])
        result = engineer_features(df)
        assert result['is_night_bad_weather'].iloc[0] == 0

    def test_dim_rainy_triggers_flag(self):
        inp = SAFE_INPUT.copy()
        inp['lighting'] = 'dim'
        inp['weather']  = 'rainy'
        df = pd.DataFrame([inp])
        result = engineer_features(df)
        assert result['is_night_bad_weather'].iloc[0] == 1

    def test_preserves_original_columns(self):
        df = pd.DataFrame([SAFE_INPUT])
        result = engineer_features(df)
        for col in SAFE_INPUT:
            assert col in result.columns

    def test_works_on_batch(self):
        df = pd.DataFrame([SAFE_INPUT, RISKY_INPUT])
        result = engineer_features(df)
        assert len(result) == 2
        assert 'speed_curvature' in result.columns


# ── Input validation tests ────────────────────────────────────────────────────

class TestValidation:

    def test_valid_input_passes(self):
        ok, warnings = validate_input(SAFE_INPUT)
        assert ok is True
        assert warnings == []

    def test_curvature_out_of_range(self):
        bad = SAFE_INPUT.copy()
        bad['curvature'] = 1.5
        ok, warnings = validate_input(bad)
        assert not ok
        assert any('urvature' in w for w in warnings)

    def test_accidents_over_max_warns(self):
        over = SAFE_INPUT.copy()
        over['num_reported_accidents'] = 10
        ok, warnings = validate_input(over)
        assert any('accident' in w.lower() for w in warnings)

    def test_speed_out_of_range_warns(self):
        fast = SAFE_INPUT.copy()
        fast['speed_limit'] = 120
        _, warnings = validate_input(fast)
        assert any('speed' in w.lower() or 'km/h' in w for w in warnings)


# ── Recommendation engine tests ───────────────────────────────────────────────

class TestRecommendations:

    def test_rainy_recommendation_fires(self):
        inp = SAFE_INPUT.copy()
        inp['weather'] = 'rainy'
        recs = generate_recommendations(inp, 0.4)
        assert any('wet' in r.lower() or 'rain' in r.lower() for r in recs)

    def test_foggy_recommendation_fires(self):
        inp = SAFE_INPUT.copy()
        inp['weather'] = 'foggy'
        recs = generate_recommendations(inp, 0.4)
        assert any('fog' in r.lower() for r in recs)

    def test_night_recommendation_fires(self):
        inp = SAFE_INPUT.copy()
        inp['lighting'] = 'night'
        recs = generate_recommendations(inp, 0.3)
        assert any('night' in r.lower() or 'headlight' in r.lower() for r in recs)

    def test_dim_recommendation_fires(self):
        inp = SAFE_INPUT.copy()
        inp['lighting'] = 'dim'
        recs = generate_recommendations(inp, 0.3)
        assert any('dim' in r.lower() or 'headlight' in r.lower() for r in recs)

    def test_high_curvature_fires(self):
        inp = SAFE_INPUT.copy()
        inp['curvature'] = 0.8
        recs = generate_recommendations(inp, 0.5)
        assert any('curve' in r.lower() or 'bend' in r.lower() for r in recs)

    def test_medium_curvature_fires(self):
        inp = SAFE_INPUT.copy()
        inp['curvature'] = 0.5
        recs = generate_recommendations(inp, 0.4)
        assert any('curve' in r.lower() or 'lane' in r.lower() for r in recs)

    def test_high_accidents_fires(self):
        inp = SAFE_INPUT.copy()
        inp['num_reported_accidents'] = 6
        recs = generate_recommendations(inp, 0.5)
        assert any('accident' in r.lower() for r in recs)

    def test_no_road_signs_fires(self):
        inp = SAFE_INPUT.copy()
        inp['road_signs_present'] = False
        recs = generate_recommendations(inp, 0.3)
        assert any('sign' in r.lower() for r in recs)

    def test_high_risk_overall_warning(self):
        recs = generate_recommendations(RISKY_INPUT, 0.85)
        assert any('high risk' in r.lower() for r in recs)

    def test_safe_conditions_fallback(self):
        # Use a truly minimal input with no triggering conditions
        minimal_safe = {
            'road_type': 'urban', 'num_lanes': 2, 'curvature': 0.1,
            'speed_limit': 30, 'lighting': 'daylight', 'weather': 'clear',
            'road_signs_present': True, 'public_road': True,
            'time_of_day': 'evening', 'holiday': False, 'school_season': False,
            'num_reported_accidents': 0,
        }
        recs = generate_recommendations(minimal_safe, 0.1)
        assert any('safe' in r.lower() for r in recs)

    def test_holiday_fires(self):
        inp = SAFE_INPUT.copy()
        inp['holiday'] = True
        recs = generate_recommendations(inp, 0.3)
        assert any('holiday' in r.lower() for r in recs)

    def test_night_plus_foggy_combined_fires(self):
        inp = SAFE_INPUT.copy()
        inp['lighting'] = 'night'
        inp['weather']  = 'foggy'
        recs = generate_recommendations(inp, 0.6)
        assert any('night' in r.lower() and ('weather' in r.lower() or 'visibility' in r.lower())
                   for r in recs)

    def test_rural_recommendation_fires(self):
        inp = SAFE_INPUT.copy()
        inp['road_type'] = 'rural'
        recs = generate_recommendations(inp, 0.3)
        assert any('rural' in r.lower() for r in recs)

    def test_always_returns_at_least_one_rec(self):
        for raw in [SAFE_INPUT, RISKY_INPUT]:
            recs = generate_recommendations(raw, 0.5)
            assert len(recs) >= 1


# ── Metadata tests (skip if model not trained) ────────────────────────────────

class TestMetadata:

    @pytest.mark.skipif(not os.path.exists('model/metadata.json'),
                        reason="Model not trained yet")
    def test_metadata_has_required_keys(self):
        with open('model/metadata.json') as f:
            meta = json.load(f)
        required = ['algorithm', 'training_samples', 'val_r2', 'val_mae',
                    'cv_r2_mean', 'feature_order', 'trained_at']
        for key in required:
            assert key in meta, f"Missing key: {key}"

    @pytest.mark.skipif(not os.path.exists('model/metadata.json'),
                        reason="Model not trained yet")
    def test_val_r2_is_realistic(self):
        with open('model/metadata.json') as f:
            meta = json.load(f)
        r2 = meta['val_r2']
        # Should be between 0.5 and 0.99 for a reasonable model
        assert 0.5 <= r2 <= 0.99, f"Val R² {r2} looks suspicious"

    @pytest.mark.skipif(not os.path.exists('model/metadata.json'),
                        reason="Model not trained yet")
    def test_feature_order_matches_expected_count(self):
        with open('model/metadata.json') as f:
            meta = json.load(f)
        # 12 original + 3 engineered = 15
        assert len(meta['feature_order']) >= 12


# ── End-to-end prediction tests (skip if model not trained) ───────────────────

class TestEndToEnd:

    @pytest.fixture(scope='class')
    def model_artefacts(self):
        if not os.path.exists('model/accident_risk_model.pkl'):
            pytest.skip("Model not trained yet — run train_and_save_model.py first")
        from src.model_utils import load_model
        return load_model()

    def test_prediction_in_range(self, model_artefacts):
        model, encoders, feature_order, _ = model_artefacts
        X = encode_input(SAFE_INPUT, encoders, feature_order)
        pred = model.predict(X)[0]
        assert 0.0 <= pred <= 1.0, f"Prediction {pred} out of range"

    def test_risky_higher_than_safe(self, model_artefacts):
        model, encoders, feature_order, _ = model_artefacts
        X_safe  = encode_input(SAFE_INPUT,  encoders, feature_order)
        X_risky = encode_input(RISKY_INPUT, encoders, feature_order)
        pred_safe  = model.predict(X_safe)[0]
        pred_risky = model.predict(X_risky)[0]
        assert pred_risky > pred_safe, (
            f"Risky scenario ({pred_risky:.3f}) should score higher than safe ({pred_safe:.3f})"
        )

    def test_confidence_interval_makes_sense(self, model_artefacts):
        from src.model_utils import predict_with_confidence
        model, encoders, feature_order, _ = model_artefacts
        X = encode_input(SAFE_INPUT, encoders, feature_order)
        pred, ci_low, ci_high = predict_with_confidence(model, X)
        assert ci_low <= pred <= ci_high, "Mean prediction should be inside CI"
        assert ci_high - ci_low < 0.5,   "CI width > 0.5 seems too wide"

    def test_feature_order_length(self, model_artefacts):
        _, _, feature_order, _ = model_artefacts
        assert len(feature_order) >= 12
