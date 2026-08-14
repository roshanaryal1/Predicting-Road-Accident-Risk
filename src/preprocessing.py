"""
Input preprocessing: feature engineering, encoding, and validation.
All functions work on both training DataFrames and single-prediction dicts.
"""

import pandas as pd
import numpy as np

# ── Column definitions (match training data) ──────────────────────────────────
CATEGORICAL_COLS = ['road_type', 'lighting', 'weather', 'time_of_day']
BOOL_COLS = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERIC_COLS = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
ENGINEERED_COLS = ['speed_curvature', 'accident_density', 'is_night_bad_weather']

# ── Valid options (from training data) ────────────────────────────────────────
WEATHER_OPTIONS  = ['clear', 'foggy', 'rainy']
LIGHTING_OPTIONS = ['daylight', 'dim', 'night']
ROAD_TYPE_OPTIONS = ['highway', 'rural', 'urban']
TIME_OPTIONS     = ['afternoon', 'evening', 'morning']

# Human-readable display names for features (used in SHAP + importance charts)
FEATURE_DISPLAY_NAMES = {
    'road_type':              'Road Type',
    'num_lanes':              'Number of Lanes',
    'curvature':              'Road Curvature',
    'speed_limit':            'Speed Limit (km/h)',
    'lighting':               'Lighting',
    'weather':                'Weather',
    'road_signs_present':     'Road Signs Present',
    'public_road':            'Public Road',
    'time_of_day':            'Time of Day',
    'holiday':                'Holiday',
    'school_season':          'School Season',
    'num_reported_accidents': 'Historical Accidents',
    'speed_curvature':        'Speed × Curvature',
    'accident_density':       'Accidents per Lane',
    'is_night_bad_weather':   'Night + Bad Weather',
}

# Training data ranges (used for validation warnings)
TRAINING_RANGES = {
    'num_lanes':              (1, 4),
    'curvature':              (0.0, 1.0),
    'speed_limit':            (25, 70),
    'num_reported_accidents': (0, 7),
}

# Defaults for "Random Scenario" demo button
DEMO_SCENARIOS = [
    {
        '_label': 'Low risk — motorway, daylight, clear, signed',
        'road_type': 'highway', 'num_lanes': 3, 'curvature': 0.1,
        'speed_limit': 70, 'lighting': 'daylight', 'weather': 'clear',
        'road_signs_present': True, 'public_road': True,
        'time_of_day': 'morning', 'holiday': False, 'school_season': True,
        'num_reported_accidents': 1,
    },
    {
        '_label': 'Elevated — urban street, night, rain, unsigned',
        'road_type': 'urban', 'num_lanes': 2, 'curvature': 0.4,
        'speed_limit': 50, 'lighting': 'night', 'weather': 'rainy',
        'road_signs_present': False, 'public_road': True,
        'time_of_day': 'evening', 'holiday': True, 'school_season': False,
        'num_reported_accidents': 5,
    },
    {
        '_label': 'High risk — rural single lane, fog, sharp curve',
        'road_type': 'rural', 'num_lanes': 1, 'curvature': 0.8,
        'speed_limit': 60, 'lighting': 'dim', 'weather': 'foggy',
        'road_signs_present': False, 'public_road': True,
        'time_of_day': 'morning', 'holiday': False, 'school_season': False,
        'num_reported_accidents': 7,
    },
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features derived from raw (pre-encoded) values.
    Safe to call on both training DataFrames (object dtype for categoricals)
    and prediction DataFrames built from user input.
    """
    df = df.copy()

    # Speed × curvature — fast road + sharp bend compounds risk
    df['speed_curvature'] = df['speed_limit'] * df['curvature']

    # Accidents per lane — normalises accident history by road width
    df['accident_density'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)

    # Night + bad weather flag (created while columns are still strings).
    # Use is_string_dtype() instead of dtype == object so this works with
    # both plain object columns AND Arrow-backed strings (used on Streamlit Cloud).
    if pd.api.types.is_string_dtype(df['weather']):
        bad_weather = df['weather'].isin(['rainy', 'foggy'])
        dark        = df['lighting'].isin(['night', 'dim'])
    else:
        # Columns already numerically encoded by LabelEncoder
        # rainy=2, foggy=1, clear=0  |  night=2, dim=1, daylight=0
        bad_weather = df['weather'].astype(int) > 0
        dark        = df['lighting'].astype(int) > 0

    df['is_night_bad_weather'] = (bad_weather & dark).astype(int)

    return df


def encode_input(raw_input: dict, encoders: dict, feature_order: list) -> pd.DataFrame:
    """
    Convert a raw user-input dict into a model-ready DataFrame.

    Steps:
      1. Build single-row DataFrame
      2. Apply feature engineering (on raw string values)
      3. Encode categorical string columns via saved LabelEncoders
      4. Encode boolean columns via saved LabelEncoders
      5. Reorder columns to match training feature_order
    """
    df = pd.DataFrame([raw_input])

    # Feature engineering first (while weather/lighting are still strings)
    df = engineer_features(df)

    # Encode categorical string columns
    for col in CATEGORICAL_COLS:
        if col in encoders and col in df.columns:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Encode boolean columns — encoders were fit on strings "True"/"False"
    for col in BOOL_COLS:
        if col in encoders and col in df.columns:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Reorder to exactly match training feature order
    df = df[feature_order]
    return df


def encode_batch(df_raw: pd.DataFrame, encoders: dict, feature_order: list) -> pd.DataFrame:
    """
    Encode a DataFrame of multiple rows (for batch prediction).
    Expects string values for categoricals and bool/string for booleans.
    """
    df = df_raw.copy()

    # Drop id column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Feature engineering
    df = engineer_features(df)

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in encoders and col in df.columns:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Encode booleans — encoders were fit on strings "True"/"False"
    for col in BOOL_COLS:
        if col in encoders and col in df.columns:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Keep only columns in feature_order (engineered cols are added automatically)
    available = [c for c in feature_order if c in df.columns]
    df = df[available]
    return df


def validate_input(raw_input: dict) -> tuple[bool, list[str]]:
    """
    Validate input values. Returns (all_ok, list_of_warnings).
    Warnings don't block prediction but are shown in the UI.
    """
    warnings = []

    curvature = raw_input.get('curvature', 0)
    if not 0.0 <= curvature <= 1.0:
        warnings.append(f"Curvature {curvature:.2f} is outside valid range 0.0–1.0.")

    speed = raw_input.get('speed_limit', 0)
    lo, hi = TRAINING_RANGES['speed_limit']
    if not lo <= speed <= hi:
        warnings.append(
            f"Speed limit {speed} km/h is outside training range ({lo}–{hi} km/h) — "
            "prediction may be less accurate."
        )

    accidents = raw_input.get('num_reported_accidents', 0)
    if accidents > TRAINING_RANGES['num_reported_accidents'][1]:
        warnings.append(
            f"Historical accidents ({accidents}) exceeds training max (7) — "
            "prediction may be less accurate."
        )

    all_ok = len(warnings) == 0
    return all_ok, warnings


def get_display_name(col: str) -> str:
    """Return human-readable name for a feature column."""
    return FEATURE_DISPLAY_NAMES.get(col, col.replace('_', ' ').title())
