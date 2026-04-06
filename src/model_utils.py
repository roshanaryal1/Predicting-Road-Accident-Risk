"""
Model loading, prediction, confidence intervals, and SHAP computation.
"""

import os
import json
import warnings
import numpy as np
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Cached explainer so we don't rebuild on every prediction
_shap_explainer = None


def load_model(model_dir: str = 'model') -> tuple:
    """
    Load trained model artefacts from disk.

    Returns:
        (model, encoders, feature_order, metadata_dict)
        metadata_dict is empty dict if metadata.json not found.
    """
    model        = joblib.load(os.path.join(model_dir, 'accident_risk_model.pkl'))
    encoders     = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    feature_order = joblib.load(os.path.join(model_dir, 'feature_order.pkl'))

    metadata_path = os.path.join(model_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        # Fallback for legacy model files (without metadata)
        metadata = {
            'algorithm': 'Random Forest Regressor',
            'n_estimators': 100,
            'training_samples': 517754,
            'val_r2': None,
            'val_mae': None,
            'val_rmse': None,
            'cv_r2_mean': None,
            'cv_r2_std': None,
            'feature_count': len(feature_order),
        }

    return model, encoders, feature_order, metadata


def predict_with_confidence(model, X) -> tuple[float, float, float]:
    """
    Predict a single sample and compute a 90 % confidence interval
    from the distribution of individual tree predictions.

    Returns:
        (mean_prediction, ci_low_5th_pct, ci_high_95th_pct)
    """
    mean_pred = float(model.predict(X)[0])

    # Collect per-tree predictions for uncertainty estimation
    # Suppress sklearn's "fitted without feature names" warning for individual trees
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree_preds = np.array([
            tree.predict(X)[0]
            for tree in model.estimators_
        ])
    ci_low  = float(np.percentile(tree_preds, 5))
    ci_high = float(np.percentile(tree_preds, 95))

    return mean_pred, ci_low, ci_high


def compute_shap(model, X, feature_order: list) -> tuple[np.ndarray, float] | tuple[None, None]:
    """
    Compute SHAP values for a single-row DataFrame X.

    Returns:
        (shap_values_1d, base_value) or (None, None) if SHAP not available.
    """
    global _shap_explainer

    if not SHAP_AVAILABLE:
        return None, None

    if _shap_explainer is None:
        _shap_explainer = shap.TreeExplainer(model)

    shap_vals = _shap_explainer.shap_values(X)

    # shap_values shape for regression RF: (n_samples, n_features)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    base_value = float(_shap_explainer.expected_value)
    return shap_vals, base_value
