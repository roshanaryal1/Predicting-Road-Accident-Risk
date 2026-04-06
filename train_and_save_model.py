"""
Training pipeline for the Road Accident Risk Predictor.

Improvements over the original:
  - Feature engineering (interaction terms) for better accuracy
  - Proper 80/20 train/validation split for honest metrics
  - 5-fold cross-validation
  - Better-regularised RandomForestRegressor hyperparameters
  - Saves model/metadata.json with true validation metrics
  - Consistent preprocessing via src/preprocessing module
"""

import os
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.preprocessing import engineer_features

warnings.filterwarnings('ignore')

print("🚀 Starting model training pipeline...")
print("=" * 60)

# ── 1. Setup ──────────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)

# ── 2. Load data ──────────────────────────────────────────────────────────────
print("\n📊 Loading training data...")
train_df = pd.read_csv('data/train.csv')
print(f"   ✅ {len(train_df):,} samples loaded")

X_raw = train_df.drop(columns=['id', 'accident_risk'])
y     = train_df['accident_risk']

# ── 3. Feature engineering (on raw values, before encoding) ───────────────────
print("\n⚙️  Engineering interaction features...")
X = engineer_features(X_raw)
engineered_cols = ['speed_curvature', 'accident_density', 'is_night_bad_weather']
for col in engineered_cols:
    print(f"   ✓ Added '{col}'")
print(f"   Total features: {X.shape[1]} (was {X_raw.shape[1]})")

# ── 4. Encode categorical + boolean columns ────────────────────────────────────
print("\n🔄 Encoding categorical and boolean variables...")
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
label_encoders   = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ Encoded '{col}'  →  classes: {list(le.classes_)}")

feature_order = X.columns.tolist()
print(f"\n   Feature order saved ({len(feature_order)} features):")
for i, f in enumerate(feature_order, 1):
    print(f"   {i:2d}. {f}")

# ── 5. Honest evaluation: 80 / 20 train / validation split ───────────────────
print("\n📐 Splitting data (80 % train / 20 % validation)...")
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training:   {len(X_tr):,} samples")
print(f"   Validation: {len(X_val):,} samples")

# ── 6. Define model (well-regularised Random Forest) ─────────────────────────
model_params = dict(
    n_estimators   = 200,
    max_depth      = 12,
    min_samples_split = 20,
    min_samples_leaf  = 10,
    max_features   = 'sqrt',
    random_state   = 42,
    n_jobs         = -1,
)

# ── 7. Cross-validation (5-fold, on training split only) ─────────────────────
print("\n🔁 Running 5-fold cross-validation...")
cv_model = RandomForestRegressor(**model_params, verbose=0)
cv_scores = cross_val_score(cv_model, X_tr, y_tr, cv=5, scoring='r2', n_jobs=-1)
print(f"   CV R² scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean CV R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 8. Train evaluation model on 80 % split, evaluate on 20 % ────────────────
print("\n🌲 Training evaluation model on 80% split...")
eval_model = RandomForestRegressor(**model_params, verbose=0)
eval_model.fit(X_tr, y_tr)

val_preds  = eval_model.predict(X_val)
val_r2     = r2_score(y_val, val_preds)
val_mae    = mean_absolute_error(y_val, val_preds)
val_rmse   = float(np.sqrt(mean_squared_error(y_val, val_preds)))

print(f"\n   ✅ Honest Validation Metrics (on held-out 20%):")
print(f"      R² Score:  {val_r2:.4f}  ({val_r2*100:.1f}%)")
print(f"      MAE:       {val_mae:.6f}")
print(f"      RMSE:      {val_rmse:.6f}")

# ── 9. Train FINAL model on 100% of data (for production) ────────────────────
print("\n🌲 Training final model on 100% of data...")
model = RandomForestRegressor(**model_params, verbose=1)
model.fit(X, y)
print("\n   ✅ Final model trained!")

# ── 10. Feature importance ────────────────────────────────────────────────────
print("\n🎯 Feature importance (top 10):")
fi_df = pd.DataFrame({
    'feature': feature_order,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)

for _, row in fi_df.head(10).iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"   {row['feature']:30s} {row['importance']:.4f}  {bar}")

# ── 11. Save artefacts ────────────────────────────────────────────────────────
print("\n💾 Saving model artefacts...")
joblib.dump(model,         'model/accident_risk_model.pkl')
joblib.dump(label_encoders,'model/label_encoders.pkl')
joblib.dump(feature_order, 'model/feature_order.pkl')
print("   ✅ model/accident_risk_model.pkl")
print("   ✅ model/label_encoders.pkl")
print("   ✅ model/feature_order.pkl")

# ── 12. Save metadata (honest metrics) ───────────────────────────────────────
metadata = {
    'algorithm':        'Random Forest Regressor',
    'n_estimators':     model_params['n_estimators'],
    'max_depth':        model_params['max_depth'],
    'max_features':     model_params['max_features'],
    'trained_at':       str(date.today()),
    'training_samples': len(X),
    'feature_count':    len(feature_order),
    'feature_order':    feature_order,
    'engineered_features': engineered_cols,
    # Honest validation metrics
    'val_r2':           round(val_r2, 6),
    'val_mae':          round(val_mae, 6),
    'val_rmse':         round(val_rmse, 6),
    'cv_r2_mean':       round(float(cv_scores.mean()), 6),
    'cv_r2_std':        round(float(cv_scores.std()), 6),
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("   ✅ model/metadata.json")

# ── 13. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"\n  Algorithm:         {metadata['algorithm']}")
print(f"  Training samples:  {metadata['training_samples']:,}")
print(f"  Features:          {metadata['feature_count']}")
print(f"  CV R² (5-fold):    {metadata['cv_r2_mean']:.4f} ± {metadata['cv_r2_std']:.4f}")
print(f"  Val R²  (20% set): {metadata['val_r2']:.4f}  ({metadata['val_r2']*100:.1f}%)")
print(f"  Val MAE:           {metadata['val_mae']:.6f}")
print(f"  Val RMSE:          {metadata['val_rmse']:.6f}")
print("\n🚀 Run:  streamlit run streamlit_app.py")
print("=" * 60)
