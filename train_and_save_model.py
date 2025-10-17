"""
Script to train and save the accident risk prediction model
Run this before deploying the web app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting model training process...")
print("="*60)

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')
    print("✅ Created 'model' directory")

# Load data
print("\n📊 Loading training data...")
train_df = pd.read_csv('data/train.csv')
print(f"✅ Loaded {len(train_df):,} training samples")

# Separate features and target
X_train = train_df.drop(['id', 'accident_risk'], axis=1)
y_train = train_df['accident_risk']

# Identify column types
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n📝 Feature types:")
print(f"   Categorical: {len(categorical_cols)} columns")
print(f"   Numerical: {len(numerical_cols)} columns")

# Encode categorical variables
print("\n🔄 Encoding categorical variables...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ Encoded {col}")

# Train the model
print("\n🌲 Training Random Forest model...")
print("This may take a few minutes... ⏳")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)
print("\n✅ Model training complete!")

# Evaluate on training data
print("\n📊 Evaluating model performance...")
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_mae = mean_absolute_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

print(f"\n📈 Training Performance:")
print(f"   Mean Squared Error: {train_mse:.6f}")
print(f"   Mean Absolute Error: {train_mae:.6f}")
print(f"   R² Score: {train_r2:.6f}")

# Save the model and column order
print("\n💾 Saving model, encoders, and feature order...")
joblib.dump(model, 'model/accident_risk_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')
joblib.dump(X_train.columns.tolist(), 'model/feature_order.pkl') # Save the column order
print("✅ Model saved to: model/accident_risk_model.pkl")
print("✅ Encoders saved to: model/label_encoders.pkl")
print("✅ Feature order saved to: model/feature_order.pkl")

# Feature importance
print("\n🎯 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*60)
print("\n✅ Files created:")
print("   • model/accident_risk_model.pkl")
print("   • model/label_encoders.pkl")
print("   • model/feature_order.pkl")
print("\n🚀 Next steps:")
print("   1. Run the web app: streamlit run app.py")
print("   2. Open your browser to interact with the model")
print("   3. Deploy to Streamlit Cloud or other platforms")
print("\n💡 The model is now ready for deployment!")
print("="*60)
