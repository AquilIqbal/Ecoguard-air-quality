"""
Legacy Baseline Training Script (Random Forest)
===============================================

Purpose:
- Trains a simple Random Forest AQI regressor as an early baseline.

Input:
- ../data/india_air_quality.csv (pollutant columns: so2, no2, rspm, pm2_5)

Output:
- aqi_model.pkl (legacy artifact used for experimentation)

Notes:
- This script is separate from the current production runtime model path,
  which uses ml/model_loader.py + ml/models/xgboost_model.pkl.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "../data/india_air_quality.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1",
    low_memory=False
)

print("\nOriginal Dataset Shape:", df.shape)


# -----------------------------
# 2. Clean column names
# -----------------------------
df.columns = df.columns.str.lower().str.strip()


# -----------------------------
# 3. Keep only useful pollutant columns
# -----------------------------
columns_needed = ['so2', 'no2', 'rspm', 'pm2_5']
df = df[columns_needed]

print("\nUsing columns:", df.columns.tolist())


# -----------------------------
# 4. Convert to numeric
# -----------------------------
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# -----------------------------
# 5. Fill missing values
# -----------------------------
df = df.fillna(df.median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# -----------------------------
# 6. Create AQI (proxy formula)
# -----------------------------
df['aqi'] = (
    0.4 * df['pm2_5'] +
    0.3 * df['rspm'] +
    0.2 * df['no2'] +
    0.1 * df['so2']
)

print("\nAQI column created.")


# -----------------------------
# 7. SAMPLE DATA (performance optimization)
# -----------------------------
df_sample = df.sample(n=50000, random_state=42)

print("\nSampled dataset shape:", df_sample.shape)


# -----------------------------
# 8. Feature / Target split
# -----------------------------
X = df_sample[['pm2_5', 'rspm', 'no2', 'so2']]
y = df_sample['aqi']


# -----------------------------
# 9. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 10. Train model (optimized)
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,          # use all CPU cores
    random_state=42
)

print("\nTraining model... please wait")
model.fit(X_train, y_train)

print("\nModel trained successfully.")


# -----------------------------
# 11. Predictions
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# 12. Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("---------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# -----------------------------
# 13. Save trained model
# -----------------------------
with open("aqi_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully as ml/aqi_model.pkl")
