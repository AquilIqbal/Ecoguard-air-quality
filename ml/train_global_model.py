"""
Global Model Training - ALL 57 Indian Cities
============================================

This script trains ONE model that works for ALL Indian cities.

Approach:
- City as categorical feature (one-hot encoded)
- Model learns city-specific patterns
- Universal predictor for any city

Comparison with 15-city approach:
- More features (131 vs 74)
- Slightly longer training (~5 min extra)
- Works for ALL cities (no API fallback needed)

Author: EcoGuard Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from feature_engineering import (
    create_temporal_features,
    create_lag_features, 
    create_rolling_features,
    create_trend_features,
    create_target_variable
)
from xgboost_forecasting import XGBoostAQIForecaster

# Import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ALL 57 Indian cities
ALL_CITIES = [
    'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai',
    'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
    'Chandigarh', 'Nagpur', 'Indore', 'Bhopal', 'Patna',
    'Guwahati', 'Shillong', 'Imphal', 'Aizawl', 'Agartala',
    'Gangtok', 'Dehradun', 'Shimla', 'Srinagar', 'Jammu',
    'Amritsar', 'Ludhiana', 'Jalandhar', 'Noida', 'Gurgaon',
    'Faridabad', 'Ghaziabad', 'Meerut', 'Bareilly', 'Aligarh',
    'Moradabad', 'Varanasi', 'Prayagraj', 'Gorakhpur', 'Raipur',
    'Bilaspur', 'Durg', 'Bhilai', 'Surat', 'Vadodara',
    'Rajkot', 'Bhavnagar', 'Jamnagar', 'Udaipur', 'Jodhpur',
    'Kota', 'Ajmer', 'Coimbatore', 'Cochin', 'Ranchi',
    'Bhubaneswar', 'Kanpur'
]


def encode_cities(df, city_column='city'):
    """
    One-hot encode cities as features
    
    Args:
        df: DataFrame with city column
        city_column: Name of city column
        
    Returns:
        DataFrame with city dummy variables
    """
    print("  🏙️  Encoding cities as features...")
    
    # Create dummy variables
    city_dummies = pd.get_dummies(df[city_column], prefix='city')
    
    # Concatenate with original dataframe
    df_encoded = pd.concat([df, city_dummies], axis=1)
    
    print(f"     Added {len(city_dummies.columns)} city features")
    
    return df_encoded


def prepare_global_dataset(df, pollutants=['pm2_5', 'rspm', 'no2', 'so2']):
    """
    Prepare dataset with city encoding for global model
    
    Args:
        df: Raw DataFrame
        pollutants: List of pollutant columns
        
    Returns:
        Fully engineered DataFrame
    """
    print("\n🔧 GLOBAL MODEL - FEATURE ENGINEERING")
    print("=" * 60)
    
    # Step 1: Temporal features
    print("  ✓ Creating temporal features...")
    df = create_temporal_features(df)
    
    # Step 2: Lag features
    print("  ✓ Creating lag features (1, 3, 7 days)...")
    df = create_lag_features(df, pollutants=pollutants)
    
    # Step 3: Rolling statistics
    print("  ✓ Creating rolling statistics (7, 30 days)...")
    df = create_rolling_features(df, pollutants=pollutants)
    
    # Step 4: Trend features
    print("  ✓ Creating trend indicators...")
    df = create_trend_features(df, pollutants=pollutants)
    
    # Step 5: City encoding ⭐ NEW
    df = encode_cities(df)
    
    # Step 6: Target variable
    print("  ✓ Creating target (predict 1 day ahead)...")
    df = create_target_variable(df, horizon=1)
    
    # Drop NaN rows
    initial_rows = len(df)
    df = df.dropna()
    print(f"  ✓ Dropped {initial_rows - len(df)} rows with missing values")
    
    # Clean infinite values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    print("  ✓ Cleaned infinite values")
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Shape: {df.shape}")
    print(f"   Total features: {len(df.columns)}")
    print(f"   City features: {len([c for c in df.columns if c.startswith('city_')])}")
    
    return df


def get_feature_columns_global(df):
    """
    Get feature columns including city encodings
    
    Args:
        df: Engineered DataFrame
        
    Returns:
        List of feature column names
    """
    exclude = ['date', 'city', 'aqi', 'aqi_target_1d']
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


def split_train_test_temporal(df, test_size=0.2):
    """
    Temporal train-test split for global model
    
    Args:
        df: Engineered DataFrame
        test_size: Proportion for test set
        
    Returns:
        X_train, X_test, y_train, y_test, feature_cols
    """
    # Sort by date
    df = df.sort_values('date')
    
    # Get features
    feature_cols = get_feature_columns_global(df)
    target_col = 'aqi_target_1d'
    
    # Split
    split_idx = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    print(f"\n✅ Temporal split complete:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {len(feature_cols)}")
    print(f"   └─ Temporal: {len([f for f in feature_cols if not f.startswith('city_')])}")
    print(f"   └─ City encoding: {len([f for f in feature_cols if f.startswith('city_')])}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def load_data(data_path='../data/kaggle_processed.csv'):
    """Load India cities dataset"""
    print("=" * 60)
    print("📂 LOADING DATA - GLOBAL MODEL")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"⚠️  Dataset not found: {data_path}")
        print("   Run: python fetch_india_data.py")
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"   Cities: {df['city'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def train_random_forest_global(X_train, y_train, X_test, y_test):
    """Train Random Forest (global)"""
    print("\n" + "=" * 60)
    print("🌲 TRAINING RANDOM FOREST - GLOBAL MODEL")
    print("=" * 60)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Random Forest (Global) Results:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    
    # Save
    os.makedirs('models', exist_ok=True)
    with open('models/random_forest_global.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Saved: models/random_forest_global.pkl")
    
    return {
        'name': 'Random Forest (Global)',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'model': model
    }


def train_xgboost_global(X_train, y_train, X_test, y_test):
    """Train XGBoost (global)"""
    print("\n" + "=" * 60)
    print("⚡ TRAINING XGBOOST - GLOBAL MODEL")
    print("=" * 60)
    
    forecaster = XGBoostAQIForecaster()
    forecaster.train(X_train, y_train)
    
    metrics = forecaster.evaluate(X_test, y_test)
    
    forecaster.save_model('models/xgboost_global.pkl')
    
    return {
        'name': 'XGBoost (Global)',
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2'],
        'model': forecaster
    }


def create_ensemble_global(models, X_test, y_test):
    """Create ensemble (global)"""
    print("\n" + "=" * 60)
    print("🎯 CREATING ENSEMBLE - GLOBAL MODEL")
    print("=" * 60)
    
    predictions = []
    
    for result in models:
        if result is None:
            continue
        
        model = result['model']
        
        if hasattr(model, 'model'):
            y_pred = model.model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        
        predictions.append(y_pred)
    
    ensemble_pred = np.mean(predictions, axis=0)
    
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\n📊 Ensemble (Global) Results:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    
    return {
        'name': 'Ensemble (Global)',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


def compare_models(results):
    """Create comparison table"""
    print("\n" + "=" * 60)
    print("📊 GLOBAL MODEL COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['name'],
            'MAE': r['MAE'],
            'RMSE': r['RMSE'],
            'R²': r['R2']
        }
        for r in results if r is not None
    ])
    
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    best = comparison_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best['Model']}")
    print(f"   MAE:  {best['MAE']:.2f}")
    print(f"   RMSE: {best['RMSE']:.2f}")
    print(f"   R²:   {best['R²']:.4f}")
    
    comparison_df.to_csv('models/model_comparison_global.csv', index=False)
    print("\n✅ Saved: models/model_comparison_global.csv")
    
    return comparison_df


def main():
    """Main training pipeline - Global Model"""
    print("\n" + "🌍 " + "=" * 56 + " 🌍")
    print("   ECOGUARD - GLOBAL MODEL (ALL 57 CITIES)")
    print("🌍 " + "=" * 56 + " 🌍\n")
    
    # Load data
    df = load_data()
    
    # Feature engineering (with city encoding)
    df_engineered = prepare_global_dataset(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test, feature_cols = split_train_test_temporal(df_engineered)
    
    # Train models
    results = []
    
    # Random Forest
    rf_result = train_random_forest_global(X_train, y_train, X_test, y_test)
    results.append(rf_result)
    
    # XGBoost
    xgb_result = train_xgboost_global(X_train, y_train, X_test, y_test)
    results.append(xgb_result)
    
    # Ensemble
    ensemble_result = create_ensemble_global(results, X_test, y_test)
    results.append(ensemble_result)
    
    # Compare
    comparison_df = compare_models(results)
    
    print("\n" + "🌍 " + "=" * 56 + " 🌍")
    print("   GLOBAL MODEL TRAINING COMPLETE! 🎉")
    print("🌍 " + "=" * 56 + " 🌍\n")
    
    print("📁 Saved files:")
    print("   - models/random_forest_global.pkl")
    print("   - models/xgboost_global.pkl")
    print("   - models/model_comparison_global.csv")
    
    print("\n💡 This model works for ALL 57 cities!")
    print("   No API fallback needed.")
    
    return results, comparison_df


if __name__ == "__main__":
    results, comparison = main()