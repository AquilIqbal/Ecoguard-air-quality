"""
Complete AQI Forecasting Training Pipeline - 15 Major Cities
============================================================

This script trains and compares multiple models for 15 major Indian cities:
1. Random Forest (baseline)
2. XGBoost (improved tree-based)
3. LSTM (deep learning) - optional
4. Ensemble (combination of all)

Usage:
    python train_all_models.py

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
from feature_engineering import prepare_dataset, split_train_test_temporal
from xgboost_forecasting import XGBoostAQIForecaster

# Try importing LSTM
try:
    from lstm_forecasting import LSTMAQIForecaster
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️  LSTM not available (TensorFlow not installed)")

# Import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_prepare_data(data_path='../data/kaggle_processed.csv', sample_size=None):
    """
    Load and prepare dataset
    
    Args:
        data_path: Path to CSV file
        sample_size: Number of samples to use (None = use all)
        
    Returns:
        Prepared DataFrame
    """
    print("=" * 60)
    print("📂 LOADING DATA")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"⚠️  Dataset not found at: {data_path}")
        print("   Creating dataset first...")
        # Run data generation
        import subprocess
        try:
            subprocess.run(['python', 'fetch_india_data.py'], check=True)
        except:
            print("❌ Failed to generate data. Run manually:")
            print("   python fetch_india_data.py")
            raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Check for required columns
    required_cols = ['date', 'pm2_5', 'rspm', 'no2', 'so2', 'aqi']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    # Sample if needed (for faster training during testing)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"✅ Sampled {sample_size} rows for training")
    
    print(f"✅ Data preparation complete")
    print(f"   Shape: {df.shape}")
    if 'city' in df.columns:
        print(f"   Cities: {df['city'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest baseline model
    """
    print("\n" + "=" * 60)
    print("🌲 TRAINING RANDOM FOREST (Baseline)")
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
    
    print("\n📊 Random Forest Results:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved to models/random_forest_model.pkl")
    
    return {
        'name': 'Random Forest',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'model': model
    }


def train_xgboost(X_train, y_train, X_test, y_test, tune_hyperparams=False):
    """
    Train XGBoost model
    """
    print("\n" + "=" * 60)
    print("⚡ TRAINING XGBOOST")
    print("=" * 60)
    
    forecaster = XGBoostAQIForecaster()
    
    if tune_hyperparams:
        print("🔍 Running hyperparameter tuning (this may take 5-10 minutes)...")
        forecaster.hyperparameter_tuning(X_train, y_train, cv=3)
    else:
        forecaster.train(X_train, y_train)
    
    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    
    # Save model
    forecaster.save_model('models/xgboost_model.pkl')
    
    return {
        'name': 'XGBoost',
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2'],
        'model': forecaster
    }


def train_lstm(X_train, y_train, X_test, y_test, epochs=50):
    """
    Train LSTM model
    """
    if not LSTM_AVAILABLE:
        print("\n⚠️  Skipping LSTM (TensorFlow not installed)")
        print("   Install with: pip install tensorflow-cpu")
        return None
    
    print("\n" + "=" * 60)
    print("🧠 TRAINING LSTM (Deep Learning)")
    print("=" * 60)
    
    # Split validation set from training
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_lstm = X_train.iloc[:-val_size]
    y_train_lstm = y_train.iloc[:-val_size]
    
    forecaster = LSTMAQIForecaster(sequence_length=7)
    
    forecaster.train(
        X_train_lstm, y_train_lstm,
        X_val, y_val,
        epochs=epochs,
        batch_size=32
    )
    
    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    
    # Save model
    forecaster.save_model('models/lstm_model')
    
    return {
        'name': 'LSTM',
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2'],
        'model': forecaster
    }


def create_ensemble(models, X_test, y_test):
    """
    Create ensemble model (average predictions)
    """
    print("\n" + "=" * 60)
    print("🎯 CREATING ENSEMBLE MODEL")
    print("=" * 60)
    
    predictions = []
    
    for result in models:
        if result is None:
            continue
        
        model = result['model']
        
        if result['name'] == 'Random Forest' or result['name'] == 'XGBoost':
            y_pred = model.model.predict(X_test) if hasattr(model, 'model') else model.predict(X_test)
        elif result['name'] == 'LSTM':
            y_pred = model.predict(X_test)
            # Align with other predictions
            y_pred = np.pad(y_pred, (model.sequence_length, 0), mode='edge')[:len(X_test)]
        
        predictions.append(y_pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    # Evaluate
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\n📊 Ensemble Results (average of {len(predictions)} models):")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    
    return {
        'name': 'Ensemble',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


def compare_models(results):
    """
    Create comparison table
    """
    print("\n" + "=" * 60)
    print("📊 FINAL MODEL COMPARISON")
    print("=" * 60)
    
    # Create DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': r['name'],
            'MAE': r['MAE'],
            'RMSE': r['RMSE'],
            'R²': r['R2']
        }
        for r in results if r is not None
    ])
    
    # Sort by R²
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best model
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   MAE:  {best_model['MAE']:.2f}")
    print(f"   RMSE: {best_model['RMSE']:.2f}")
    print(f"   R²:   {best_model['R²']:.4f}")
    
    # Save comparison
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print("\n✅ Comparison saved to models/model_comparison.csv")
    
    return comparison_df


def main():
    """
    Main training pipeline
    """
    print("\n" + "🌿 " + "=" * 56 + " 🌿")
    print("   ECOGUARD - AQI FORECASTING MODEL TRAINING PIPELINE")
    print("🌿 " + "=" * 56 + " 🌿\n")
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    
    # Step 2: Feature engineering
    print("\n" + "=" * 60)
    print("🔧 FEATURE ENGINEERING")
    print("=" * 60)
    df_engineered = prepare_dataset(df, target_horizon=1)
    
    # Step 3: Train-test split
    X_train, X_test, y_train, y_test, feature_cols = split_train_test_temporal(
        df_engineered, 
        test_size=0.2
    )
    
    # Clean infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    print("✅ Cleaned infinite values from data")
    
    # Step 4: Train models
    results = []
    
    # 4a. Random Forest
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_result)
    
    # 4b. XGBoost
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test, tune_hyperparams=False)
    results.append(xgb_result)
    
    # 4c. LSTM
    lstm_result = train_lstm(X_train, y_train, X_test, y_test, epochs=50)
    if lstm_result:
        results.append(lstm_result)
    
    # 4d. Ensemble
    ensemble_result = create_ensemble(results, X_test, y_test)
    results.append(ensemble_result)
    
    # Step 5: Compare models
    comparison_df = compare_models(results)
    
    print("\n" + "🌿 " + "=" * 56 + " 🌿")
    print("   TRAINING COMPLETE! 🎉")
    print("🌿 " + "=" * 56 + " 🌿\n")
    
    print("📁 Saved files:")
    print("   - models/random_forest_model.pkl")
    print("   - models/xgboost_model.pkl")
    if LSTM_AVAILABLE:
        print("   - models/lstm_model.h5")
    print("   - models/model_comparison.csv")
    
    print("\n💡 Next steps:")
    print("   1. Check model_comparison.csv for detailed metrics")
    print("   2. Update app.py to use the best model")
    print("   3. Run the Jupyter notebook for visualizations")
    
    return results, comparison_df


if __name__ == "__main__":
    results, comparison_df = main()