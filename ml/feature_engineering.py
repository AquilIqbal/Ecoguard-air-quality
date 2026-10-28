"""
Feature Engineering for AQI Time-Series Forecasting
==================================================

This module creates temporal features for predicting tomorrow's AQI.

Features Created:
- Temporal: day_of_week, month, hour, is_weekend
- Lag features: pollutant values from 1, 3, 7 days ago
- Rolling statistics: 7-day and 30-day averages
- Trend indicators: increasing/decreasing patterns

Author: EcoGuard Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_temporal_features(df):
    """
    Add temporal features (day, month, weekday, etc.)
    
    Args:
        df: DataFrame with 'date' column
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Ensure date is datetime
    if 'date' not in df.columns:
        print("Warning: No 'date' column found. Creating synthetic dates.")
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Cyclical encoding for better ML performance
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(df, pollutants=['pm2_5', 'rspm', 'no2', 'so2'], lags=[1, 3, 7]):
    """
    Create lag features (past values)
    
    Args:
        df: DataFrame with pollutant columns
        pollutants: List of pollutant column names
        lags: List of lag periods (e.g., [1, 3, 7] for 1, 3, 7 days ago)
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    for pollutant in pollutants:
        if pollutant not in df.columns:
            print(f"Warning: {pollutant} not found in DataFrame")
            continue
            
        for lag in lags:
            df[f'{pollutant}_lag{lag}'] = df[pollutant].shift(lag)
    
    return df


def create_rolling_features(df, pollutants=['pm2_5', 'rspm', 'no2', 'so2'], 
                           windows=[7, 30]):
    """
    Create rolling statistics (moving averages, std, etc.)
    
    Args:
        df: DataFrame with pollutant columns
        pollutants: List of pollutant column names
        windows: List of window sizes for rolling statistics
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    for pollutant in pollutants:
        if pollutant not in df.columns:
            continue
            
        for window in windows:
            # Rolling mean
            df[f'{pollutant}_roll{window}_mean'] = df[pollutant].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Rolling std (volatility)
            df[f'{pollutant}_roll{window}_std'] = df[pollutant].rolling(
                window=window, min_periods=1
            ).std()
            
            # Rolling min/max
            df[f'{pollutant}_roll{window}_min'] = df[pollutant].rolling(
                window=window, min_periods=1
            ).min()
            
            df[f'{pollutant}_roll{window}_max'] = df[pollutant].rolling(
                window=window, min_periods=1
            ).max()
    
    return df


def create_trend_features(df, pollutants=['pm2_5', 'rspm', 'no2', 'so2']):
    """
    Create trend indicators (increasing/decreasing patterns)
    
    Args:
        df: DataFrame with pollutant columns
        pollutants: List of pollutant column names
        
    Returns:
        DataFrame with trend features
    """
    df = df.copy()
    
    for pollutant in pollutants:
        if pollutant not in df.columns:
            continue
        
        # Difference from previous day
        df[f'{pollutant}_diff'] = df[pollutant].diff()
        
        # Percentage change
        df[f'{pollutant}_pct_change'] = df[pollutant].pct_change()
        
        # Trend direction (1 = up, 0 = flat, -1 = down)
        df[f'{pollutant}_trend'] = np.sign(df[f'{pollutant}_diff'])
    
    return df


def create_target_variable(df, aqi_col='aqi', horizon=1):
    """
    Create target variable (future AQI to predict)
    
    Args:
        df: DataFrame with AQI column
        aqi_col: Name of AQI column
        horizon: How many days ahead to predict (default: 1 for tomorrow)
        
    Returns:
        DataFrame with target column
    """
    df = df.copy()
    
    if aqi_col not in df.columns:
        print(f"Warning: {aqi_col} not found. Creating proxy AQI.")
        # Create proxy AQI if not exists
        df[aqi_col] = (
            0.4 * df.get('pm2_5', 0) +
            0.3 * df.get('rspm', 0) +
            0.2 * df.get('no2', 0) +
            0.1 * df.get('so2', 0)
        )
    
    # Shift AQI backwards to create future target
    df[f'aqi_target_{horizon}d'] = df[aqi_col].shift(-horizon)
    
    return df


def prepare_dataset(df, pollutants=['pm2_5', 'rspm', 'no2', 'so2'], 
                   target_horizon=1, dropna=True):
    """
    Complete feature engineering pipeline
    
    Args:
        df: Input DataFrame
        pollutants: List of pollutant columns
        target_horizon: Days ahead to predict (1 = tomorrow)
        dropna: Whether to drop rows with missing values
        
    Returns:
        Fully engineered DataFrame ready for ML
    """
    print("🔧 Starting feature engineering...")
    
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
    
    # Step 5: Target variable
    print(f"  ✓ Creating target (predict {target_horizon} day(s) ahead)...")
    df = create_target_variable(df, horizon=target_horizon)
    
    # Drop rows with NaN (due to lag/rolling features)
    if dropna:
        initial_rows = len(df)
        df = df.dropna()
        print(f"  ✓ Dropped {initial_rows - len(df)} rows with missing values")
    
    print(f"✅ Feature engineering complete! Shape: {df.shape}")
    print(f"📊 Total features: {len(df.columns)}")
    
    return df


def get_feature_columns(df, exclude=['date', 'city', 'aqi', 'aqi_target_1d']):
    """
    Get list of feature columns for ML training
    
    Args:
        df: Engineered DataFrame
        exclude: Columns to exclude from features
        
    Returns:
        List of feature column names
    """
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


def split_train_test_temporal(df, test_size=0.2, target_col='aqi_target_1d'):
    """
    Split data temporally (important for time-series!)
    
    Args:
        df: Engineered DataFrame
        test_size: Proportion for test set
        target_col: Target variable column name
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Sort by date
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Get feature columns
    feature_cols = get_feature_columns(df, exclude=['date', 'city', 'aqi', target_col])    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    print(f"✅ Temporal split complete:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols


# Example usage
if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nThis module provides functions for:")
    print("1. Temporal feature creation")
    print("2. Lag feature creation")
    print("3. Rolling statistics")
    print("4. Trend indicators")
    print("5. Complete feature engineering pipeline")
    print("\nUse in your training scripts!")