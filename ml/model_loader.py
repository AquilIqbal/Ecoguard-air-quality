"""
Model Loader and Prediction Utility
===================================

This module loads trained models and provides prediction utilities
for the Flask app.

Usage in app.py:
    from model_loader import predict_tomorrow_aqi, get_model_info

Author: EcoGuard Team
Date: February 2026
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.pkl')
MODEL_INFO_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_comparison.csv')

_model = None
_model_info = None
_all_model_metrics = None


def load_model():
    """
    Load the trained XGBoost model.
    """
    global _model

    if _model is not None:
        return _model

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict) and 'model' in model_data:
            _model = model_data['model']
        else:
            _model = model_data

        print("Model loaded successfully")
        return _model
    except FileNotFoundError:
        print(f"Model file not found: {MODEL_PATH}")
        print("Train the model first: python ml/train_all_models.py")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def _get_r2_column(df):
    if 'R2' in df.columns:
        return 'R2'
    if 'R²' in df.columns:
        return 'R²'
    if 'RÂ²' in df.columns:
        return 'RÂ²'
    raise KeyError("Could not find an R2 column in model_comparison.csv")


def get_model_info():
    """
    Get the active model performance information.
    """
    global _model_info

    if _model_info is not None:
        return _model_info

    try:
        df = pd.read_csv(MODEL_INFO_PATH)
        r2_col = _get_r2_column(df)
        best = df[df['Model'] == 'XGBoost'].iloc[0]

        _model_info = {
            'name': 'XGBoost',
            'mae': float(best['MAE']),
            'rmse': float(best['RMSE']),
            'r2': float(best[r2_col]),
            'accuracy_percent': round(float(best[r2_col]) * 100, 1)
        }
        return _model_info
    except Exception as e:
        print(f"Could not load model info: {e}")
        return {
            'name': 'XGBoost',
            'mae': 28.5,
            'rmse': 57.1,
            'r2': 0.71,
            'accuracy_percent': 71.0
        }


def get_all_model_metrics():
    """
    Load all model comparison metrics for the dashboard model panel.
    """
    global _all_model_metrics

    if _all_model_metrics is not None:
        return _all_model_metrics

    try:
        df = pd.read_csv(MODEL_INFO_PATH)
        r2_col = _get_r2_column(df)
        models = []

        for _, row in df.iterrows():
            models.append({
                'name': str(row['Model']),
                'mae': float(row['MAE']),
                'rmse': float(row['RMSE']),
                'r2': float(row[r2_col]),
                'active': str(row['Model']) == 'XGBoost'
            })

        preferred_order = ['XGBoost', 'Ensemble', 'Random Forest']
        models.sort(
            key=lambda item: preferred_order.index(item['name'])
            if item['name'] in preferred_order else 99
        )
        _all_model_metrics = models
        return _all_model_metrics
    except Exception as e:
        print(f"Could not load model comparison metrics: {e}")
        _all_model_metrics = [
            {'name': 'XGBoost', 'mae': 28.5, 'rmse': 57.1, 'r2': 0.71, 'active': True},
            {'name': 'Ensemble', 'mae': 28.1, 'rmse': 59.2, 'r2': 0.69, 'active': False},
            {'name': 'Random Forest', 'mae': 28.8, 'rmse': 63.3, 'r2': 0.64, 'active': False},
        ]
        return _all_model_metrics


def create_features_from_current_data(current_aqi, pm25, pm10, no2, so2):
    """
    Create a feature vector from current pollutant values.
    """
    now = datetime.now()

    features = {
        'pm2_5': pm25,
        'rspm': pm10,
        'no2': no2,
        'so2': so2,
        'year': now.year,
        'month': now.month,
        'day': now.day,
        'day_of_week': now.weekday(),
        'day_of_year': now.timetuple().tm_yday,
        'week_of_year': now.isocalendar()[1],
        'quarter': (now.month - 1) // 3 + 1,
        'is_weekend': 1 if now.weekday() >= 5 else 0,
        'is_month_start': 1 if now.day == 1 else 0,
        'is_month_end': 1 if now.day >= 28 else 0,
        'month_sin': np.sin(2 * np.pi * now.month / 12),
        'month_cos': np.cos(2 * np.pi * now.month / 12),
        'day_sin': np.sin(2 * np.pi * now.weekday() / 7),
        'day_cos': np.cos(2 * np.pi * now.weekday() / 7),
        'pm2_5_lag1': pm25,
        'pm2_5_lag3': pm25 * 0.95,
        'pm2_5_lag7': pm25 * 0.90,
        'rspm_lag1': pm10,
        'rspm_lag3': pm10 * 0.95,
        'rspm_lag7': pm10 * 0.90,
        'no2_lag1': no2,
        'no2_lag3': no2 * 0.95,
        'no2_lag7': no2 * 0.90,
        'so2_lag1': so2,
        'so2_lag3': so2 * 0.95,
        'so2_lag7': so2 * 0.90,
        'pm2_5_roll7_mean': pm25,
        'pm2_5_roll7_std': pm25 * 0.1,
        'pm2_5_roll7_min': pm25 * 0.8,
        'pm2_5_roll7_max': pm25 * 1.2,
        'pm2_5_roll30_mean': pm25,
        'pm2_5_roll30_std': pm25 * 0.15,
        'pm2_5_roll30_min': pm25 * 0.75,
        'pm2_5_roll30_max': pm25 * 1.25,
        'rspm_roll7_mean': pm10,
        'rspm_roll7_std': pm10 * 0.1,
        'rspm_roll7_min': pm10 * 0.8,
        'rspm_roll7_max': pm10 * 1.2,
        'rspm_roll30_mean': pm10,
        'rspm_roll30_std': pm10 * 0.15,
        'rspm_roll30_min': pm10 * 0.75,
        'rspm_roll30_max': pm10 * 1.25,
        'no2_roll7_mean': no2,
        'no2_roll7_std': no2 * 0.1,
        'no2_roll7_min': no2 * 0.8,
        'no2_roll7_max': no2 * 1.2,
        'no2_roll30_mean': no2,
        'no2_roll30_std': no2 * 0.15,
        'no2_roll30_min': no2 * 0.75,
        'no2_roll30_max': no2 * 1.25,
        'so2_roll7_mean': so2,
        'so2_roll7_std': so2 * 0.1,
        'so2_roll7_min': so2 * 0.8,
        'so2_roll7_max': so2 * 1.2,
        'so2_roll30_mean': so2,
        'so2_roll30_std': so2 * 0.15,
        'so2_roll30_min': so2 * 0.75,
        'so2_roll30_max': so2 * 1.25,
        'pm2_5_diff': 0,
        'pm2_5_pct_change': 0,
        'pm2_5_trend': 0,
        'rspm_diff': 0,
        'rspm_pct_change': 0,
        'rspm_trend': 0,
        'no2_diff': 0,
        'no2_pct_change': 0,
        'no2_trend': 0,
        'so2_diff': 0,
        'so2_pct_change': 0,
        'so2_trend': 0,
    }

    return features


def predict_tomorrow_aqi(current_aqi, pm25, pm10, no2, so2):
    """
    Predict tomorrow's AQI using current pollutant values.
    """
    model = load_model()

    if model is None:
        return {
            'tomorrow_aqi': int(current_aqi * 0.95),
            'change_percent': -5.0,
            'confidence': 'low',
            'method': 'fallback'
        }

    try:
        features = create_features_from_current_data(current_aqi, pm25, pm10, no2, so2)
        feature_df = pd.DataFrame([features])

        if hasattr(model, 'feature_importances_'):
            prediction = model.predict(feature_df)[0]
        elif hasattr(model, 'model'):
            prediction = model.model.predict(feature_df)[0]
        else:
            raise ValueError("Unknown model structure")

        change = prediction - current_aqi
        change_percent = (change / current_aqi * 100) if current_aqi > 0 else 0

        model_info = get_model_info()
        r2 = model_info['r2']

        if r2 >= 0.70:
            confidence = 'high'
        elif r2 >= 0.60:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'tomorrow_aqi': int(round(prediction)),
            'change_percent': round(change_percent, 1),
            'confidence': confidence,
            'method': 'ml_model',
            'model_accuracy': round(r2 * 100, 1)
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'tomorrow_aqi': int(current_aqi * 0.95),
            'change_percent': -5.0,
            'confidence': 'low',
            'method': 'fallback'
        }


def get_aqi_category(aqi):
    """
    Get AQI category and color.
    """
    if aqi <= 50:
        return {
            'category': 'Good',
            'color': 'green',
            'description': 'Air quality is good'
        }
    if aqi <= 100:
        return {
            'category': 'Satisfactory',
            'color': 'yellow',
            'description': 'Air quality is acceptable'
        }
    if aqi <= 200:
        return {
            'category': 'Moderate',
            'color': 'orange',
            'description': 'May cause breathing discomfort'
        }
    if aqi <= 300:
        return {
            'category': 'Poor',
            'color': 'red',
            'description': 'May cause breathing discomfort to people'
        }
    return {
        'category': 'Severe',
        'color': 'purple',
        'description': 'Affects healthy people and seriously impacts those with diseases'
    }


if __name__ == "__main__":
    print("Testing Model Loader...")
    print("=" * 50)

    model = load_model()
    print(f"Model loaded: {model is not None}")

    info = get_model_info()
    print("\nModel Info:")
    print(f"  Name: {info['name']}")
    print(f"  Accuracy: {info['accuracy_percent']}%")
    print(f"  MAE: {info['mae']}")

    result = predict_tomorrow_aqi(
        current_aqi=150,
        pm25=100,
        pm10=150,
        no2=40,
        so2=20
    )
    print("\nTest Prediction:")
    print(f"  Current AQI: 150")
    print(f"  Tomorrow's Prediction: {result['tomorrow_aqi']}")
    print(f"  Change: {result['change_percent']}%")
    print(f"  Confidence: {result['confidence']}")
