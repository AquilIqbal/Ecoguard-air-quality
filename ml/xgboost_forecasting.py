"""
XGBoost Time-Series Forecasting for AQI
========================================

This module implements XGBoost for predicting tomorrow's AQI.

XGBoost is chosen because:
- Excellent for tabular data
- Handles non-linear relationships
- Fast training
- Built-in feature importance
- Often outperforms Random Forest

Author: EcoGuard Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostAQIForecaster:
    """
    XGBoost model for AQI forecasting
    """
    
    def __init__(self, random_state=42):
        """
        Initialize XGBoost forecaster
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.best_params = None
        
    def create_model(self, n_estimators=200, max_depth=8, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8):
        """
        Create XGBoost model with specified parameters
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            
        Returns:
            XGBoost regressor
        """
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            objective='reg:squarederror',
            n_jobs=-1,
            verbosity=0
        )
        
        return model
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            verbose: Print training progress
            
        Returns:
            Trained model
        """
        if verbose:
            print("🚀 Training XGBoost model...")
        
        # Create model with default good parameters
        self.model = self.create_model()
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if verbose:
            print("✅ XGBoost training complete!")
            print(f"   Features used: {len(X_train.columns)}")
            print(f"   Training samples: {len(X_train)}")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, cv=3, verbose=True):
        """
        Find best hyperparameters using GridSearch
        
        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            verbose: Print progress
            
        Returns:
            Best parameters
        """
        if verbose:
            print("🔍 Starting hyperparameter tuning...")
            print("   This may take 5-10 minutes...")
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # GridSearch with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=cv)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        # Fit
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        if verbose:
            print("✅ Hyperparameter tuning complete!")
            print(f"   Best parameters: {self.best_params}")
        
        return self.best_params
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            verbose: Print metrics
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        if verbose:
            print("\n📊 XGBoost Performance Metrics:")
            print("=" * 40)
            print(f"   MAE  (Mean Absolute Error):  {mae:.2f}")
            print(f"   RMSE (Root Mean Squared):    {rmse:.2f}")
            print(f"   R²   (R-squared):            {r2:.4f}")
            print(f"   MAPE (Mean Abs % Error):     {mape:.2f}%")
            print("=" * 40)
        
        return metrics
    
    def plot_feature_importance(self, top_n=20, figsize=(10, 8), save_path=None):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        if self.feature_importance is None:
            print("❌ No feature importance available. Train model first.")
            return
        
        plt.figure(figsize=figsize)
        
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(
            data=top_features,
            y='feature',
            x='importance',
            palette='viridis'
        )
        
        plt.title(f'Top {top_n} Feature Importance - XGBoost', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_test, y_pred, figsize=(12, 6), save_path=None):
        """
        Plot actual vs predicted values
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            figsize: Figure size
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Time series plot
        ax1.plot(y_test.values, label='Actual', alpha=0.7, linewidth=2)
        ax1.plot(y_pred, label='Predicted', alpha=0.7, linewidth=2)
        ax1.set_title('Actual vs Predicted AQI', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('AQI Value', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(y_test, y_pred, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Actual AQI', fontsize=12)
        ax2.set_ylabel('Predicted AQI', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath='models/xgboost_aqi_model.pkl'):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            print("❌ No model to save. Train first!")
            return
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ XGBoost model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='models/xgboost_aqi_model.pkl'):
        """
        Load trained model from file
        
        Args:
            filepath: Path to model file
            
        Returns:
            XGBoostAQIForecaster instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        forecaster = XGBoostAQIForecaster()
        forecaster.model = model_data['model']
        forecaster.feature_importance = model_data.get('feature_importance')
        forecaster.best_params = model_data.get('best_params')
        
        print(f"✅ XGBoost model loaded from {filepath}")
        
        return forecaster


# Example usage
if __name__ == "__main__":
    print("XGBoost AQI Forecasting Module")
    print("=" * 50)
    print("\nThis module provides XGBoost implementation for:")
    print("1. Training with optimized parameters")
    print("2. Hyperparameter tuning")
    print("3. Model evaluation")
    print("4. Feature importance analysis")
    print("5. Prediction visualization")
    print("\nUse with feature_engineering.py for best results!")