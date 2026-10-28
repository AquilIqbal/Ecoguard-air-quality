# """
# LSTM Time-Series Forecasting for AQI
# =====================================

# This module implements LSTM (Long Short-Term Memory) neural network
# for predicting tomorrow's AQI.

# LSTM is chosen because:
# - Specialized for time-series data
# - Captures long-term dependencies
# - Handles sequential patterns
# - State-of-the-art for temporal prediction

# Note: Requires TensorFlow/Keras

# Author: EcoGuard Team
# Date: February 2026
# """

# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # TensorFlow imports (will be installed if needed)
# try:
#     import tensorflow as tf
#     from tensorflow import keras
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import LSTM, Dense, Dropout
#     from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#     TENSORFLOW_AVAILABLE = True
# except ImportError:
#     TENSORFLOW_AVAILABLE = False
#     print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")


# class LSTMAQI Forecaster:
#     """
#     LSTM model for AQI forecasting
#     """
    
#     def __init__(self, sequence_length=7, random_state=42):
#         """
#         Initialize LSTM forecaster
        
#         Args:
#             sequence_length: Number of past days to use (default: 7)
#             random_state: Random seed
#         """
#         if not TENSORFLOW_AVAILABLE:
#             raise ImportError("TensorFlow is required for LSTM. Install: pip install tensorflow")
        
#         self.sequence_length = sequence_length
#         self.random_state = random_state
#         self.model = None
#         self.scaler_X = StandardScaler()
#         self.scaler_y = StandardScaler()
#         self.history = None
        
#         # Set random seeds
#         np.random.seed(random_state)
#         tf.random.set_seed(random_state)
    
#     def prepare_sequences(self, X, y):
#         """
#         Prepare sequences for LSTM (3D input required)
        
#         Args:
#             X: Features (2D array)
#             y: Target (1D array)
            
#         Returns:
#             X_seq (3D), y_seq (1D)
#         """
#         X_seq, y_seq = [], []
        
#         for i in range(self.sequence_length, len(X)):
#             X_seq.append(X[i-self.sequence_length:i])
#             y_seq.append(y[i])
        
#         return np.array(X_seq), np.array(y_seq)
    
#     def create_model(self, input_shape, units=[64, 32], dropout=0.2, learning_rate=0.001):
#         """
#         Create LSTM model architecture
        
#         Args:
#             input_shape: Shape of input (sequence_length, n_features)
#             units: List of LSTM units per layer
#             dropout: Dropout rate
#             learning_rate: Learning rate for optimizer
            
#         Returns:
#             Compiled Keras model
#         """
#         model = Sequential([
#             # First LSTM layer
#             LSTM(units[0], return_sequences=True, input_shape=input_shape),
#             Dropout(dropout),
            
#             # Second LSTM layer
#             LSTM(units[1], return_sequences=False),
#             Dropout(dropout),
            
#             # Dense layers
#             Dense(32, activation='relu'),
#             Dropout(dropout),
#             Dense(16, activation='relu'),
#             Dense(1)  # Output layer
#         ])
        
#         # Compile
#         optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#         model.compile(
#             optimizer=optimizer,
#             loss='mse',
#             metrics=['mae']
#         )
        
#         return model
    
#     def train(self, X_train, y_train, X_val=None, y_val=None, 
#              epochs=100, batch_size=32, verbose=True):
#         """
#         Train LSTM model
        
#         Args:
#             X_train: Training features (2D)
#             y_train: Training target (1D)
#             X_val: Validation features (optional)
#             y_val: Validation target (optional)
#             epochs: Number of training epochs
#             batch_size: Batch size
#             verbose: Print training progress
            
#         Returns:
#             Training history
#         """
#         if verbose:
#             print("🚀 Training LSTM model...")
#             print(f"   Sequence length: {self.sequence_length} days")
        
#         # Scale features
#         X_train_scaled = self.scaler_X.fit_transform(X_train)
#         y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
#         # Prepare sequences
#         X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        
#         if verbose:
#             print(f"   Training sequences: {len(X_train_seq)}")
#             print(f"   Input shape: {X_train_seq.shape}")
        
#         # Prepare validation data if provided
#         validation_data = None
#         if X_val is not None and y_val is not None:
#             X_val_scaled = self.scaler_X.transform(X_val)
#             y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
#             X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled)
#             validation_data = (X_val_seq, y_val_seq)
            
#             if verbose:
#                 print(f"   Validation sequences: {len(X_val_seq)}")
        
#         # Create model
#         input_shape = (self.sequence_length, X_train.shape[1])
#         self.model = self.create_model(input_shape)
        
#         if verbose:
#             print(f"\n📐 Model Architecture:")
#             self.model.summary()
        
#         # Callbacks
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss' if validation_data else 'loss',
#                 patience=15,
#                 restore_best_weights=True,
#                 verbose=1 if verbose else 0
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss' if validation_data else 'loss',
#                 factor=0.5,
#                 patience=5,
#                 min_lr=0.00001,
#                 verbose=1 if verbose else 0
#             )
#         ]
        
#         # Train
#         self.history = self.model.fit(
#             X_train_seq, y_train_seq,
#             validation_data=validation_data,
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=callbacks,
#             verbose=1 if verbose else 0
#         )
        
#         if verbose:
#             print("✅ LSTM training complete!")
        
#         return self.history
    
#     def predict(self, X):
#         """
#         Make predictions
        
#         Args:
#             X: Features (2D array)
            
#         Returns:
#             Predictions (unscaled)
#         """
#         if self.model is None:
#             raise ValueError("Model not trained yet! Call train() first.")
        
#         # Scale
#         X_scaled = self.scaler_X.transform(X)
        
#         # Create sequences
#         X_seq, _ = self.prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        
#         # Predict (scaled)
#         y_pred_scaled = self.model.predict(X_seq, verbose=0)
        
#         # Unscale
#         y_pred = self.scaler_y.inverse_transform(y_pred_scaled).ravel()
        
#         return y_pred
    
#     def evaluate(self, X_test, y_test, verbose=True):
#         """
#         Evaluate model performance
        
#         Args:
#             X_test: Test features
#             y_test: Test target
#             verbose: Print metrics
            
#         Returns:
#             Dictionary of metrics
#         """
#         # Get predictions
#         y_pred = self.predict(X_test)
        
#         # Align test data (remove first sequence_length samples)
#         y_test_aligned = y_test.iloc[self.sequence_length:].values
        
#         # Calculate metrics
#         mae = mean_absolute_error(y_test_aligned, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred))
#         r2 = r2_score(y_test_aligned, y_pred)
#         mape = np.mean(np.abs((y_test_aligned - y_pred) / y_test_aligned)) * 100
        
#         metrics = {
#             'MAE': mae,
#             'RMSE': rmse,
#             'R2': r2,
#             'MAPE': mape
#         }
        
#         if verbose:
#             print("\n📊 LSTM Performance Metrics:")
#             print("=" * 40)
#             print(f"   MAE  (Mean Absolute Error):  {mae:.2f}")
#             print(f"   RMSE (Root Mean Squared):    {rmse:.2f}")
#             print(f"   R²   (R-squared):            {r2:.4f}")
#             print(f"   MAPE (Mean Abs % Error):     {mape:.2f}%")
#             print("=" * 40)
        
#         return metrics
    
#     def plot_training_history(self, figsize=(12, 5), save_path=None):
#         """
#         Plot training history (loss curves)
        
#         Args:
#             figsize: Figure size
#             save_path: Path to save plot
#         """
#         if self.history is None:
#             print("❌ No training history. Train model first!")
#             return
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
#         # Loss plot
#         ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
#         if 'val_loss' in self.history.history:
#             ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
#         ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
#         ax1.set_xlabel('Epoch', fontsize=12)
#         ax1.set_ylabel('Loss (MSE)', fontsize=12)
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # MAE plot
#         ax2.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
#         if 'val_mae' in self.history.history:
#             ax2.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
#         ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
#         ax2.set_xlabel('Epoch', fontsize=12)
#         ax2.set_ylabel('MAE', fontsize=12)
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"✅ Plot saved to {save_path}")
        
#         plt.show()
    
#     def plot_predictions(self, y_test, y_pred, figsize=(12, 6), save_path=None):
#         """
#         Plot actual vs predicted values
        
#         Args:
#             y_test: Actual values
#             y_pred: Predicted values
#             figsize: Figure size
#             save_path: Path to save plot
#         """
#         # Align test data
#         y_test_aligned = y_test.iloc[self.sequence_length:].values
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
#         # Time series plot
#         ax1.plot(y_test_aligned, label='Actual', alpha=0.7, linewidth=2)
#         ax1.plot(y_pred, label='Predicted', alpha=0.7, linewidth=2)
#         ax1.set_title('Actual vs Predicted AQI (LSTM)', fontsize=14, fontweight='bold')
#         ax1.set_xlabel('Sample Index', fontsize=12)
#         ax1.set_ylabel('AQI Value', fontsize=12)
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # Scatter plot
#         ax2.scatter(y_test_aligned, y_pred, alpha=0.5)
#         ax2.plot([y_test_aligned.min(), y_test_aligned.max()], 
#                 [y_test_aligned.min(), y_test_aligned.max()], 
#                 'r--', lw=2, label='Perfect Prediction')
#         ax2.set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
#         ax2.set_xlabel('Actual AQI', fontsize=12)
#         ax2.set_ylabel('Predicted AQI', fontsize=12)
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"✅ Plot saved to {save_path}")
        
#         plt.show()
    
#     def save_model(self, filepath='models/lstm_aqi_model'):
#         """
#         Save trained model
        
#         Args:
#             filepath: Path to save model (without extension)
#         """
#         if self.model is None:
#             print("❌ No model to save. Train first!")
#             return
        
#         # Save Keras model
#         self.model.save(f"{filepath}.h5")
        
#         # Save scalers and config
#         config = {
#             'scaler_X': self.scaler_X,
#             'scaler_y': self.scaler_y,
#             'sequence_length': self.sequence_length
#         }
        
#         with open(f"{filepath}_config.pkl", 'wb') as f:
#             pickle.dump(config, f)
        
#         print(f"✅ LSTM model saved to {filepath}.h5")
#         print(f"✅ Config saved to {filepath}_config.pkl")
    
#     @staticmethod
#     def load_model(filepath='models/lstm_aqi_model'):
#         """
#         Load trained model
        
#         Args:
#             filepath: Path to model file (without extension)
            
#         Returns:
#             LSTMAQIForecaster instance
#         """
#         if not TENSORFLOW_AVAILABLE:
#             raise ImportError("TensorFlow required to load LSTM model")
        
#         # Load config
#         with open(f"{filepath}_config.pkl", 'rb') as f:
#             config = pickle.load(f)
        
#         # Create forecaster
#         forecaster = LSTMAQIForecaster(sequence_length=config['sequence_length'])
#         forecaster.scaler_X = config['scaler_X']
#         forecaster.scaler_y = config['scaler_y']
        
#         # Load Keras model
#         forecaster.model = keras.models.load_model(f"{filepath}.h5")
        
#         print(f"✅ LSTM model loaded from {filepath}.h5")
        
#         return forecaster


# # Example usage
# if __name__ == "__main__":
#     print("LSTM AQI Forecasting Module")
#     print("=" * 50)
    
#     if TENSORFLOW_AVAILABLE:
#         print("\n✅ TensorFlow is installed!")
#         print(f"   TensorFlow version: {tf.__version__}")
#         print("\nThis module provides LSTM implementation for:")
#         print("1. Sequential time-series modeling")
#         print("2. Multi-layer LSTM architecture")
#         print("3. Automatic feature scaling")
#         print("4. Early stopping & learning rate reduction")
#         print("5. Training visualization")
#     else:
#         print("\n❌ TensorFlow not found!")
#         print("   Install with: pip install tensorflow")
#         print("   Or: pip install tensorflow-cpu (for CPU only)")