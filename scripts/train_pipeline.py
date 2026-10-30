"""
Production Retraining Pipeline for Sri Lanka Tourism Forecasting.
Automates end-to-end training of the Deep LSTM sequence model and the Hybrid Stacking Ensemble (Extra Trees).

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --epochs 100 --evaluate
"""

import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Feature Definitions
BASE_FEATURES = [
    'usd_lkr_rate_monthly',
    'gtrends_flights',
    'gtrends_hotels',
    'gtrends_visa',
    'shock_flag',
    'month_sin',
    'month_cos',
    'arrivals_lag_1',
    'arrivals_lag_3',
    'arrivals_lag_12',
    'arrivals_rolling_mean_3'
]

HYBRID_FEATURES = BASE_FEATURES + ['lstm_pred']
LOOKBACK = 24  # Lookback window identified in Phase 4 research

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM-ML Hybrid Forecasting Pipeline")
    parser.add_argument('--data-path', type=str, default='data/processed/tourism_dataset_final.csv',
                        help='Path to preprocessed ML dataset')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained model artifacts')
    parser.add_argument('--results-dir', type=str, default='results/metrics',
                        help='Directory to save output evaluation metrics')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Maximum training epochs for LSTM')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for LSTM training')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='Evaluate models on test split')
    return parser.parse_args()

def build_sequences(X_scaled: np.ndarray, y_scaled: np.ndarray, lookback: int):
    """Builds sliding 3D sequence windows (n_samples, lookback, n_features)."""
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i-lookback:i])
        y_seq.append(y_scaled[i])
    return np.array(X_seq), np.array(y_seq)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes RMSE, MAE, MAPE, and R2 metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {'rmse': round(rmse, 1), 'mae': round(mae, 1), 'mape_pct': round(mape, 2), 'r2_score': round(r2, 4)}

def run_pipeline(args):
    logger.info("Starting Production ML Retraining Pipeline...")
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load and validate data
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset not found at {args.data_path}")
        sys.exit(1)

    df = pd.read_csv(args.data_path, parse_dates=['date'])
    ml_df = df[df['ml_ready'] == 1].reset_index(drop=True)
    logger.info(f"Loaded dataset: {len(ml_df)} ML-ready monthly records ({ml_df['date'].min():%Y-%m} to {ml_df['date'].max():%Y-%m})")

    train_mask = ml_df['dataset_split'] == 'train'
    test_mask = ml_df['dataset_split'] == 'test'

    train_df = ml_df[train_mask].reset_index(drop=True)
    test_df = ml_df[test_mask].reset_index(drop=True)
    logger.info(f"Splits: Train = {len(train_df)} rows, Test = {len(test_df)} rows")

    # 2. Scaler Fitting (Training Split Only)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(train_df[BASE_FEATURES])
    scaler_y.fit(train_df[['tourist_arrivals']])

    joblib.dump(scaler_X, os.path.join(args.models_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(args.models_dir, 'scaler_y.pkl'))
    logger.info("Fitted and exported MinMaxScaler artifacts.")

    # 3. Build Sequences across Full Timeline for Unified Prediction
    X_all_scaled = scaler_X.transform(ml_df[BASE_FEATURES])
    y_all_scaled = scaler_y.transform(ml_df[['tourist_arrivals']])

    X_train_scaled = scaler_X.transform(train_df[BASE_FEATURES])
    y_train_scaled = scaler_y.transform(train_df[['tourist_arrivals']])

    X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train_scaled, LOOKBACK)
    logger.info(f"Constructed LSTM training sequences: shape {X_train_seq.shape}")

    # 4. Train Keras LSTM Neural Network
    tf.random.set_seed(42)
    np.random.seed(42)

    lstm_model = Sequential([
        LSTM(32, activation='tanh', return_sequences=False, input_shape=(LOOKBACK, len(BASE_FEATURES))),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    logger.info(f"Training LSTM model (max {args.epochs} epochs, batch_size={args.batch_size})...")
    lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    lstm_path = os.path.join(args.models_dir, 'lstm_model.keras')
    lstm_model.save(lstm_path)
    logger.info(f"Saved LSTM sequence model to {lstm_path}")

    # 5. Generate LSTM Meta-Predictions for All Rows (t >= LOOKBACK)
    all_lstm_preds = [np.nan] * len(ml_df)
    for i in range(LOOKBACK, len(ml_df)):
        seq = np.expand_dims(X_all_scaled[i-LOOKBACK:i], axis=0)
        scaled_p = lstm_model.predict(seq, verbose=0)
        unscaled_p = scaler_y.inverse_transform(scaled_p)[0, 0]
        all_lstm_preds[i] = float(max(1000.0, round(unscaled_p, 0)))

    ml_df['lstm_pred'] = all_lstm_preds

    # Backfill earliest LOOKBACK rows using seasonal inertia for training continuity
    first_valid = ml_df['lstm_pred'].dropna().iloc[0]
    ml_df['lstm_pred'] = ml_df['lstm_pred'].fillna(first_valid)

    # Save lstm predictions CSV
    lstm_preds_csv = os.path.join(args.results_dir, 'lstm_all_predictions.csv')
    ml_df[['date', 'tourist_arrivals', 'lstm_pred']].to_csv(lstm_preds_csv, index=False)
    logger.info(f"Exported full LSTM meta-predictions to {lstm_preds_csv}")

    # 6. Train Hybrid Meta-Regressor (Extra Trees)
    train_hybrid = ml_df[ml_df['dataset_split'] == 'train'].reset_index(drop=True)
    test_hybrid = ml_df[ml_df['dataset_split'] == 'test'].reset_index(drop=True)

    X_train_hyb = train_hybrid[HYBRID_FEATURES]
    y_train_hyb = train_hybrid['tourist_arrivals']
    X_test_hyb = test_hybrid[HYBRID_FEATURES]
    y_test_hyb = test_hybrid['tourist_arrivals']

    logger.info("Training Extra Trees meta-regressor on 12-feature hybrid matrix...")
    hybrid_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    hybrid_model.fit(X_train_hyb, y_train_hyb)

    hybrid_path = os.path.join(args.models_dir, 'hybrid_model.joblib')
    joblib.dump(hybrid_model, hybrid_path)
    logger.info(f"Saved Hybrid Ensemble model to {hybrid_path}")

    # 7. Evaluate on Test Split Across Regimes
    if args.evaluate:
        preds = hybrid_model.predict(X_test_hyb)
        test_hybrid['hybrid_pred'] = preds

        full_m = compute_metrics(y_test_hyb, preds)
        
        stable_sub = test_hybrid[test_hybrid['shock_flag'] == 0]
        stable_m = compute_metrics(stable_sub['tourist_arrivals'], stable_sub['hybrid_pred'])

        shock_sub = test_hybrid[test_hybrid['shock_flag'] == 1]
        shock_m = compute_metrics(shock_sub['tourist_arrivals'], shock_sub['hybrid_pred'])

        logger.info("=" * 60)
        logger.info("PRODUCTION HYBRID MODEL TEST EVALUATION:")
        logger.info(f"  * Full Test (N={len(test_hybrid)}):   RMSE = {full_m['rmse']:,.1f} | MAE = {full_m['mae']:,.1f} | MAPE = {full_m['mape_pct']}% | R2 = {full_m['r2_score']}")
        logger.info(f"  * Stable Only (N={len(stable_sub)}): RMSE = {stable_m['rmse']:,.1f} | MAE = {stable_m['mae']:,.1f} | MAPE = {stable_m['mape_pct']}% | R2 = {stable_m['r2_score']}")
        logger.info(f"  * Shock Only (N={len(shock_sub)}):  RMSE = {shock_m['rmse']:,.1f} | MAE = {shock_m['mae']:,.1f} | MAPE = {shock_m['mape_pct']}% | R2 = {shock_m['r2_score']}")
        logger.info("=" * 60)

        # Update hybrid_metrics.csv
        metrics_df = pd.DataFrame([
            {'Model': 'Hybrid Stacking Ensemble', 'Regime': 'Full Horizon', 'N_Samples': len(test_hybrid), **full_m},
            {'Model': 'Hybrid Stacking Ensemble', 'Regime': 'Stable Regime', 'N_Samples': len(stable_sub), **stable_m},
            {'Model': 'Hybrid Stacking Ensemble', 'Regime': 'Shock Regime', 'N_Samples': len(shock_sub), **shock_m}
        ])
        metrics_csv = os.path.join(args.results_dir, 'hybrid_metrics.csv')
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"Saved evaluation metrics to {metrics_csv}")

    logger.info("Pipeline execution successfully completed!")

if __name__ == '__main__':
    run_pipeline(parse_args())
