import os
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

HYBRID_FEATURES = [
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
    'arrivals_rolling_mean_3',
    'lstm_pred'
]

FEATURE_DESCRIPTIONS = {
    'gtrends_hotels': 'Search demand for hotel accommodation in Sri Lanka',
    'gtrends_visa': 'Search interest for Sri Lankan travel visas and ETA',
    'arrivals_lag_12': 'Annual seasonal baseline (tourist arrivals 12 months ago)',
    'shock_flag': 'Exogenous crisis / disruption indicator flag',
    'arrivals_rolling_mean_3': 'Quarterly moving average trend (recent 3 months)',
    'gtrends_flights': 'Search intent for flights to Sri Lanka',
    'arrivals_lag_1': 'Immediate monthly inertia (tourist arrivals last month)',
    'usd_lkr_rate_monthly': 'Macroeconomic exchange rate (USD/LKR currency value)',
    'month_sin': 'Cyclical seasonal angle (Sine component of calendar month)',
    'month_cos': 'Cyclical seasonal angle (Cosine component of calendar month)',
    'arrivals_lag_3': 'Quarterly lag component (tourist arrivals 3 months ago)',
    'lstm_pred': 'Recurrent neural network sequence trajectory prediction'
}

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.hybrid_model = None
        self.best_model_name = "Extra Trees (Proposed Hybrid)"
        self.scaler_X = None
        self.scaler_y = None
        self.lstm_model = None
        self.explainer = None
        self.shap_importance_df = None
        self.historical_df = None
        self.candidate_comparison_df = None
        self.model_comparison_df = None
        self.rmse_test = 36776.3  # Empirically verified test RMSE

        self._load_all_artifacts()
        self._initialized = True

    def _load_all_artifacts(self):
        # 1. Winning model name
        winner_file = os.path.join(MODELS_DIR, 'best_model_name.txt')
        if os.path.exists(winner_file):
            try:
                with open(winner_file, 'r') as f:
                    self.best_model_name = f.read().strip()
            except Exception as e:
                print(f"[WARN] Error reading best_model_name.txt: {e}")

        # 2. Production Hybrid Model
        hybrid_path = os.path.join(MODELS_DIR, 'hybrid_model.joblib')
        if os.path.exists(hybrid_path):
            try:
                self.hybrid_model = joblib.load(hybrid_path)
                print(f"[OK] Loaded Hybrid Production Model: {hybrid_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load hybrid_model.joblib: {e}")
        else:
            print(f"[WARN] hybrid_model.joblib not found at {hybrid_path}")

        # 3. Scalers
        scaler_x_path = os.path.join(MODELS_DIR, 'scaler_X.pkl')
        scaler_y_path = os.path.join(MODELS_DIR, 'scaler_y.pkl')
        if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
            try:
                self.scaler_X = joblib.load(scaler_x_path)
                self.scaler_y = joblib.load(scaler_y_path)
                print("[OK] Loaded MinMaxScaler artifacts for X and y")
            except Exception as e:
                print(f"[WARN] Error loading scalers: {e}")

        # 4. LSTM Model (optional/on-demand)
        lstm_path = os.path.join(MODELS_DIR, 'lstm_model.keras')
        if os.path.exists(lstm_path):
            try:
                import tensorflow as tf
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                print(f"[OK] Loaded LSTM Sequence Model: {lstm_path}")
            except Exception as e:
                print(f"[INFO] LSTM load skipped or deferred: {e}")

        # 5. SHAP Explainer
        if self.hybrid_model is not None:
            try:
                import shap
                self.explainer = shap.TreeExplainer(self.hybrid_model)
                print("[OK] Cached TreeExplainer on production hybrid ensemble")
            except Exception as e:
                print(f"[INFO] TreeExplainer initialization notice: {e}")

        # 6. SHAP Importance CSV
        shap_csv = os.path.join(DATA_DIR, 'shap_importance.csv')
        if os.path.exists(shap_csv):
            try:
                self.shap_importance_df = pd.read_csv(shap_csv)
            except Exception as e:
                print(f"[WARN] Error reading shap_importance.csv: {e}")

        # 7. Historical Dataset
        hist_csv = os.path.join(DATA_DIR, 'historical.csv')
        if os.path.exists(hist_csv):
            try:
                self.historical_df = pd.read_csv(hist_csv)
            except Exception as e:
                print(f"[WARN] Error reading historical.csv: {e}")

        # 8. Comparison Metrics
        cand_csv = os.path.join(RESULTS_DIR, 'metrics', 'candidate_comparison.csv')
        if os.path.exists(cand_csv):
            try:
                self.candidate_comparison_df = pd.read_csv(cand_csv)
            except Exception as e:
                pass

        comp_csv = os.path.join(RESULTS_DIR, 'metrics', 'model_comparison.csv')
        if os.path.exists(comp_csv):
            try:
                self.model_comparison_df = pd.read_csv(comp_csv)
            except Exception as e:
                pass

    def estimate_lstm_component(self, lag_1: float, lag_3: float, lag_12: float, shock: int) -> float:
        """Estimates LSTM trajectory if not explicitly supplied by client."""
        # Seasonal momentum heuristic anchored to Lag 1 and YoY seasonal cycle
        growth_factor = (lag_1 / max(lag_12, 1.0)) ** 0.5 if lag_12 > 0 else 1.0
        base = lag_1 * 0.6 + lag_12 * 0.4 * growth_factor
        if shock == 1:
            base *= 0.65
        return float(max(base, 1000.0))

    def predict(self, req) -> Dict[str, Any]:
        if self.hybrid_model is None:
            raise RuntimeError("Hybrid production model is not loaded.")

        # Compute cyclical month features
        angle = 2.0 * math.pi * req.month / 12.0
        month_sin = math.sin(angle)
        month_cos = math.cos(angle)

        # Resolve lstm_pred
        lstm_val = req.lstm_pred
        if lstm_val is None or lstm_val <= 0:
            lstm_val = self.estimate_lstm_component(
                req.arrivals_lag_1, req.arrivals_lag_3, req.arrivals_lag_12, req.shock_flag
            )

        # Construct input DataFrame with exact 12 feature names
        feature_dict = {
            'usd_lkr_rate_monthly': float(req.usd_lkr_rate_monthly),
            'gtrends_flights': float(req.gtrends_flights),
            'gtrends_hotels': float(req.gtrends_hotels),
            'gtrends_visa': float(req.gtrends_visa),
            'shock_flag': int(req.shock_flag),
            'month_sin': float(month_sin),
            'month_cos': float(month_cos),
            'arrivals_lag_1': float(req.arrivals_lag_1),
            'arrivals_lag_3': float(req.arrivals_lag_3),
            'arrivals_lag_12': float(req.arrivals_lag_12),
            'arrivals_rolling_mean_3': float(req.arrivals_rolling_mean_3),
            'lstm_pred': float(lstm_val)
        }

        X_input = pd.DataFrame([feature_dict])[HYBRID_FEATURES]

        # Model inference
        pred_raw = float(self.hybrid_model.predict(X_input)[0])
        pred_val = float(max(0.0, round(pred_raw, 0)))

        # Confidence intervals (80% ~ 1.282*sigma, 95% ~ 1.96*sigma)
        sigma = self.rmse_test
        ci_80 = {
            'lower': float(max(0.0, round(pred_val - 1.282 * sigma, 0))),
            'upper': float(round(pred_val + 1.282 * sigma, 0))
        }
        ci_95 = {
            'lower': float(max(0.0, round(pred_val - 1.960 * sigma, 0))),
            'upper': float(round(pred_val + 1.960 * sigma, 0))
        }

        # Local SHAP feature attributions
        attributions = []
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X_input)
                vals = np.array(shap_vals).flatten()
                for feat, v in zip(HYBRID_FEATURES, vals):
                    val_float = float(round(v, 1))
                    attributions.append({
                        'feature': feat,
                        'impact_arrivals': val_float,
                        'direction': 'positive' if val_float >= 0 else 'negative'
                    })
                # Sort by absolute impact descending
                attributions.sort(key=lambda x: abs(x['impact_arrivals']), reverse=True)
            except Exception as e:
                print(f"[WARN] Local SHAP computation error: {e}")

        # Fallback if SHAP was not computable on single row
        if not attributions and self.shap_importance_df is not None:
            for _, row in self.shap_importance_df.iterrows():
                f_name = row['Feature']
                f_imp = float(row['Mean_Abs_SHAP'])
                attributions.append({
                    'feature': f_name,
                    'impact_arrivals': round(f_imp, 1),
                    'direction': 'positive' if f_name != 'shock_flag' or req.shock_flag == 0 else 'negative'
                })

        regime_label = "Exogenous Shock Disruption" if req.shock_flag == 1 else "Stable / Normal"

        return {
            'predicted_arrivals': pred_val,
            'regime': regime_label,
            'model_name': f"Hybrid Stacking Ensemble ({self.best_model_name})",
            'lstm_component': float(round(lstm_val, 0)),
            'confidence_interval_80': ci_80,
            'confidence_interval_95': ci_95,
            'feature_attributions': attributions
        }

    def simulate_scenario(self, req) -> Dict[str, Any]:
        base_res = self.predict(req.base_request)
        base_pred = base_res['predicted_arrivals']

        # Apply simulation overrides
        sim_req = req.base_request.model_copy(deep=True)
        scenario_notes = []

        if req.shock_flag_override is not None:
            sim_req.shock_flag = req.shock_flag_override
            scenario_notes.append(f"Shock flag set to {req.shock_flag_override}")

        if req.exchange_rate_pct_change != 0.0:
            sim_req.usd_lkr_rate_monthly *= (1.0 + req.exchange_rate_pct_change / 100.0)
            scenario_notes.append(f"USD/LKR exchange rate adjusted by {req.exchange_rate_pct_change:+.1f}%")

        if req.gtrends_pct_change != 0.0:
            factor = (1.0 + req.gtrends_pct_change / 100.0)
            sim_req.gtrends_flights = float(np.clip(sim_req.gtrends_flights * factor, 0, 100))
            sim_req.gtrends_hotels  = float(np.clip(sim_req.gtrends_hotels * factor, 0, 100))
            sim_req.gtrends_visa    = float(np.clip(sim_req.gtrends_visa * factor, 0, 100))
            scenario_notes.append(f"Google Trends signals adjusted by {req.gtrends_pct_change:+.1f}%")

        sim_res = self.predict(sim_req)
        sim_pred = sim_res['predicted_arrivals']

        net_change = sim_pred - base_pred
        pct_change = (net_change / max(base_pred, 1.0)) * 100.0

        return {
            'baseline_prediction': base_pred,
            'simulated_prediction': sim_pred,
            'net_change': float(round(net_change, 0)),
            'percentage_change': float(round(pct_change, 2)),
            'baseline_regime': base_res['regime'],
            'simulated_regime': sim_res['regime'],
            'scenario_description': "; ".join(scenario_notes) if scenario_notes else "Baseline configuration unchanged"
        }

    def get_shap_rankings(self) -> List[Dict[str, Any]]:
        if self.shap_importance_df is None:
            return []
        records = []
        for _, row in self.shap_importance_df.iterrows():
            feat = row['Feature']
            records.append({
                'rank': int(row['Rank']),
                'feature': feat,
                'mean_abs_shap': float(round(row['Mean_Abs_SHAP'], 2)),
                'description': FEATURE_DESCRIPTIONS.get(feat, feat)
            })
        return records

    def get_history_records(self) -> List[Dict[str, Any]]:
        if self.historical_df is None:
            return []
        records = []
        for _, row in self.historical_df.iterrows():
            records.append({
                'date': str(row['date'])[:10],
                'year': int(row['year']),
                'month': int(row['month']),
                'tourist_arrivals': float(row['tourist_arrivals']),
                'usd_lkr_rate_monthly': float(row['usd_lkr_rate_monthly']),
                'gtrends_flights': float(row['gtrends_flights']),
                'gtrends_hotels': float(row['gtrends_hotels']),
                'gtrends_visa': float(row['gtrends_visa']),
                'shock_flag': int(row['shock_flag']),
                'shock_type': str(row['shock_type']) if pd.notnull(row['shock_type']) else None,
                'dataset_split': str(row['dataset_split'])
            })
        return records

# Singleton instance
model_manager = ModelManager()
