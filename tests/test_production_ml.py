"""
Automated Test Suite for Production ML Serving & Inference Pipeline.
Tests:
- Model artifact loading
- Live 24-step LSTM sequence inference
- Hybrid stacking ensemble prediction
- Regime-adaptive confidence interval width expansion
- Scenario shock counterfactual simulation
- FastAPI HTTP endpoint contracts
"""

import os
import sys
import unittest
from pathlib import Path

# Add backend directory to path
backend_dir = str(Path(__file__).resolve().parent.parent / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from main import app
from model_loader import model_manager, BASE_FEATURES, HYBRID_FEATURES
from schemas import ForecastRequest, ScenarioSimulationRequest

class TestProductionMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_model_manager_artifacts_loaded(self):
        """Verify all production models, scalers, and historical buffers are in memory."""
        self.assertIsNotNone(model_manager.hybrid_model, "Hybrid Extra Trees model should be loaded")
        self.assertIsNotNone(model_manager.lstm_model, "Keras LSTM model should be loaded")
        self.assertIsNotNone(model_manager.scaler_X, "MinMaxScaler for X should be loaded")
        self.assertIsNotNone(model_manager.scaler_y, "MinMaxScaler for y should be loaded")
        self.assertIsNotNone(model_manager.historical_df, "Historical dataset should be loaded")
        self.assertGreaterEqual(len(model_manager.historical_df), 24, "History should contain >= 24 records")

    def test_02_live_lstm_sequence_inference(self):
        """Verify autonomous 24-step tensor construction and live LSTM prediction."""
        dummy_features = {
            'usd_lkr_rate_monthly': 305.0,
            'gtrends_flights': 60.0,
            'gtrends_hotels': 70.0,
            'gtrends_visa': 55.0,
            'shock_flag': 0,
            'month_sin': 0.5,
            'month_cos': 0.866,
            'arrivals_lag_1': 190000.0,
            'arrivals_lag_3': 180000.0,
            'arrivals_lag_12': 185000.0,
            'arrivals_rolling_mean_3': 185000.0
        }
        lstm_val = model_manager.compute_live_lstm_prediction(dummy_features)
        self.assertIsInstance(lstm_val, float)
        self.assertGreater(lstm_val, 10000.0, "LSTM output should be positive realistic arrival volume")

    def test_03_predict_stable_regime(self):
        """Verify forecast under Stable / Normal operating regime."""
        req = ForecastRequest(
            month=1,
            usd_lkr_rate_monthly=302.5,
            gtrends_flights=55.0,
            gtrends_hotels=68.0,
            gtrends_visa=50.0,
            shock_flag=0,
            arrivals_lag_1=205000.0,
            arrivals_lag_3=175000.0,
            arrivals_lag_12=208000.0,
            arrivals_rolling_mean_3=195000.0
        )
        res = model_manager.predict(req)

        self.assertEqual(res['regime'], "Stable / Normal")
        self.assertGreater(res['predicted_arrivals'], 10000.0)
        self.assertGreater(res['lstm_component'], 0.0)

        # Confidence interval assertions
        ci95 = res['confidence_interval_95']
        self.assertLess(ci95['lower'], res['predicted_arrivals'])
        self.assertGreater(ci95['upper'], res['predicted_arrivals'])

        # Attributions
        self.assertEqual(len(res['feature_attributions']), 12, "Should return attributions for all 12 hybrid features")

    def test_04_regime_adaptive_confidence_intervals(self):
        """Verify shock regime widens prediction interval compared to stable regime."""
        base_kwargs = dict(
            month=6,
            usd_lkr_rate_monthly=320.0,
            gtrends_flights=35.0,
            gtrends_hotels=40.0,
            gtrends_visa=30.0,
            arrivals_lag_1=110000.0,
            arrivals_lag_3=150000.0,
            arrivals_lag_12=120000.0,
            arrivals_rolling_mean_3=130000.0
        )

        stable_req = ForecastRequest(shock_flag=0, **base_kwargs)
        shock_req = ForecastRequest(shock_flag=1, **base_kwargs)

        res_stable = model_manager.predict(stable_req)
        res_shock = model_manager.predict(shock_req)

        width_stable = res_stable['confidence_interval_95']['upper'] - res_stable['confidence_interval_95']['lower']
        width_shock = res_shock['confidence_interval_95']['upper'] - res_shock['confidence_interval_95']['lower']

        self.assertGreater(
            width_shock, width_stable,
            f"Shock uncertainty interval ({width_shock:,.0f}) must be wider than stable interval ({width_stable:,.0f})"
        )

    def test_05_counterfactual_simulation(self):
        """Verify scenario shock simulation produces counterfactual delta."""
        base_req = ForecastRequest(
            month=3,
            usd_lkr_rate_monthly=300.0,
            gtrends_flights=50.0,
            gtrends_hotels=60.0,
            gtrends_visa=45.0,
            shock_flag=0,
            arrivals_lag_1=180000.0,
            arrivals_lag_3=170000.0,
            arrivals_lag_12=175000.0,
            arrivals_rolling_mean_3=175000.0
        )
        sim_req = ScenarioSimulationRequest(
            base_request=base_req,
            shock_flag_override=1,
            exchange_rate_pct_change=15.0
        )
        sim_res = model_manager.simulate_scenario(sim_req)

        self.assertIn("Shock flag set to 1", sim_res['scenario_description'])
        self.assertNotEqual(sim_res['baseline_prediction'], sim_res['simulated_prediction'])
        self.assertNotEqual(sim_res['net_change'], 0.0)

    def test_06_fastapi_endpoints(self):
        """Verify FastAPI REST API endpoints return 200 OK and valid schemas."""
        # 1. Health check
        h_res = self.client.get("/")
        self.assertEqual(h_res.status_code, 200)
        self.assertEqual(h_res.json()['status'], "healthy")

        # 2. Predict endpoint
        p_res = self.client.post("/predict", json={
            "month": 12,
            "usd_lkr_rate_monthly": 305.0,
            "gtrends_flights": 75.0,
            "gtrends_hotels": 85.0,
            "gtrends_visa": 70.0,
            "shock_flag": 0,
            "arrivals_lag_1": 215000.0,
            "arrivals_lag_3": 175000.0,
            "arrivals_lag_12": 206000.0,
            "arrivals_rolling_mean_3": 195000.0
        })
        self.assertEqual(p_res.status_code, 200)
        p_json = p_res.json()
        self.assertIn("predicted_arrivals", p_json)
        self.assertIn("lstm_component", p_json)
        self.assertIn("confidence_interval_95", p_json)

        # 3. Simulate endpoint
        s_res = self.client.post("/simulate", json={
            "base_request": {
                "month": 12,
                "usd_lkr_rate_monthly": 305.0,
                "gtrends_flights": 75.0,
                "gtrends_hotels": 85.0,
                "gtrends_visa": 70.0,
                "shock_flag": 0,
                "arrivals_lag_1": 215000.0,
                "arrivals_lag_3": 175000.0,
                "arrivals_lag_12": 206000.0,
                "arrivals_rolling_mean_3": 195000.0
            },
            "shock_flag_override": 1
        })
        self.assertEqual(s_res.status_code, 200)
        self.assertIn("simulated_prediction", s_res.json())

if __name__ == '__main__':
    unittest.main()
