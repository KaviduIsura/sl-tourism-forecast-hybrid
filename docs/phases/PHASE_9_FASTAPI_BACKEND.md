# Phase 9: FastAPI Production Backend Service Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Backend Directory:** [`backend/`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/backend/)  
**Core Files:**  
- API Server: [`backend/main.py`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/backend/main.py)  
- Model Manager: [`backend/model_loader.py`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/backend/model_loader.py)  
- Pydantic Schemas: [`backend/schemas.py`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/backend/schemas.py)  
- Dependencies: [`backend/requirements.txt`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/backend/requirements.txt)  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Purpose

While machine learning models in academic research frequently remain trapped in static Jupyter notebooks, high-impact applied AI requires bridging the gap between empirical experimentation and operational software engineering. 

**Phase 9** operationalizes the entire research artifact suite into a high-performance, asynchronous REST API powered by **FastAPI** and **Uvicorn**. The backend serves four primary roles:

1. **Production Inference Engine (`POST /predict`)**: Evaluates incoming 12-dimensional feature payloads, dynamically computes seasonal trigonometric coordinates and LSTM sequence momentum, and returns real-time arrival forecasts alongside 80% and 95% parametric confidence intervals.
2. **Explainable AI Integration (`GET /shap-importance` & local SHAP in `/predict`)**: Exposes cooperative game-theoretic feature attributions from Phase 8 to provide transparent explanations of model predictions.
3. **What-If Policy & Shock Simulator (`POST /simulate`)**: Provides an interactive scenario simulation endpoint where policymakers and tourism analysts can stress-test the impact of currency depreciation (USD/LKR exchange rate shifts) and tourist search collapses.
4. **Historical & Diagnostic Telemetry (`GET /history`, `GET /models/info`, `GET /metrics`)**: Feeds chronological ground truth time-series and verified three-way evaluation benchmarks directly into the Phase 10 React web dashboard.

---

## 2. API Architecture & Design Patterns

### 2.1 The Singleton Model Manager Pattern (`model_loader.py`)
To prevent memory leaks and costly disk I/O on every incoming HTTP request, `model_loader.py` implements the **Singleton Pattern**:
- All model artifacts (`models/hybrid_model.joblib`, `models/best_model_name.txt`, `models/scaler_X.pkl`, `models/scaler_y.pkl`, and `models/lstm_model.keras`) are loaded into memory exactly **once** upon application startup.
- The `shap.TreeExplainer` instance is pre-compiled and cached in RAM for low-latency ($\le 50\text{ ms}$) local feature attribution.
- In-memory data caches serve historical records (`backend/data/historical.csv`) and SHAP rankings (`backend/data/shap_importance.csv`) with zero database overhead.

### 2.2 Strict Data Validation with Pydantic v2 (`schemas.py`)
Incoming payloads are rigorously validated against domain rules:
- Calendar month strictly constrained to $1 \le \text{month} \le 12$.
- Google Trends indices constrained to $0 \le I_t \le 100$.
- Exogenous shock flags constrained to binary states $\{0, 1\}$.
- Exchange rates and arrival lags strictly validated as positive floats ($x > 0$).

---

## 3. Verified Endpoints Reference Table

| HTTP Method | Path | Request Body | Response Model | Description |
|:---:|:---|:---:|:---:|:---|
| `GET` | `/` | None | `HealthResponse` | Operational liveness check and status of all 6 loaded artifacts |
| `POST` | `/predict` | `ForecastRequest` | `ForecastResponse` | Generates arrival forecast, operating regime, CI bands, and local SHAP attributions |
| `POST` | `/simulate` | `ScenarioSimulationRequest` | `ScenarioSimulationResponse` | What-if scenario stress-testing (shock toggles, exchange rate shocks, search drops) |
| `GET` | `/history` | None | `HistoryResponse` | Canonical historical monthly records (2015 to 2025) for charting |
| `GET` | `/history/summary` | None | `HistorySummary` | Aggregate historical statistics (peak month, total volume, shock counts) |
| `GET` | `/shap-importance` | None | `ShapImportanceResponse` | Phase 8 SHAP feature rankings with descriptions |
| `GET` | `/models/info` | None | `ModelInfoResponse` | Active model architecture, 12 features, training & testing splits |
| `GET` | `/metrics` | None | `List[ModelMetricRow]` | 3-way benchmark table (SARIMA vs LSTM vs Proposed Hybrid) |

---

## 4. Verification & Testing

All endpoints were validated using FastAPI's `TestClient` across all response codes:
- `GET /` $\rightarrow$ `200 OK` (Health status verified: `hybrid_ensemble: True`, `tree_explainer: True`, `lstm: True`)
- `GET /models/info` $\rightarrow$ `200 OK` (Confirmed active model: `Hybrid Stacking Ensemble (Extra Trees)`)
- `GET /shap-importance` $\rightarrow$ `200 OK` (12 features retrieved; Rank 1: `gtrends_hotels` with 22,872.9 mean abs SHAP)
- `POST /predict` $\rightarrow$ `200 OK` (Generated forecast: 190,025 arrivals; 80% CI: `[142,878, 237,172]`)
- `POST /simulate` $\rightarrow$ `200 OK` (Simulated crisis shock with +30% USD rate & -35% Google Trends $\rightarrow$ -31.3% arrival drop)
- `GET /metrics` $\rightarrow$ `200 OK` (9 benchmark rows served)

### Local Launch Command:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.
