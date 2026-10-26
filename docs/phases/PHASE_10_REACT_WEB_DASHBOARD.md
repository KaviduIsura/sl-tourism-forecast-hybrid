# Phase 10: React Web Dashboard & Policy Observatory Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Frontend Directory:** [`frontend/`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/frontend/)  
**Live URL (Local):** `http://localhost:5173/`  
**Core Components:**  
- Master Application: [`frontend/src/App.jsx`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/frontend/src/App.jsx)  
- Ceylon Tourism Stylesheet: [`frontend/src/App.css`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/frontend/src/App.css)  
- Design Tokens & Utilities: [`frontend/src/index.css`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/frontend/src/index.css)  
- Pre-Cached Telemetry: [`frontend/src/data/staticData.js`](file:///c:/Users/KAVIDU%20ISURA/Desktop/Tourist%20Arrival%20Focasting/sl-tourism-forecast-hybrid/frontend/src/data/staticData.js)  
- Hero Visual: Sigiriya sunrise banner at `frontend/public/assets/sl_tourism_hero.jpg`  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Design Philosophy

To satisfy both academic rigor and real-world institutional relevance, the web dashboard was designed specifically to reflect the official brand identity of the **Sri Lanka Tourism Development Authority (SLTDA)** and the world-renowned *"So Sri Lanka • You'll Come Back For More"* campaign.

The visual language harmonizes:
- **Ceylon Gold & Saffron Amber (`#E5A93C`, `#FCD34D`)**: Honoring the ancient cultural triangle, Sigiriya fortress, and golden sunrises.
- **Deep Ocean Sapphire & Slate (`#070D1E`, `#0F2C59`)**: Evoking the Indian Ocean coastlines.
- **Emerald Green (`#059669`, `#34D399`)**: Representing the lush Central Highland tea country and positive stability.
- **Crisis Crimson Alert (`#E11D48`, `#FB7185`)**: Providing distinct visual alerts during exogenous shock disruptions.

---

## 2. Dashboard Components

### 2.1 Hero Navigation & Real-Time Telemetry Bar
- Dynamic live probe connecting to the FastAPI backend (`http://localhost:8000/`) with zero-latency fallback to pre-cached canonical telemetry.
- Regal branding with Lotus motif and official research accreditation.
- High-level KPI cards displaying:
  1. Next Month Predicted Target
  2. Operating Regime (`Stable Recovery` vs `Exogenous Shock Disruption`)
  3. Verified Test $R^2$ Score (**0.666**)
  4. Full Test RMSE (**36,776**)

### 2.2 Interactive Arrival Forecast & Policy What-If Simulator
- **One-Click Presets**:
  - 🌴 *Peak Winter Season* (December/January high European search intent)
  - ⚡ *2022 Crisis Re-play* (Severe currency devaluation + active shock flag)
  - 🌧️ *Monsoon Low* (Off-season low demand)
  - 🛡️ *2026 Normal Baseline*
- **Fine-Grained Policy Controls**:
  - Calendar month selector with seasonal icons (Winter Peak, Cultural Spring, Monsoon, Summer Escape).
  - USD/LKR exchange rate slider (150 to 450 LKR).
  - Google Trends search intent sliders (Hotels, Visa ETA, Flights 0–100).
  - Autoregressive momentum inputs (Lag 1, Lag 3, Lag 12, Rolling Mean 3).
  - Exogenous Shock Switch (toggles between normal dynamics and crisis disruption).
- **Real-Time Output Card**:
  - Animated glowing arrival counter.
  - Parametric 80% and 95% Confidence Intervals.
  - **Local SHAP Feature Attribution Waterfall**: Quantifies exactly how many arrivals were added or subtracted by each feature.
  - Counterfactual simulation stress-tester.

### 2.3 10-Year Historical Arrivals & Shock Timeline (SVG Chart)
- Complete monthly chronology from 2015 to 2025 ($N=132$ observations).
- Visual shock disruption shading for:
  - 2019 Easter Sunday Attacks
  - 2020–2021 COVID-19 Pandemic
  - 2022 Economic Crisis
- Interactive multi-model overlays:
  - Actual Arrivals (Blue)
  - Proposed Hybrid Ensemble (Purple)
  - SARIMA Baseline (Orange dotted)
  - LSTM Sequence Model (Green dashed)
- Interactive hover tooltips showing date, arrivals, exchange rate, and shock regime.

### 2.4 Phase 8 SHAP Explainability Visualizer
- Global mean absolute SHAP value rankings.
- Empirical policy insights explaining why hotel searches (`22,873` arrivals), visa searches (`16,909` arrivals), and annual seasonal lags (`15,792` arrivals) dominate arrival forecasting.

### 2.5 Multi-Model Research Benchmarking Table
- Cross-regime performance matrix comparing SARIMA, LSTM, and Hybrid Stacking across Full Test, Stable Regime, and Shock Regime.

---

## 3. Local Execution Guide

Both services are active:
- **FastAPI Backend**: `http://localhost:8000` (Swagger docs at `/docs`)
- **React Web Dashboard**: `http://localhost:5173/`

To start the frontend dev server independently:
```bash
cd frontend
npm run dev
```
Production build verification:
```bash
cd frontend
npm run build
```
*(Build time: 171ms, 0 errors, 0 warnings).*
