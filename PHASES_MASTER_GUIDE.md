# Sri Lanka Tourism Forecasting — Master Phases Guide & Progress Tracker

Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  

---

## Phase Overview & Checklist

- [x] **[Phase 0: Environment & Project Setup](#phase-0---environment--project-setup)**
- [x] **[Phase 1: Data Preparation](#phase-1---data-preparation-notebooks01_data_prepipynb)**
- [x] **[Phase 2: Exploratory Data Analysis (EDA)](#phase-2---exploratory-data-analysis-eda-notebooks02_edaipynb)**
- [x] **[Phase 3: SARIMA Baseline Model](#phase-3---sarima-baseline-model-notebooks03_sarimaipynb)**
- [x] **[Phase 4: LSTM Deep Learning Model](#phase-4---lstm-model-notebooks04_lstmipynb)**
- [x] **[Phase 5: Candidate ML Model Comparison](#phase-5---candidate-ml-model-comparison-notebooks05_candidate_ml_comparisonipynb)**
- [x] **[Phase 6: Hybrid Ensemble Model](#phase-6---hybrid-ensemble-model-notebooks06_hybrid_ensembleipynb)**
- [x] **[Phase 7: Evaluation & Statistical Testing](#phase-7---evaluation-notebooks07_evaluationipynb)**
- [x] **[Phase 8: SHAP Explainability](#phase-8---shap-explainability-notebooks08_shap_analysisipynb)**
- [ ] **[Phase 9: FastAPI Backend](#phase-9---fastapi-backend-backend)**
- [ ] **[Phase 10: React Web Dashboard](#phase-10---react-web-dashboard-frontend)**
- [ ] **[Phase 11: Mobile App (PWA - Stretch Goal)](#phase-11---mobile-app-pwa---stretch-goal)**
- [ ] **[Phase 12: Deployment & DevOps](#phase-12---deployment--devops)**
- [ ] **[Phase 13: Documentation & Final Submission](#phase-13---documentation--final-submission)**

---

## Detailed Phases Breakdown

### Phase 0 - Environment & Project Setup
- [x] **Step 0.1**: Verify Python 3.12+ and core packages (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `shap`, `fastapi`, `uvicorn`, `pmdarima`, `jupyter`).
- [x] **Step 0.2**: Ensure scripts and CLI utilities are properly on PATH.
- [x] **Step 0.3**: Confirm directory structure:
  - `data/raw/`, `data/processed/`
  - `notebooks/`
  - `models/`
  - `results/metrics/`, `results/plots/`, `results/shap/`
  - `backend/data/`
  - `frontend/`
  - `docs/`
  - `scripts/`
- [x] **Step 0.4**: Set up Git version control and verify `.gitignore` (`__pycache__/`, `*.pyc`, `.env`, `node_modules/`, `.ipynb_checkpoints/`).
- [x] **Step 0.5**: Verify Jupyter / notebook environment readiness.

---

### Phase 1 - Data Preparation [`notebooks/01_data_prep.ipynb`]
- [x] **Step 1.1 - Load Master Dataset**: Read `data/raw/Dataset.xlsx` (`sheet_name='Dataset'`). Confirm 17 columns: `date`, `year`, `month`, `tourist_arrivals`, `usd_lkr_rate_monthly`, `gtrends_flights`, `gtrends_hotels`, `gtrends_visa`, `shock_flag`, `shock_type`, `month_sin`, `month_cos`, `arrivals_lag_1`, `arrivals_lag_3`, `arrivals_lag_12`, `arrivals_rolling_mean_3`, `dataset_split`.
- [x] **Step 1.2 - Fix Date Column**: Convert Excel serial integers (base `1899-12-30`) to `datetime`, normalized to the 1st of each month (`2015-01-01` to `2025-12-01`).
- [x] **Step 1.3 - Fix Data Types**: Replace `'None'` in `shock_type` with `np.nan`. Ensure all numeric features are `float`.
- [x] **Step 1.4 - Verify Feature Engineering**: Check `arrivals_lag_1`, cyclical `month_sin`, `arrivals_rolling_mean_3`.
- [x] **Step 1.5 - Validate Shock Flags**: Audit counts for Easter Attacks, COVID-19, and Economic Crisis.
- [x] **Step 1.6 - Mark ML-Ready Rows**: Create `ml_ready` column (1 where all 12 lag/rolling features are non-null, else 0).
- [x] **Step 1.7 - Save Processed Dataset**: Export to `data/processed/tourism_dataset_final.csv` and `backend/data/historical.csv`.
- **Git Commit**: `git commit -m "Phase 1: Data preparation complete"` ✅ Done

---

### Phase 2 - Exploratory Data Analysis (EDA) [`notebooks/02_eda.ipynb`]
- [x] **Step 2.1 - Load Processed Data**: Read `data/processed/tourism_dataset_final.csv`.
- [x] **Step 2.2 - Time Series Plot with Shocks**: Matplotlib plot with `ax.axvspan()` shading $\rightarrow$ `results/plots/01_arrivals_timeseries.png`.
- [x] **Step 2.3 - Seasonal Decomposition**: `statsmodels.tsa.seasonal.seasonal_decompose(period=12, model='additive')` $\rightarrow$ `results/plots/02_seasonal_decomposition.png`.
- [x] **Step 2.4 - Correlation Heatmap**: Correlation matrix for arrivals, USD/LKR, Google Trends, lags, shocks $\rightarrow$ `results/plots/03_correlation_heatmap.png`.
- [x] **Step 2.5 - Exogenous Feature Overlay**: Dual Y-axis subplots (Arrivals vs USD/LKR & Arrivals vs Google Trends) $\rightarrow$ `results/plots/04_exogenous_overlays.png`.
- [x] **Step 2.6 - ACF & PACF Analysis**: Lags = 36 $\rightarrow$ `results/plots/05_acf_pacf.png`.
- [x] **Step 2.7 - Monthly Seasonality Boxplot**: Stable months only (`shock_flag==0`) $\rightarrow$ `results/plots/06_monthly_seasonality.png`.
- [x] **Step 2.8 - Summary & Insights**: Record findings for dissertation EDA section.
- **Git Commit**: `git commit -m "Phase 2: EDA complete, 6 plots saved"` ✅ Done

---

### Phase 3 - SARIMA Baseline Model [`notebooks/03_sarima.ipynb`]
- [x] **Step 3.1 - Prepare Split**: Train/Test split via `dataset_split` on univariate `tourist_arrivals`.
- [x] **Step 3.2 - Auto ARIMA Fitting**: `pmdarima.auto_arima(m=12, seasonal=True, stepwise=True)`.
- [x] **Step 3.3 - Out-of-Sample Forecasting**: Forecast on test period (`n_periods=len(test)`).
- [x] **Step 3.4 - Regime Metrics**: Compute RMSE, MAE, MAPE on (Full test set, Stable regime `shock_flag==0`, Shock regime `shock_flag==1`).
- [x] **Step 3.5 - Visualisation**: Plot forecast vs actual with shock bands $\rightarrow$ `results/plots/07_sarima_forecast.png`.
- [x] **Step 3.6 - Save Model**: Save to `models/sarima_model.pkl`.
- **Git Commit**: `git commit -m "Phase 3: SARIMA baseline trained and evaluated"` ✅ Done

---

### Phase 4 - LSTM Model [`notebooks/04_lstm.ipynb`]
- [x] **Step 4.1 - Feature Scaling**: Fit `MinMaxScaler` on training set only for $X$ (11 features) and $y$ (`tourist_arrivals`). Save to `models/scaler_X.pkl` and `models/scaler_y.pkl`.
- [x] **Step 4.2 - Sequence Windowing**: Build 3D sequence windows with lookback = 12 months $\rightarrow$ shape `(n_samples, 12, 11)`.
- [x] **Step 4.3 - Architecture**: `LSTM(32, activation='tanh')` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16, activation='relu')` $\rightarrow$ `Dense(1)`.
- [x] **Step 4.4 - Training**: `epochs=200`, `batch_size=16`, `validation_split=0.1`, `EarlyStopping(patience=15, restore_best_weights=True)`.
- [x] **Step 4.5 - Inverse Transform & Prediction**: Generate train and test predictions (`lstm_pred`), inverse scaled.
- [x] **Step 4.6 - Metrics**: Calculate overall, stable-only, and shock-only RMSE, MAE, MAPE.
- [x] **Step 4.7 - Save Model**: Save to `models/lstm_model.keras`.
- **Git Commit**: `git commit -m "Phase 4: LSTM model trained, scaler saved"` ✅ Done

---

### Phase 5 - Candidate ML Model Comparison [`notebooks/05_candidate_ml_comparison.ipynb`]
- [x] **Step 5.1 - Tabular Features**: Prepare 11 exogenous/lag features on `ml_ready==1`.
- [x] **Step 5.2 - Define 3 Candidates**:
  - SVR (`rbf`, $C=10, \epsilon=0.1$)
  - Random Forest (`n_estimators=300, max_depth=6, random_state=42`)
  - Gradient Boosting (`n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42`)
- [x] **Step 5.3 - Cross-Validation**: 5-fold `TimeSeriesSplit` calculating validation RMSE.
- [x] **Step 5.4 - Select Winner**: Save comparison to `results/metrics/candidate_comparison.csv` and winner name to `models/best_model_name.txt`.
- [x] **Step 5.5 - SHAP Strategy Mapping**: Document TreeExplainer vs KernelExplainer choice based on winner.
- **Git Commit**: `git commit -m "Phase 5: Candidate ML comparison complete, winner selected"` ✅ Done

---

### Phase 6 - Hybrid Ensemble Model [`notebooks/06_hybrid_ensemble.ipynb`]
- [x] **Step 6.1 - Load LSTM Predictions**: Append `lstm_pred` feature to train and test sets.
- [x] **Step 6.2 - Build 12-Feature Hybrid Matrix**: 11 features + `lstm_pred`.
- [x] **Step 6.3 - Retrain Candidate Winner**: Train baseline with stacking features.
- [x] **Step 6.4 - Hyperparameter Optimization**: `GridSearchCV` with `TimeSeriesSplit(n_splits=5)`, scoring `neg_root_mean_squared_error`.
- [x] **Step 6.5 - Evaluate on Test**: Predict on test set using `best_estimator_`.
- [x] **Step 6.6 - Save Stacking Model**: Export fitted model to `models/hybrid_model.joblib`.
- **Git Commit**: `git commit -m "Phase 6: Hybrid ensemble model saved"` ✅ Done

---

### Phase 7 - Evaluation & Statistical Testing [`notebooks/07_evaluation.ipynb`]
- [x] **Step 7.1 - Aggregate Predictions**: Load ground truth, SARIMA, LSTM, and Hybrid predictions.
- [x] **Step 7.2 - Master Metrics Table**: Compute RMSE, MAE, MAPE, $R^2$ across Full, Stable, and Shock regimes $\rightarrow$ `results/metrics/model_comparison.csv`.
- [x] **Step 7.3 - Diebold-Mariano Test**: Implement DM test in scipy (SARIMA vs Hybrid & LSTM vs Hybrid on Full and Shock-only periods).
- [x] **Step 7.4 - Master Comparison Chart**: Plot Actual vs SARIMA vs LSTM vs Hybrid with shock highlights $\rightarrow$ `results/plots/08_forecast_comparison.png`.
- [x] **Step 7.5 - Residual Analysis**: Monthly error decomposition and shock error tracking $\rightarrow$ `results/plots/09_error_analysis.png`.
- **Git Commit**: `git commit -m "Phase 7: Full evaluation with DM test complete"` ✅ Done

---

### Phase 8 - SHAP Explainability [`notebooks/08_shap_analysis.ipynb`]
- [x] **Step 8.1 - Load Model & Data**: `hybrid_model.joblib` and 12 test features.
- [x] **Step 8.2 - Compute SHAP Values**: Generate explainer and compute feature contributions.
- [x] **Step 8.3 - Global Summary**: Summary beeswarm plot $\rightarrow$ `results/shap/shap_summary.png`.
- [x] **Step 8.4 - Shock Regime SHAP**: SHAP analysis for `shock_flag==1` $\rightarrow$ `results/shap/shap_shock_periods.png`.
- [x] **Step 8.5 - Stable Regime SHAP**: SHAP analysis for `shock_flag==0` $\rightarrow$ `results/shap/shap_stable_periods.png`.
- [x] **Step 8.6 - Export Feature Importance**: Save descending mean absolute SHAP values to `backend/data/shap_importance.csv`.
- **Git Commit**: `git commit -m "Phase 8: SHAP analysis complete, importance CSV saved"` ✅ Done

---

### Phase 9 - FastAPI Backend [`backend/`]
- [ ] **Step 9.1 - Requirements**: Define `backend/requirements.txt`.
- [ ] **Step 9.2 - Model Loader**: `backend/model_loader.py` with singleton cached `get_model()`.
- [ ] **Step 9.3 - Schemas**: `backend/schemas.py` with Pydantic `ForecastRequest` and `ForecastResponse`.
- [ ] **Step 9.4 - API Routes & CORS**: `backend/main.py` with endpoints:
  - `GET /` (Health check)
  - `POST /predict` (Returns forecast)
  - `GET /history` (Serves historical monthly data)
  - `GET /shap-importance` (Serves SHAP feature rankings)
- [ ] **Step 9.5 - Local Verification**: Test endpoints and Swagger UI at `/docs`.
- **Git Commit**: `git commit -m "Phase 9: FastAPI backend complete and tested"`

---

### Phase 10 - React Web Dashboard [`frontend/`]
- [ ] **Step 10.1 - Frontend Setup**: Initialize React application.
- [ ] **Step 10.2 - Dependencies**: Install `recharts` and `axios`.
- [ ] **Step 10.3 - API Service Client**: `frontend/src/api/client.js` with `.env` configuration.
- [ ] **Step 10.4 - ArrivalsChart**: Interactive line chart with shock reference zones.
- [ ] **Step 10.5 - ForecastPanel**: 12-feature input form with interactive forecasting.
- [ ] **Step 10.6 - ShapExplainer**: Horizontal bar chart of feature contributions.
- [ ] **Step 10.7 - ShockTimeline**: Historical event timeline component.
- [ ] **Step 10.8 - Dashboard Assembly**: `frontend/src/pages/Dashboard.jsx` and responsive layout.
- [ ] **Step 10.9 - Styling**: Modern UI theme, typography, and responsive CSS.
- [ ] **Step 10.10 - End-to-End Verification**: Complete full-stack integration test.
- **Git Commit**: `git commit -m "Phase 10: React dashboard complete"`

---

### Phase 11 - Mobile App (PWA)
- [ ] **Step 11.1 - PWA Configuration**: Service worker registration in `frontend/src/index.js`.
- [ ] **Step 11.2 - Manifest & Icons**: Configure `manifest.json`, theme colors, and icons.
- [ ] **Step 11.3 - Mobile Verification**: Mobile browser testing and Add-to-Home-Screen validation.
- **Git Commit**: `git commit -m "Phase 11: PWA mobile support complete"`

---

### Phase 12 - Deployment & DevOps
- [ ] **Step 12.1 - Architecture Setup**: Backend on Render, Frontend on Vercel.
- [ ] **Step 12.2 - Render Backend Deployment**: Web service configuration and model asset bundling.
- [ ] **Step 12.3 - Vercel Frontend Deployment**: Build configuration and environment variables.
- [ ] **Step 12.4 - CORS Production Lockdown**: Restrict origins to production frontend domain.
- [ ] **Step 12.5 - CD Pipeline**: Automated deployments on git push.
- [ ] **Step 12.6 - GitHub Actions CI**: Automated linting and test execution.
- [ ] **Step 12.7 - Monitoring**: Uptime monitoring setup.
- **Git Commit**: `git commit -m "Phase 12: Deployed to Render + Vercel"`

---

### Phase 13 - Documentation & Final Submission
- [ ] **Step 13.1 - Master README**: Comprehensive `README.md` with architecture, local setup guide, and live links.
- [ ] **Step 13.2 - API Documentation**: Swagger docs export and endpoint descriptions.
- [ ] **Step 13.3 - Results & Dissertation Assets**: Formatted tables and high-res plot exports.
- [ ] **Step 13.4 - Git History Audit**: Verification of clean, progressive commit logs.
- **Git Commit**: `git commit -m "Phase 13: README and final documentation"`
