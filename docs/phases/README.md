# Sri Lanka Tourism Forecasting — Phase Documentation Index

This directory contains standalone, detailed documentation for each phase of the research project:  
**"Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks"**  
- **Student ID:** KIC-BSCCSAI-251P-034  
- **Programme:** BSc Computer Science with AI  

---

## Phase Documentation Directory

| Phase | Phase Name | Status | Documentation File | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| **Phase 0** | **Environment & Project Setup** | ✅ Completed | [PHASE_0_ENVIRONMENT_SETUP.md](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/docs/phases/PHASE_0_ENVIRONMENT_SETUP.md) | Project architecture, directory design, package dependencies, and reproducibility guidelines. |
| **Phase 1** | **Data Preparation & Feature Engineering** | ✅ Completed | [PHASE_1_DATA_PREPARATION.md](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/docs/phases/PHASE_1_DATA_PREPARATION.md) | Raw data ingestion, date normalization, shock classification, lag & cyclical features, and ML-ready splits. |
| **Phase 2** | **Exploratory Data Analysis (EDA)** | ✅ Completed | [PHASE_2_EXPLORATORY_DATA_ANALYSIS.md](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/docs/phases/PHASE_2_EXPLORATORY_DATA_ANALYSIS.md) | Deep dive into all 6 generated plots, statistical regime breakdown, seasonality proof, and dissertation talking points. |
| **Phase 3** | **SARIMA Baseline Model** | ⏳ Planned | *To be generated upon phase completion* | Classical statistical benchmark, auto-ARIMA orders, train/test evaluation, regime performance. |
| **Phase 4** | **LSTM Deep Learning Model** | ⏳ Planned | *To be generated upon phase completion* | 3D windowing, MinMaxScaler fit, LSTM architecture, loss curves, sequence predictions. |
| **Phase 5** | **Candidate ML Model Comparison** | ⏳ Planned | *To be generated upon phase completion* | SVR vs Random Forest vs Gradient Boosting under TimeSeriesSplit cross-validation. |
| **Phase 6** | **Hybrid Ensemble Model** | ⏳ Planned | *To be generated upon phase completion* | Stacking architecture, meta-regressor fitting, hyperparameter optimization via GridSearchCV. |
| **Phase 7** | **Evaluation & Statistical Testing** | ⏳ Planned | *To be generated upon phase completion* | Master metric tables (RMSE, MAE, MAPE, R²), Diebold-Mariano hypothesis testing. |
| **Phase 8** | **SHAP Explainability** | ⏳ Planned | *To be generated upon phase completion* | Feature attribution, beeswarm summary plots, regime-specific SHAP value comparisons. |
| **Phase 9** | **FastAPI Backend** | ⏳ Planned | *To be generated upon phase completion* | REST API endpoints, Pydantic data validation, cached model inference engine. |
| **Phase 10** | **React Web Dashboard** | ⏳ Planned | *To be generated upon phase completion* | Full-stack interactive dashboard, forecasting sliders, real-time shock simulations. |
| **Phase 11** | **Mobile App (PWA)** | ⏳ Planned | *To be generated upon phase completion* | Progressive web app manifest, offline service worker, responsive touch UI. |
| **Phase 12** | **Deployment & DevOps** | ⏳ Planned | *To be generated upon phase completion* | Render backend hosting, Vercel frontend hosting, CI/CD pipelines, production CORS. |
| **Phase 13** | **Final Dissertation Submission** | ⏳ Planned | *To be generated upon phase completion* | Master README, publication asset exports, viva defense preparation. |

---

## How to Use These Documents

1. **For Personal Understanding**: Read through each phase guide sequentially to understand *why* each step was performed, not just *how* the code was written.
2. **For Supervisor Meetings & Progress Reviews**: Use the tables, bullet points, and summaries in each document to present clear progress reports to your project supervisor.
3. **For Dissertation Writing**: The "Empirical Insights", "Mathematical Formulations", and "Examiner Q&A" sections are directly aligned with academic standards and can be adapted for the Methodology, Results, and Discussion chapters of your dissertation.
