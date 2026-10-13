# Phase 0: Environment & Project Setup — Comprehensive Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Goal

Phase 0 establishes the computational foundation, package dependencies, repository layout, and reproducibility standards for the entire research project. In machine learning research—especially when combining classical econometrics (SARIMA), deep learning (LSTM), machine learning (Random Forest, Gradient Boosting, SVR), and full-stack software deployment (FastAPI & React)—having a strict, modular environment is mandatory to prevent dependency conflicts, data leakage, and untracked code modifications.

---

## 2. Directory Architecture & Purpose of Each Folder

The repository follows a professional data science standard that strictly separates raw data, processed assets, training code, exported models, and evaluation outputs:

```
sl-tourism-forecast-hybrid/
├── data/
│   ├── raw/                  <- IMMUTABLE raw files (Dataset.xlsx, CBSL exchange rates)
│   └── processed/            <- Cleaned, transformed datasets (tourism_dataset_final.csv)
├── notebooks/                <- Jupyter notebooks for iterative experimentation (Phase 1–8)
│   ├── 01_data_prep.ipynb    <- Data cleaning & feature engineering
│   └── 02_eda.ipynb          <- Exploratory data analysis & publication plots
├── models/                   <- Serialized model weights & scalers (.keras, .joblib, .pkl)
├── results/                  <- Academic outputs for dissertation & papers
│   ├── metrics/              <- Performance evaluation tables (.csv)
│   ├── plots/                <- High-resolution visualization figures (.png)
│   └── shap/                 <- Model interpretability summary figures (.png)
├── backend/                  <- Production FastAPI web application (Phase 9)
│   └── data/                 <- Historical records served to frontend (historical.csv)
├── frontend/                 <- React.js user dashboard & PWA interface (Phase 10 & 11)
├── docs/                     <- Research proposals, papers, guides, and phase documentation
│   └── phases/               <- Individual detailed phase documentation guides
├── scripts/                  <- Automation and utility scripts
├── requirements.txt          <- Pinned project Python package dependencies
└── PHASES_MASTER_GUIDE.md    <- Master checklist tracking project milestones
```

### Why This Directory Design Matters for Your Dissertation:
1. **Separation of Raw vs Processed Data (`data/raw/` vs `data/processed/`)**:
   - The raw data is treated as **read-only** (immutable). If you ever need to change a feature engineering formula, you can re-run the pipeline without worrying that your original raw records were modified or corrupted.
2. **Decoupled Experimentation and Production (`notebooks/` vs `backend/`)**:
   - Notebooks are optimized for visual validation and exploratory testing.
   - The `backend/` directory will host clean, lean Python code that loads the pre-trained `.keras` and `.joblib` model artifacts to serve predictions with sub-100ms response times.
3. **Dedicated Results Storage (`results/`)**:
   - Separates code from figures and metrics, making it easy to drag-and-drop publication assets directly into LaTeX or Microsoft Word for your final dissertation report.

---

## 3. Technology Stack & Package Dependencies Explained

All core Python packages are declared in [requirements.txt](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/requirements.txt). Here is why each specific tool was selected for this project:

| Category | Package | Version | Purpose in this Research |
| :--- | :--- | :---: | :--- |
| **Data Manipulation** | `pandas` | $\ge 2.2.0$ | Time series indexing, date parsing, lagged variables, rolling windows, and CSV/Excel I/O. |
| **Numerical Computing** | `numpy` | $\ge 1.26.0$ | Multidimensional array operations, cyclical angle transformations ($\sin/\cos$). |
| **Econometric Baseline** | `pmdarima` & `statsmodels` | $\ge 2.0.4$ / $\ge 0.14.0$ | Automated Akaike Information Criterion (AIC) search for optimal SARIMA $(p,d,q)(P,D,Q)_{12}$ orders. |
| **Classical ML** | `scikit-learn` | $\ge 1.4.0$ | Feature scaling (`MinMaxScaler`), `TimeSeriesSplit` cross-validation, and candidate models (Random Forest, Gradient Boosting, SVR). |
| **Deep Learning** | `tensorflow` | $\ge 2.16.0$ | Recurrent Neural Network (LSTM) construction, tensor windowing, Dropout regularization, EarlyStopping. |
| **Explainable AI (XAI)**| `shap` | $\ge 0.45.0$ | SHapley Additive exPlanations for global and shock-regime local feature attribution. |
| **Interactive Research** | `jupyter`, `ipykernel` | Latest | Cell-by-cell execution with embedded plots for rapid verification. |
| **Data Visualisation** | `matplotlib`, `seaborn` | $\ge 3.8.0$ / $\ge 0.13.0$ | Publication-ready 150+ DPI charts, correlation heatmaps, seasonal boxplots. |
| **Web API Backend** | `fastapi`, `uvicorn`, `pydantic` | $\ge 0.110.0$ | High-throughput asynchronous REST API for serving forecasts and SHAP explanations to the web dashboard. |

---

## 4. Git Version Control & `.gitignore` Strategy

The project utilizes Git for version tracking. The `.gitignore` file was configured to prevent committing:
- Temporary Python cache files (`__pycache__/`, `*.pyc`)
- OS-specific junk (`.DS_Store`)
- Large virtual environment directories (`venv/`, `.env`)
- Temporary notebook checkpoints (`.ipynb_checkpoints/`)
- Node.js dependencies (`node_modules/` for the frontend)

This ensures the repository remains lightweight, reproducible, and compliant with academic open-source repository standards.

---

## 5. How to Explain Phase 0 to Your Project Supervisor / Examiner

> **Sample Viva / Interview Response:**  
> *"In Phase 0, we established an enterprise-grade computational research environment. We adopted a strict folder separation pattern to isolate raw immutable records from processed feature stores and production code. We standardized all software dependencies across econometrics (`pmdarima`), machine learning (`scikit-learn`), and deep recurrent learning (`tensorflow`). This ensures that all experiments across subsequent phases are fully deterministic, auditable, and easily reproducible."*
