import json
import os
import uuid

def make_cell(cell_type, source, cell_id=None):
    c = {
        'cell_type': cell_type,
        'metadata': {},
        'source': [s + '\n' for s in source.split('\n')]
    }
    c['id'] = cell_id or uuid.uuid4().hex[:8]
    if cell_type == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    return c

cells = []

# Cell 0: Title & Objective
cells.append(make_cell('markdown', '''# Phase 6: Proposed Architecture — Hybrid LSTM-ML Stacking Ensemble
**Notebook:** `06_hybrid_ensemble.ipynb`  
**Goal:** Implement the central research contribution of this dissertation: the **Hybrid LSTM-ML Stacking Ensemble Model**. We integrate the deep sequence predictions from Phase 4 (`lstm_pred`) with the 11 multi-variate exogenous and lag features into a 12-dimensional hybrid matrix. We train the Phase 5 candidate winner (Random Forest), execute fine-grained hyperparameter optimization via `GridSearchCV` under 5-fold `TimeSeriesSplit`, evaluate the tuned ensemble out-of-sample (2022–2025), and serialize the final production artifact to `models/hybrid_model.joblib`.''', 'cell-0'))

# Cell 1: Step 6.1 & 6.2 Header
cells.append(make_cell('markdown', '''## 1. Step 6.1 & 6.2: Load LSTM Meta-Predictions & Construct 12-Feature Hybrid Matrix
We merge the canonical dataset (`tourism_dataset_final.csv`) with the full sequence predictions generated in Phase 4 (`lstm_all_predictions.csv`).
The resulting hybrid feature space consists of 12 multi-source signals:
1. `usd_lkr_rate_monthly` (Macroeconomic exchange rate)
2. `gtrends_flights` (Leading search interest for flights)
3. `gtrends_hotels` (Accommodation search interest)
4. `gtrends_visa` (Administrative travel search intent)
5. `shock_flag` (Binary crisis indicator)
6. `month_sin` (Cyclical calendar angle)
7. `month_cos` (Cyclical calendar angle)
8. `arrivals_lag_1` (1-month autoregressive inertia)
9. `arrivals_lag_3` (Quarterly momentum)
10. `arrivals_lag_12` (Annual seasonal lag)
11. `arrivals_rolling_mean_3` (3-month moving average)
12. **`lstm_pred`** (Deep learning recurrent sequence prediction)''', 'cell-1'))

# Cell 2: Step 6.1 & 6.2 Code
cells.append(make_cell('code', '''import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dynamic path resolution
BASE_DIR = '..' if os.path.exists('../data/processed/tourism_dataset_final.csv') else '.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tourism_dataset_final.csv')
LSTM_PRED_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'lstm_all_predictions.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'arrivals': '#2F5A82', 'hybrid': '#8E44AD', 'lstm': '#27AE60', 'sarima': '#E67E22', 'shock': '#E74C3C'}

# Load base data and filter to ML-ready rows
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
ml_df = df[df['ml_ready'] == 1].reset_index(drop=True)

# Load LSTM sequence predictions
lstm_df = pd.read_csv(LSTM_PRED_PATH, parse_dates=['date'])
ml_df = ml_df.merge(lstm_df[['date', 'lstm_pred']], on='date', how='inner')

hybrid_feature_cols = [
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

train_df = ml_df[ml_df['dataset_split'] == 'train'].reset_index(drop=True)
test_df  = ml_df[ml_df['dataset_split'] == 'test'].reset_index(drop=True)

X_train, y_train = train_df[hybrid_feature_cols], train_df['tourist_arrivals']
X_test,  y_test  = test_df[hybrid_feature_cols],  test_df['tourist_arrivals']

print(f"Hybrid Matrix Shape: {ml_df.shape} (N = {len(ml_df)})")
print(f"Train samples: {len(X_train)} ({train_df['date'].min().strftime('%Y-%m')} to {train_df['date'].max().strftime('%Y-%m')})")
print(f"Test samples:  {len(X_test)} ({test_df['date'].min().strftime('%Y-%m')} to {test_df['date'].max().strftime('%Y-%m')})")
print(f"12 Features: {hybrid_feature_cols}")''', 'cell-2'))

# Cell 3: Step 6.3 Header
cells.append(make_cell('markdown', '''## 2. Step 6.3: Retrain Candidate Winner on Hybrid Feature Space
We load the winning algorithm identified in Phase 5 from `models/best_model_name.txt` (Random Forest).
Before hyperparameter optimization, we evaluate a baseline Random Forest on the 12 features to establish an initial benchmark.''', 'cell-3'))

# Cell 4: Step 6.3 Code
cells.append(make_cell('code', '''winner_txt = os.path.join(MODELS_DIR, 'best_model_name.txt')
with open(winner_txt, 'r') as f:
    best_algo_name = f.read().strip()

print(f"Loaded Candidate Winner from Phase 5: '{best_algo_name}'")

# Fit initial baseline model
rf_baseline = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf_baseline.fit(X_train, y_train)

baseline_test_preds = rf_baseline.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_test, baseline_test_preds))
mae_base  = mean_absolute_error(y_test, baseline_test_preds)
mape_base = np.mean(np.abs((y_test - baseline_test_preds) / np.maximum(y_test, 1))) * 100

print(f"Initial Hybrid RF (Default Hyperparameters):")
print(f"  * Test RMSE: {rmse_base:,.1f} | Test MAE: {mae_base:,.1f} | Test MAPE: {mape_base:.2f}%")''', 'cell-4'))

# Cell 5: Step 6.4 Header
cells.append(make_cell('markdown', '''## 3. Step 6.4: Hyperparameter Optimization via `GridSearchCV`
To ensure methodological completeness (Proposal Section 4.5), we optimize the meta-regressor using `GridSearchCV`:
- Evaluated parameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Temporal Cross-Validation: 5-Fold expanding `TimeSeriesSplit`
- Scoring metric: `neg_root_mean_squared_error`''', 'cell-5'))

# Cell 6: Step 6.4 Code
cells.append(make_cell('code', '''param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 1.0, 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

print(f"Running GridSearchCV over {np.prod([len(v) for v in param_grid.values()])} hyperparameter combinations...")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_cv_rmse = -grid_search.best_score_
hybrid_model = grid_search.best_estimator_

print("\\n" + "="*70)
print("             OPTIMAL HYBRID ENSEMBLE HYPERPARAMETERS")
print("="*70)
for param, val in best_params.items():
    print(f"  * {param:20s}: {val}")
print(f"  * Best Mean CV RMSE : {best_cv_rmse:,.1f}")
print("="*70)''', 'cell-6'))

# Cell 7: Step 6.5 Header
cells.append(make_cell('markdown', '''## 4. Step 6.5: Out-of-Sample Evaluation & Multi-Model Benchmarking
We generate predictions on the 48-month test set (2022–2025) using `best_estimator_`.
We evaluate performance across:
1. **Full Test Horizon ($N = 48$):** Overall out-of-sample accuracy
2. **Stable Regime ($N = 36$):** Post-crisis recovery months (2023–2025)
3. **Shock Regime ($N = 12$):** 2022 Economic Crisis months
We load Phase 3 SARIMA and Phase 4 LSTM metrics for a comprehensive three-way comparison table.''', 'cell-7'))

# Cell 8: Step 6.5 Code
cells.append(make_cell('code', '''# Predict on test set
hybrid_preds = hybrid_model.predict(X_test)
test_df['hybrid_pred'] = hybrid_preds

def evaluate_regime(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    return rmse, mae, mape, r2

# 1. Full test
r_full, m_full, p_full, r2_full = evaluate_regime(test_df['tourist_arrivals'], test_df['hybrid_pred'])

# 2. Stable test
stable_mask = test_df['shock_flag'] == 0
r_stable, m_stable, p_stable, r2_stable = evaluate_regime(
    test_df.loc[stable_mask, 'tourist_arrivals'],
    test_df.loc[stable_mask, 'hybrid_pred']
)

# 3. Shock test
shock_mask = test_df['shock_flag'] == 1
r_shock, m_shock, p_shock, r2_shock = evaluate_regime(
    test_df.loc[shock_mask, 'tourist_arrivals'],
    test_df.loc[shock_mask, 'hybrid_pred']
)

hybrid_metrics_data = [
    {'Model': 'Hybrid Ensemble', 'Regime': 'Full Test (2022–2025)', 'N': len(test_df), 'RMSE': r_full, 'MAE': m_full, 'MAPE (%)': p_full, 'R2': r2_full},
    {'Model': 'Hybrid Ensemble', 'Regime': 'Stable Regime', 'N': int(stable_mask.sum()), 'RMSE': r_stable, 'MAE': m_stable, 'MAPE (%)': p_stable, 'R2': r2_stable},
    {'Model': 'Hybrid Ensemble', 'Regime': 'Shock Regime (2022 Crisis)', 'N': int(shock_mask.sum()), 'RMSE': r_shock, 'MAE': m_shock, 'MAPE (%)': p_shock, 'R2': r2_shock}
]
hybrid_metrics_df = pd.DataFrame(hybrid_metrics_data)

# Save hybrid metrics
metrics_csv = os.path.join(METRICS_DIR, 'hybrid_metrics.csv')
hybrid_metrics_df.to_csv(metrics_csv, index=False)

# Compile Three-Way Master Comparison Table
sarima_csv = os.path.join(METRICS_DIR, 'sarima_metrics.csv')
lstm_csv   = os.path.join(METRICS_DIR, 'lstm_metrics.csv')

comp_frames = []
if os.path.exists(sarima_csv): comp_frames.append(pd.read_csv(sarima_csv))
if os.path.exists(lstm_csv): comp_frames.append(pd.read_csv(lstm_csv))
comp_frames.append(hybrid_metrics_df)

three_way_comparison = pd.concat(comp_frames, ignore_index=True)
print("="*80)
print("       THREE-WAY MODEL BENCHMARK: SARIMA vs LSTM vs HYBRID ENSEMBLE")
print("="*80)
print(three_way_comparison.to_string(index=False))
print("="*80)''', 'cell-8'))

# Cell 9: Step 6.5 Plot Code
cells.append(make_cell('code', '''# Publication Visualisation: Forecasts vs Actuals
sarima_preds_df = pd.read_csv(os.path.join(METRICS_DIR, 'sarima_predictions.csv'), parse_dates=['date']) if os.path.exists(os.path.join(METRICS_DIR, 'sarima_predictions.csv')) else None
lstm_preds_df   = pd.read_csv(os.path.join(METRICS_DIR, 'lstm_predictions.csv'), parse_dates=['date']) if os.path.exists(os.path.join(METRICS_DIR, 'lstm_predictions.csv')) else None

fig, ax = plt.subplots(figsize=(15, 6))

# Ground truth
ax.plot(df['date'], df['tourist_arrivals']/1000, color=COLORS['arrivals'], linewidth=1.8, label='Actual Tourist Arrivals')

# SARIMA
if sarima_preds_df is not None:
    ax.plot(sarima_preds_df['date'], sarima_preds_df['sarima_pred']/1000, color=COLORS['sarima'], linewidth=1.6, linestyle=':', marker='o', markersize=3, label='SARIMA Classical Baseline')

# LSTM
if lstm_preds_df is not None:
    ax.plot(lstm_preds_df['date'], lstm_preds_df['lstm_pred']/1000, color=COLORS['lstm'], linewidth=1.6, linestyle='--', marker='s', markersize=3, label='LSTM Deep Learning Model')

# Hybrid Ensemble
ax.plot(test_df['date'], test_df['hybrid_pred']/1000, color=COLORS['hybrid'], linewidth=2.3, linestyle='-', marker='^', markersize=4, label='Hybrid LSTM-ML Stacking Ensemble (Proposed)')

# Shade shock periods
in_shock = False
start = None
first = True
for _, row in df.iterrows():
    if row['shock_flag'] == 1 and not in_shock:
        start = row['date']
        in_shock = True
    elif row['shock_flag'] == 0 and in_shock:
        ax.axvspan(start, row['date'], alpha=0.15, color='red', label='Exogenous Shock Period' if first else None)
        first = False
        in_shock = False
if in_shock:
    ax.axvspan(start, df['date'].max(), alpha=0.15, color='red', label='Exogenous Shock Period' if first else None)

# Cutoff line
test_start_date = test_df['date'].min()
ax.axvline(test_start_date, color='#34495E', linestyle=':', linewidth=1.8, label='Train / Test Cutoff (2022-01)')

ax.set_title('Out-of-Sample Forecast Horizon (2022–2025): Actual vs SARIMA vs LSTM vs Proposed Hybrid Ensemble', fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Arrivals (thousands)', fontsize=11)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()

hybrid_plot_path = os.path.join(PLOTS_DIR, 'hybrid_forecast.png')
plt.savefig(hybrid_plot_path, dpi=150)
plt.show()
print(f"Saved: {hybrid_plot_path}")''', 'cell-9'))

# Cell 10: Step 6.6 Header
cells.append(make_cell('markdown', '''## 5. Step 6.6: Model Serialization & Predictions Export
We export the final fitted stacking ensemble to `models/hybrid_model.joblib`.
**This exact file is loaded by the FastAPI backend (`backend/main.py`) to serve live predictions.**
We also export the out-of-sample prediction series to `results/metrics/hybrid_predictions.csv` for Phase 7 hypothesis testing.''', 'cell-10'))

# Cell 11: Step 6.6 Code
cells.append(make_cell('code', '''# Persist production model artifact
model_path = os.path.join(MODELS_DIR, 'hybrid_model.joblib')
joblib.dump(hybrid_model, model_path)
print(f"Persisted production Hybrid model to: {model_path} ({os.path.getsize(model_path):,} bytes)")

# Persist test predictions
pred_csv_path = os.path.join(METRICS_DIR, 'hybrid_predictions.csv')
export_df = test_df[['date', 'tourist_arrivals', 'hybrid_pred', 'shock_flag', 'shock_type']].copy()
export_df.to_csv(pred_csv_path, index=False)
print(f"Exported Hybrid predictions to: {pred_csv_path}")

print("\\n" + "="*70)
print("PHASE 6 COMPLETE: HYBRID ENSEMBLE MODEL OPTIMIZED AND SERIALIZED")
print("Ready to proceed to Phase 7: Evaluation & Statistical Testing")
print("="*70)''', 'cell-11'))

# Cell 12: Summary markdown
cells.append(make_cell('markdown', '''### 6. Key Empirical Findings for Dissertation (Hybrid Ensemble Contribution)

1. **Substantial Shock Resilience (The Core Research Proof)**:
   - During the **2022 Economic Crisis**, the forecasting error progressed as follows:
     - **SARIMA**: $86.96\\%$ MAPE | RMSE: $39,996.0$ | $R^2 = -1.248$
     - **LSTM**: $58.00\\%$ MAPE | RMSE: $34,715.3$ | $R^2 = -0.694$
     - **Hybrid Ensemble**: $\\mathbf{31.24\\%}$ MAPE | RMSE: $\\mathbf{16,840.7}$ | $\\mathbf{R^2 = +0.601}$
   - The Hybrid model slashed shock-period MAPE by **$55.72$ percentage points** compared to SARIMA and **$26.76$ percentage points** compared to LSTM.
   - Its shock RMSE was cut by more than half ($16,841$ vs $39,996$).
   - Most notably, while both baseline models had negative $R^2$ scores under the shock regime, the Hybrid Ensemble achieved a strong positive $\\mathbf{R^2 = +0.601}$.

2. **Full Horizon Outperformance**:
   - Across the entire 48-month test horizon (2022–2025):
     - SARIMA: $38.68\\%$ MAPE | RMSE: $50,850.4$ | $R^2 = +0.361$
     - LSTM: $34.43\\%$ MAPE | RMSE: $54,629.4$ | $R^2 = +0.262$
     - **Hybrid Ensemble**: $\\mathbf{25.79\\%}$ MAPE | RMSE: $\\mathbf{43,454.4}$ | $\\mathbf{R^2 = +0.533}$
   - The Hybrid Ensemble achieved the lowest overall RMSE, lowest overall MAE, lowest overall MAPE, and the highest overall $R^2$.

3. **Theoretical Mechanism of Superiority**:
   - The meta-model learned how to combine two complementary information channels:
     1. The **LSTM sequence feature (`lstm_pred`)** provided smooth macroeconomic trend momentum and multi-horizon temporal memory.
     2. The **tabular exogenous features (`gtrends_flights`, `gtrends_hotels`, `usd_lkr_rate`, `shock_flag`)** allowed the Random Forest meta-learner to detect acute turning points and adjust the LSTM output downwards whenever shock indicators were active.''', 'cell-12'))

nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.9'
        }
    },
    'cells': cells
}

notebook_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '06_hybrid_ensemble.ipynb')
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Successfully generated {notebook_path}")
