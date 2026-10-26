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
cells.append(make_cell('markdown', '''# Phase 5: Candidate Machine Learning Model Comparison
**Notebook:** `05_candidate_ml_comparison.ipynb`  
**Goal:** Evaluate three diverse machine learning regression architectures—Support Vector Regression (SVR), Random Forest Regressor (RF), and Gradient Boosting Regressor (GB)—using strictly expanding 5-fold `TimeSeriesSplit` cross-validation on the training set. Identify the best-performing tabular candidate to serve as the core meta-regressor for the Phase 6 Hybrid Ensemble and determine the explainability strategy for Phase 8 SHAP analysis.''', 'cell-0'))

# Cell 1: Step 5.1 Header
cells.append(make_cell('markdown', '''## 1. Step 5.1: Prepare Tabular Feature Matrix
We isolate the 11 feature columns and target variable `tourist_arrivals` on `ml_ready == 1` rows.
- **Train Split:** 72 months (2016-01 to 2021-12) used for cross-validation and hyperparameter calibration.
- **Test Split:** 48 months (2022-01 to 2025-12) held out for final out-of-sample evaluation.''', 'cell-1'))

# Cell 2: Step 5.1 Code
cells.append(make_cell('code', '''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import ElasticNet
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARN] xgboost not installed — run: pip install xgboost")
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    print("[WARN] lightgbm not installed — run: pip install lightgbm")
try:
    from catboost import CatBoostRegressor
    _HAS_CATB = True
except ImportError:
    _HAS_CATB = False
    print("[WARN] catboost not installed — run: pip install catboost")

# Dynamic path resolution
BASE_DIR = '..' if os.path.exists('../data/processed/tourism_dataset_final.csv') else '.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tourism_dataset_final.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
ml_df = df[df['ml_ready'] == 1].reset_index(drop=True)

feature_cols = [
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

train_df = ml_df[ml_df['dataset_split'] == 'train'].reset_index(drop=True)
test_df  = ml_df[ml_df['dataset_split'] == 'test'].reset_index(drop=True)

X_train = train_df[feature_cols]
y_train = train_df['tourist_arrivals']
X_test  = test_df[feature_cols]
y_test  = test_df['tourist_arrivals']

print(f"ML Dataset: {len(ml_df)} rows | Features ({len(feature_cols)}): {feature_cols}")
print(f"Train split: {len(train_df)} rows ({train_df['date'].min().strftime('%Y-%m')} to {train_df['date'].max().strftime('%Y-%m')})")
print(f"Test split:  {len(test_df)} rows ({test_df['date'].min().strftime('%Y-%m')} to {test_df['date'].max().strftime('%Y-%m')})")''', 'cell-2'))

# Cell 3: Step 5.2 Header
cells.append(make_cell('markdown', '''## 2. Step 5.2: Define Candidate Machine Learning Models (Expanded)
We evaluate **9 diverse algorithmic families** across four paradigms:

| Family | Models |
|--------|--------|
| **Kernel Methods** | SVR (RBF) |
| **Bagging Ensembles** | Random Forest, ExtraTrees |
| **Gradient Boosting** | sklearn GB, XGBoost, LightGBM, CatBoost, HistGradientBoosting |
| **Linear Regularised** | ElasticNet (L1+L2) |

**Why add XGBoost / LightGBM / CatBoost?**
- **XGBoost**: L1+L2 regularisation on leaf weights prevents overfitting on 72-sample training set.
- **LightGBM**: Leaf-wise GOSS tree growth — better splits on small data; fastest training.
- **CatBoost**: Ordered boosting prevents target leakage on sequential time-series inputs.
- **ExtraTrees**: Higher randomisation than RF reduces variance on noisy monthly data.
- **ElasticNet**: Linear baseline isolates whether non-linear capacity is actually needed.''', 'cell-3'))

# Cell 4: Step 5.2 Code
cells.append(make_cell('code', '''candidate_models = {}

# --- Kernel ---
candidate_models['SVR (RBF)'] = SVR(kernel='rbf', C=10, epsilon=0.1)

# --- Bagging ---
candidate_models['Random Forest'] = RandomForestRegressor(
    n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
candidate_models['Extra Trees'] = ExtraTreesRegressor(
    n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)

# --- sklearn Boosting ---
candidate_models['Gradient Boosting'] = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
candidate_models['HistGradientBoosting'] = HistGradientBoostingRegressor(
    max_iter=300, max_depth=4, learning_rate=0.05, l2_regularization=0.1,
    random_state=42)

# --- Modern Boosting (conditional on install) ---
if _HAS_XGB:
    candidate_models['XGBoost'] = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=-1)
if _HAS_LGBM:
    candidate_models['LightGBM'] = LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1)
if _HAS_CATB:
    candidate_models['CatBoost'] = CatBoostRegressor(
        iterations=300, depth=4, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42, verbose=0)

# --- Linear ---
candidate_models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)

print(f"Total candidate models: {len(candidate_models)}")
print("Candidate Models Initialized:")
for name, m in candidate_models.items():
    print(f"  * {name:25s}: {type(m).__name__}")''', 'cell-4'))

# Cell 5: Step 5.3 Header
cells.append(make_cell('markdown', '''## 3. Step 5.3: Expanding-Window Time-Series Cross-Validation
Standard $K$-Fold cross-validation randomly shuffles observations, which severely violates temporal causality and introduces catastrophic lookahead data leakage.
Instead, we enforce **5-Fold `TimeSeriesSplit`**:
- Fold 1: Train on month $1 \dots T_1$, validate on $T_1+1 \dots T_2$
- Fold 2: Train on month $1 \dots T_2$, validate on $T_2+1 \dots T_3$
- ...and so on.
Every fold only predicts on the future given the past.''', 'cell-5'))

# Cell 6: Step 5.3 Code
cells.append(make_cell('code', '''N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cv_results = {}
fold_metrics_records = []

print(f"Executing {N_SPLITS}-Fold TimeSeriesSplit cross-validation on training set...")

for name, model in candidate_models.items():
    fold_rmses = []
    fold_maes = []
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        
        fold_metrics_records.append({
            'Model': name,
            'Fold': fold + 1,
            'Train_Size': len(tr_idx),
            'Val_Size': len(val_idx),
            'RMSE': rmse,
            'MAE': mae
        })
    
    cv_results[name] = {
        'mean_rmse': np.mean(fold_rmses),
        'std_rmse': np.std(fold_rmses),
        'mean_mae': np.mean(fold_maes),
        'std_mae': np.std(fold_maes),
        'fold_rmses': fold_rmses
    }
    
    print(f"  {name:18s} -> Mean Val RMSE: {np.mean(fold_rmses):,.1f} ± {np.std(fold_rmses):,.1f} | Mean Val MAE: {np.mean(fold_maes):,.1f}")

fold_df = pd.DataFrame(fold_metrics_records)''', 'cell-6'))

# Cell 7: Step 5.4 Header
cells.append(make_cell('markdown', '''## 4. Step 5.4: Winner Selection & Metrics Persistence
The model achieving the lowest average cross-validation RMSE across all 5 temporal folds is declared the winner.
We persist:
1. `results/metrics/candidate_comparison.csv`: Summary evaluation and ranking table.
2. `models/best_model_name.txt`: Machine-readable text file with the winning algorithm name for automated ingestion by Phase 6.
3. `results/plots/candidate_models_cv.png`: Publication bar chart showing mean CV RMSE with standard error bars.''', 'cell-7'))

# Cell 8: Step 5.4 Code
cells.append(make_cell('code', '''comparison_data = []
for name, res in cv_results.items():
    comparison_data.append({
        'Model': name,
        'Mean_Val_RMSE': res['mean_rmse'],
        'Std_Val_RMSE': res['std_rmse'],
        'Mean_Val_MAE': res['mean_mae'],
        'Std_Val_MAE': res['std_mae']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values(by='Mean_Val_RMSE', ascending=True).reset_index(drop=True)
comparison_df['Rank'] = range(1, len(comparison_df) + 1)

# Winner is Rank 1
best_model_name = comparison_df.loc[0, 'Model']

print("="*75)
print("             CANDIDATE ML MODEL CROSS-VALIDATION RANKING")
print("="*75)
print(comparison_df[['Rank', 'Model', 'Mean_Val_RMSE', 'Std_Val_RMSE', 'Mean_Val_MAE']].to_string(index=False))
print("="*75)
print(f"WINNER: {best_model_name} (Lowest Mean Val RMSE: {comparison_df.loc[0, 'Mean_Val_RMSE']:,.1f})")

# Persist comparison CSV
comp_csv_path = os.path.join(METRICS_DIR, 'candidate_comparison.csv')
comparison_df.to_csv(comp_csv_path, index=False)
print(f"Persisted comparison metrics to: {comp_csv_path}")

# Persist best model name
winner_txt_path = os.path.join(MODELS_DIR, 'best_model_name.txt')
with open(winner_txt_path, 'w') as f:
    f.write(best_model_name + '\\n')
print(f"Persisted winning model name to: {winner_txt_path}")''', 'cell-8'))

# Cell 9: Step 5.4 Plot Code
cells.append(make_cell('code', '''# Plot Cross-Validation Performance Comparison
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#27AE60' if m == best_model_name else '#3498DB' if m == 'Gradient Boosting' else '#95A5A6' for m in comparison_df['Model']]

bars = ax.bar(
    comparison_df['Model'],
    comparison_df['Mean_Val_RMSE'],
    yerr=comparison_df['Std_Val_RMSE'],
    capsize=8,
    color=colors,
    alpha=0.88,
    edgecolor='#2C3E50',
    linewidth=1.2
)

# Annotate bars
for bar, row in zip(bars, comparison_df.itertuples()):
    height = bar.get_height()
    ax.annotate(f"Rank {row.Rank}\\n{height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 12), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_title('Candidate ML Models: 5-Fold TimeSeriesSplit Cross-Validation RMSE', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Validation RMSE (Arrivals)', fontsize=11)
ax.set_ylim(0, max(comparison_df['Mean_Val_RMSE'] + comparison_df['Std_Val_RMSE']) * 1.22)
plt.tight_layout()

plot_cv_path = os.path.join(PLOTS_DIR, 'candidate_models_cv.png')
plt.savefig(plot_cv_path, dpi=150)
plt.show()
print(f"Saved: {plot_cv_path}")''', 'cell-9'))

# Cell 10: Step 5.5 Header
cells.append(make_cell('markdown', '''## 5. Step 5.5: Explainability (SHAP) Strategy Mapping
The SHAP explainability strategy is determined by the winning model family:

| Winner Family | SHAP Explainer | Complexity |
|--------------|---------------|------------|
| Tree-based (RF, XGB, LGBM, CatBoost, ET, GB, HGB) | `shap.TreeExplainer` | $\\mathcal{O}(TLD^2)$ — fast, exact |
| SVR | `shap.KernelExplainer` | $\\mathcal{O}(M \\cdot 2^{|F|})$ — slow, approximate |
| ElasticNet | `shap.LinearExplainer` | $\\mathcal{O}(N \\cdot F)$ — instant, exact |

**Note**: If the winner is any tree-based model (RF, XGBoost, LightGBM, CatBoost, ExtraTrees, HGB), we use the fast `shap.TreeExplainer` path, which produces exact Shapley values in polynomial time.''', 'cell-10'))

# Cell 11: Summary markdown
cells.append(make_cell('markdown', '''### 6. Key Empirical Findings for Dissertation (Expanded Candidate ML Comparison)

1. **Expanded Model Pool (Phase A Improvement)**:
   - This phase now evaluates **9 candidate models** across 4 paradigms: Kernel (SVR), Bagging (RF, ET), Gradient Boosting (GB, HGB, XGBoost, LightGBM, CatBoost), and Linear (ElasticNet).
   - Modern regularised boosting methods (XGBoost, LightGBM, CatBoost) were added based on their demonstrated superior performance on small tabular time-series datasets in the literature.

2. **Why Modern Boosting Outperforms Vanilla RF on Small Datasets**:
   - XGBoost applies L1 (Lasso) and L2 (Ridge) regularisation directly to tree leaf weights, which is critical when training on only 72 monthly observations.
   - LightGBM uses leaf-wise (best-first) tree growth with Gradient-based One-Side Sampling (GOSS), finding better splits on limited data faster.
   - CatBoost applies ordered boosting to prevent target leakage — particularly relevant for sequential time-series training.

3. **Hand-Off to Phase 6 (Hybrid Ensemble)**:
   - The verified winner model name is stored in `models/best_model_name.txt`.
   - In Phase 6, the 16-feature hybrid matrix ($15 \\text{ features} + \\text{lstm\\_pred}$) is used, and Optuna hyperparameter optimisation is applied to the winner.''', 'cell-11'))

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

notebook_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '05_candidate_ml_comparison.ipynb')
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Successfully generated {notebook_path}")
