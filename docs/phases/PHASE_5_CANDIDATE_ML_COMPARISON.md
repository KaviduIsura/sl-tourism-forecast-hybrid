# Phase 5: Candidate Machine Learning Model Comparison — Comprehensive Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Notebook:** [`notebooks/05_candidate_ml_comparison.ipynb`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/notebooks/05_candidate_ml_comparison.ipynb)  
**Output Plot:** [`results/plots/candidate_models_cv.png`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/plots/candidate_models_cv.png)  
**Winning Model Artifact:** [`models/best_model_name.txt`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/best_model_name.txt)  
**Comparison Metrics Output:** [`results/metrics/candidate_comparison.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/metrics/candidate_comparison.csv)  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Objective

In academic machine learning research, choosing an algorithm arbitrarily without comparative justification is penalized by examiners. To build a robust **Hybrid Stacking Ensemble (Phase 6)**, we must first empirically test competing machine learning paradigms under identical, leakage-free conditions.

The objective of Phase 5 was to:
1. Define three structurally distinct candidate regressors:
   - **Support Vector Regression (SVR)**: Kernel-based margin maximization.
   - **Random Forest (RF)**: Bootstrap aggregation (Bagging) ensemble of decision trees.
   - **Gradient Tree Boosting (GB)**: Sequential stage-wise residual correction (Boosting).
2. Evaluate all three candidates using an expanding 5-fold **`TimeSeriesSplit`** cross-validation on the training set.
3. Empirically determine the winning model based on lowest average validation RMSE and persist the winner name to [`models/best_model_name.txt`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/best_model_name.txt) for automated loading in Phase 6.
4. Establish the Explainable AI (XAI) strategy for Phase 8 based on whether the winner is tree-based (`TreeExplainer`) or kernel-based (`KernelExplainer`).

---

## 2. The Three Candidate Model Families

| Model | Family | Primary Mechanism | Hyperparameter Specification |
| :--- | :--- | :--- | :--- |
| **SVR** | Kernel Margin Regressor | Projects inputs into an infinite-dimensional Hilbert space using Radial Basis Functions (RBF) to fit a flat $\epsilon$-insensitive tube. | `kernel='rbf', C=10, epsilon=0.1` |
| **Random Forest** | Bagging Ensemble | Trains 300 independent, de-correlated decision trees on bootstrap subsamples and averages predictions to minimize variance. | `n_estimators=300, max_depth=6, random_state=42` |
| **Gradient Boosting** | Boosting Ensemble | Sequentially adds 300 shallow trees, where each tree fits the negative gradient (pseudo-residuals) of the loss function. | `n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42` |

---

## 3. Methodological Rigor: Why `TimeSeriesSplit` is Mandatory

A common mistake in beginner data science is applying standard random $K$-Fold cross-validation to temporal data:
- **Standard $K$-Fold Flaw**: Randomly shuffles observations. A model might be trained on data from 2020 to predict an event in 2017. This introduces **lookahead data leakage** (peeking into the future).
- **TimeSeriesSplit Solution**: Enforces strict chronological causality across 5 expanding splits:
  - **Fold 1**: Train on $[1 \dots 20]$, Validate on $[21 \dots 30]$
  - **Fold 2**: Train on $[1 \dots 30]$, Validate on $[31 \dots 40]$
  - **Fold 3**: Train on $[1 \dots 40]$, Validate on $[41 \dots 50]$
  - **Fold 4**: Train on $[1 \dots 50]$, Validate on $[51 \dots 60]$
  - **Fold 5**: Train on $[1 \dots 60]$, Validate on $[61 \dots 72]$

Every validation fold is strictly evaluated on the **future** given the **past**, mirroring real-world operational forecasting.

---

## 4. Empirical Cross-Validation Results

The 5-fold cross-validation performance on the 72-month training dataset is summarized below:

| Rank | Candidate Model | Mean Validation RMSE | Std Validation RMSE | Mean Validation MAE | Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Random Forest** | **43,510.7** | $\pm 32,853.5$ | **35,892.5** | **WINNER (Selected for Phase 6)** |
| 🥈 **2** | **Gradient Boosting** | **48,130.5** | $\pm 34,314.4$ | **38,659.9** | Runner-up |
| 🥉 **3** | **SVR** | **92,944.5** | $\pm 51,718.5$ | **86,741.9** | Substantially underperformed |

---

## 5. Core Dissertation Insights: Why Random Forest Won

### 1. Superior Robustness of Bagging over Boosting on Small, Noisy Datasets
- Our training dataset contains 72 monthly samples that include two major shocks (Easter Attacks in 2019 and COVID-19 border closures in 2020–2021).
- **Gradient Boosting** builds trees sequentially by fitting residuals from previous trees. In the presence of extreme outliers (such as months where arrivals plunged to near zero), boosting tends to chase the high residual errors, leading to overfitting on crisis-distorted training folds.
- **Random Forest**, by contrast, builds trees independently using random feature subsampling and bootstrap aggregation. Outlier shocks are dampened across the 300 averaged trees, resulting in a substantially lower validation error (**43,510.7 vs 48,130.5**).

### 2. Why Support Vector Regression (SVR) Failed
- SVR operates by establishing a global continuous hyperplane in kernel space.
- Tourism time series under exogenous shocks exhibit **discontinuous, non-linear step collapses** (instantaneous drops of 80% to 100%).
- Continuous distance-based kernels (RBF) struggle with abrupt regime switches, resulting in an error more than double that of tree models (**92,944.5 RMSE**).

---

## 6. Downstream Implications for Phases 6 and 8

### A. Hand-Off to Phase 6 (Hybrid Ensemble)
- The winning algorithm name (`Random Forest`) was persisted to [`models/best_model_name.txt`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/best_model_name.txt).
- In Phase 6, we will load `Random Forest` as our base model, feed it the 12-feature hybrid matrix ($11 \text{ tabular features} + \text{lstm\_pred}$), and fine-tune its hyperparameters using `GridSearchCV`.

### B. Hand-Off to Phase 8 (SHAP Explainability Strategy)
Because **Random Forest** won over SVR:
- We can utilize **`shap.TreeExplainer`** rather than `shap.KernelExplainer`.
- `TreeExplainer` computes exact local and global Shapley values in polynomial time $\mathcal{O}(TLD^2)$, completely avoiding the slow sampling approximations required by kernel methods.

---

## 7. How to Explain Phase 5 in Your Project Viva / Examiner Defense

> **Sample Viva / Interview Response:**  
> *"In Phase 5, we conducted an empirical candidate comparison across three distinct machine learning paradigms: Support Vector Regression, Random Forest, and Gradient Tree Boosting. To prevent temporal data leakage, we evaluated all models using strictly expanding 5-fold `TimeSeriesSplit` cross-validation on our training dataset.  
> Random Forest emerged as the clear winner with the lowest average validation RMSE of 43,510.7, outperforming Gradient Boosting (48,130.5) and SVR (92,944.5). Because Random Forest uses bootstrap aggregation (bagging), it provided superior variance reduction and was far more resilient to the noise generated by black-swan shocks than sequential boosting or continuous kernel hyperplanes.  
> The winning model was saved to `models/best_model_name.txt` to serve as the core meta-regressor in our Phase 6 Hybrid Stacking pipeline, and it enables us to use fast, exact `shap.TreeExplainer` for Phase 8 interpretability."*
