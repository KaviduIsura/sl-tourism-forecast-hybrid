# Phase 4: Deep Learning Sequence Model (LSTM) — Comprehensive Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Notebook:** [`notebooks/04_lstm.ipynb`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/notebooks/04_lstm.ipynb)  
**Output Plots:**  
- Forecast Comparison: [`results/plots/lstm_forecast.png`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/plots/lstm_forecast.png)  
- Training Loss Curve: [`results/plots/lstm_training_loss.png`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/plots/lstm_training_loss.png)  
**Model Artifact:** [`models/lstm_model.keras`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/lstm_model.keras)  
**Scaler Artifacts:** [`models/scaler_X.pkl`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/scaler_X.pkl), [`models/scaler_y.pkl`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/scaler_y.pkl)  
**Metrics Output:** [`results/metrics/lstm_metrics.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/metrics/lstm_metrics.csv)  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Objective

In Phase 3, we observed that the classical econometric model (SARIMA) failed severely when an exogenous shock occurred: during the 2022 Sri Lankan Economic Crisis, SARIMA's error rate exploded to **86.96% MAPE** because it was blind to macroeconomic and real-time behavioral signals.

Phase 4 introduces our first non-linear deep learning model: a **Long Short-Term Memory (LSTM) Recurrent Neural Network**.

The objective of Phase 4 was to:
1. Construct 3D tensor sequence windows with an annual lookback ($L = 12$ months) across 11 multi-variate features (exogenous Google Trends, USD/LKR exchange rates, crisis flags, cyclical sinusoidal months, and autoregressive lags).
2. Fit an architecture with `LSTM(32)`, `Dropout(0.2)`, `Dense(16)`, and `Dense(1)` using Early Stopping to prevent overfitting on monthly data.
3. Compare the LSTM's regime-segmented performance against the SARIMA baseline to measure shock resilience.
4. Export the serialized model (`models/lstm_model.keras`), scalers (`scaler_X.pkl`, `scaler_y.pkl`), and predictions (`lstm_all_predictions.csv`) to serve as the sequence meta-feature for Phase 6 Hybrid Stacking.

---

## 2. Theoretical Foundation: How LSTM Works

Standard Recurrent Neural Networks (RNNs) suffer from the **vanishing gradient problem**, making them incapable of learning dependencies across long horizons (such as a 12-month annual tourism cycle). LSTMs resolve this through a specialized **Memory Cell** regulated by three multiplicative gating mechanisms:

$$\begin{aligned}
\mathbf{f}_t &= \sigma\left(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right) && \text{(Forget Gate: Decides what past context to discard)} \\
\mathbf{i}_t &= \sigma\left(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\right) && \text{(Input Gate: Decides which new information to store)} \\
\tilde{\mathbf{C}}_t &= \tanh\left(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\right) && \text{(Candidate Memory: New candidate values)} \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t && \text{(Cell State: Long-term memory updated along linear highway)} \\
\mathbf{o}_t &= \sigma\left(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\right) && \text{(Output Gate: Decides what parts of memory to output)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh\left(\mathbf{C}_t\right) && \text{(Hidden State: Current output passed to next step)}
\end{aligned}$$

### Why This Matters for Sri Lanka Tourism:
- The **Cell State ($\mathbf{C}_t$)** maintains the repeating 12-month seasonal memory of winter peaks and monsoon troughs.
- When an exogenous shock strikes (indicated by `shock_flag = 1`, surging `usd_lkr_rate`, and plunging `gtrends_flights`), the **Forget Gate ($\mathbf{f}_t$)** can dynamically down-weight obsolete pre-crisis memory, while the **Input Gate ($\mathbf{i}_t$)** integrates the real-time crash signal to dampen arrival predictions.

---

## 3. Step-by-Step Implementation Breakdown

### Step 4.1 — Data Preparation & Scaler Serialization
- Filtered to `ml_ready == 1` observations (120 months: 2016-01 to 2025-12).
- **11 Feature Columns**:
  1. `usd_lkr_rate_monthly` (Macroeconomic signal)
  2. `gtrends_flights` (Leading search intent)
  3. `gtrends_hotels` (Accommodation search intent)
  4. `gtrends_visa` (ETA administrative search intent)
  5. `shock_flag` (Binary crisis indicator)
  6. `month_sin` (Cyclical calendar angle)
  7. `month_cos` (Cyclical calendar angle)
  8. `arrivals_lag_1` (1-month inertia)
  9. `arrivals_lag_3` (Quarterly momentum)
  10. `arrivals_lag_12` (Annual seasonal lag)
  11. `arrivals_rolling_mean_3` (Short-term rolling smoother)
- **Strict Leakage Prevention**:
  `MinMaxScaler` for $X$ and `MinMaxScaler` for $y$ were fitted **exclusively on the training split** (72 months: 2016–2021).
- Persisted to [`models/scaler_X.pkl`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/scaler_X.pkl) and [`models/scaler_y.pkl`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/scaler_y.pkl).

### Step 4.2 — 3D Sequence Windowing ($L = 12$)
- LSTM inputs must be 3-dimensional tensors: `(samples, timesteps, features)`.
- We configured a lookback window of $L = 12$ months.
- **Train Tensor Shape**: `(60, 12, 11)` paired with targets of shape `(60, 1)`.
- **Test Tensor Shape**: `(48, 12, 11)`. By prepending the final 12 training months, every one of the 48 test months (2022-01 to 2025-12) receives an out-of-sample prediction without future leakage.

### Step 4.3 — Network Architecture
- `Input(shape=(12, 11))`
- `LSTM(32, activation='tanh')`: 32 recurrent units (5,632 parameters).
- `Dropout(0.2)`: Drops 20% of recurrent activations during training to prevent memorization on small monthly datasets.
- `Dense(16, activation='relu')`: Intermediate compression layer (528 parameters).
- `Dense(1)`: Linear prediction head (17 parameters).
- **Total Trainable Parameters**: **6,177** (compact architecture specifically chosen to prevent overfitting).

### Step 4.4 — Training with Early Stopping
- Fitted for up to 200 epochs, `batch_size = 16`, `validation_split = 0.1`.
- `EarlyStopping` monitored `val_loss` with `patience = 15` and restored best weights.
- Training converged at epoch 21 with a validation loss of $0.0163$.
- Visualized learning dynamics in [`results/plots/lstm_training_loss.png`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/plots/lstm_training_loss.png).

### Step 4.5 — Multi-Horizon Prediction & Inverse Scaling
- Evaluated test tensor `X_test_seq`.
- Inverted normalization via `scaler_y.inverse_transform()`.
- Clipped predictions to $\ge 0$ (tourist arrivals cannot be negative).

### Step 4.6 — Dual-Regime Evaluation
- Computed RMSE, MAE, MAPE, and $R^2$ across Full Test, Stable Regime, and Shock Regime.
- Exported table to [`results/metrics/lstm_metrics.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/metrics/lstm_metrics.csv).

### Step 4.7 — Model Persistence & Stacking Export
- Persisted trained neural network to [`models/lstm_model.keras`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/lstm_model.keras) (103 KB).
- Exported test predictions to [`results/metrics/lstm_predictions.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/metrics/lstm_predictions.csv).
- Exported complete `lstm_pred` column across all ML-ready rows to [`results/metrics/lstm_all_predictions.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/metrics/lstm_all_predictions.csv) for input into Phase 6 Hybrid Ensemble Stacking.

---

## 4. Empirical Performance: SARIMA vs LSTM Comparison

The table below contrasts the classical econometric baseline against the LSTM deep learning network across all three test regimes:

| Model | Evaluation Regime | Months ($N$) | RMSE | MAE | MAPE (%) | $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SARIMA** | Full Test Horizon | 48 | **50,850.4** | **40,564.8** | 38.68% | $+0.361$ |
| **LSTM** | Full Test Horizon | 48 | 54,629.4 | 43,910.4 | **34.43%** | $+0.262$ |
| --- | --- | --- | --- | --- | --- | --- |
| **SARIMA** | Stable Regime | 36 | **53,985.7** | **41,853.3** | **22.59%** | $-0.188$ |
| **LSTM** | Stable Regime | 36 | 59,811.7 | 48,467.0 | 26.58% | $-0.458$ |
| --- | --- | --- | --- | --- | --- | --- |
| **SARIMA** | **Shock Regime (2022 Crisis)** | 12 | 39,996.0 | 36,699.3 | 86.96% | $-1.248$ |
| **LSTM** | **Shock Regime (2022 Crisis)** | 12 | **34,715.3** | **30,240.7** | **58.00%** | $\mathbf{-0.694}$ |

---

## 5. Core Dissertation Insights: Why LSTM Outperforms During Shocks

### 1. Significant Shock Reduction (28.96 Percentage Points Cut)
- During the 12 months of the 2022 Economic Crisis, SARIMA generated an error rate of **86.96% MAPE**.
- The LSTM reduced this error to **58.00% MAPE**, cutting relative error by nearly **29 percentage points**.
- RMSE dropped from **39,996 to 34,715**, and MAE dropped from **36,699 to 30,240**.

### 2. The Mechanism of Improvement
- Unlike SARIMA, which only saw prior arrivals, the LSTM's memory cells received real-time updates:
  - `gtrends_flights` and `gtrends_hotels` search volumes plummeted, signaling demand destruction.
  - `usd_lkr_rate_monthly` surged from 200 to >360, signaling sovereign macroeconomic distress.
  - `shock_flag = 1` alerted the recurrent gates to enter a crisis regime.
- As a result, the LSTM flattened its forecast line and avoided SARIMA's severe over-prediction during the winter peak of 2022.

### 3. The Complementary Role in the Hybrid Architecture
- While the LSTM demonstrates superior shock resilience, SARIMA still captured slightly better stability during normal seasonal years (22.59% vs 26.58%).
- This is the exact motivation for the **Hybrid Stacking Ensemble (Phase 6)**:
  By combining the LSTM's sequence predictions with tree-based machine learning models (Random Forest, Gradient Boosting), the meta-learner learns to trust the LSTM during shocks and rely on tabular regressors during stable periods.

---

## 6. How to Explain Phase 4 in Your Project Viva / Examiner Defense

> **Sample Viva / Interview Response:**  
> *"In Phase 4, we transitioned from classical linear econometrics to deep recurrent learning by designing an LSTM neural network. We windowed 11 multi-variate features with a 12-month annual lookback and applied Early Stopping to prevent overfitting on monthly data.  
> The comparative results validate our hypothesis: during the 2022 Economic Crisis, the LSTM reduced forecasting MAPE from 86.96% (SARIMA) to 58.00%—a reduction of almost 29 percentage points. Because the LSTM's memory cells integrate exogenous signals like Google Trends and macroeconomic exchange rates, it adapts dynamically when crises occur.  
> These LSTM predictions are saved to serve as the sequential meta-feature for our Phase 6 Hybrid Stacking model."*
