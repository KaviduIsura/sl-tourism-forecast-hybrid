# Phase 3: Classical Econometric Baseline (SARIMA) — Comprehensive Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Notebook:** [`notebooks/03_sarima.ipynb`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/notebooks/03_sarima.ipynb)  
**Output Plot:** [`results/plots/07_sarima_forecast.png`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/plots/07_sarima_forecast.png)  
**Model Artifact:** [`models/sarima_model.pkl`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/sarima_model.pkl)  
**Metrics Output:** [`results/metrics/sarima_metrics.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/metrics/sarima_metrics.csv)  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Objective

In time-series forecasting research, developing complex machine learning or deep learning models without a recognized statistical benchmark is considered poor methodology. Before building LSTM neural networks (Phase 4) or Hybrid ML Ensembles (Phase 6), we must establish a **classical econometric baseline**.

The standard benchmark in international tourism literature is the **Seasonal Autoregressive Integrated Moving Average (SARIMA)** model.

The primary objective of Phase 3 was to:
1. Calibrate an optimal SARIMA model on univariate historical arrivals using the training split (2015–2021).
2. Produce an out-of-sample multi-step forecast across the 48-month test split (2022–2025).
3. Evaluate the model across both **Stable** and **Shock** regimes to quantitatively demonstrate the structural failure of univariate linear models under exogenous crises.
4. Persist model artifacts and test predictions for downstream statistical significance testing (Diebold-Mariano test in Phase 7).

---

## 2. Mathematical Foundation of SARIMA

A SARIMA model is denoted as:
$$\text{SARIMA}(p, d, q)(P, D, Q)_m$$

Where:
- $p$: Non-seasonal autoregressive (AR) order — past values used as predictors.
- $d$: Non-seasonal integration order — degree of differencing required for stationarity.
- $q$: Non-seasonal moving average (MA) order — past forecast errors used as predictors.
- $P$: Seasonal autoregressive order — past values from multiples of the seasonal cycle (e.g. 12 months ago).
- $D$: Seasonal differencing order — differencing across the seasonal cycle ($y_t - y_{t-m}$).
- $Q$: Seasonal moving average order — past seasonal errors.
- $m$: Seasonal period ($m = 12$ for monthly tourism cycles).

In lag operator polynomial form:
$$\Phi_P(B^m) \phi_p(B) (1 - B)^d (1 - B^m)^D y_t = \Theta_Q(B^m) \theta_q(B) \epsilon_t$$

Where:
- $B$ is the backshift operator ($B^k y_t = y_{t-k}$).
- $\epsilon_t \sim \mathcal{WN}(0, \sigma^2)$ is white noise.

### Optimal Order Identified by Auto-ARIMA:
Using stepwise Akaike Information Criterion (AIC) minimization, `pmdarima.auto_arima` identified:
$$\mathbf{\text{SARIMAX}(2, 1, 0)(1, 0, 0)_{12}}$$

- **Non-seasonal part $(2, 1, 0)$**: First difference ($d=1$) with AR lag 1 and lag 2 ($p=2$). This directly confirms our Phase 2 PACF findings where lags 1 and 2 showed prominent spikes before cutting off.
- **Seasonal part $(1, 0, 0)_{12}$**: Seasonal AR of lag 12 ($P=1$) with no seasonal differencing ($D=0$). This confirms the Phase 2 ACF findings where lag 12 was highly significant.
- **AIC Score**: **1964.59** (the lowest among all candidate configurations evaluated).

---

## 3. Step-by-Step Implementation Breakdown

### Step 3.1 — Train/Test Split
We partitioned the monthly arrivals series according to the project's canonical boundary:
- **Train Split (2015-01 to 2021-12)**: 84 monthly observations. Includes the stable pre-pandemic years (2015–2018), Easter Sunday 2019, and the COVID-19 pandemic (2020–2021).
- **Test Split (2022-01 to 2025-12)**: 48 monthly observations. Represents a true out-of-sample forward evaluation horizon ($h = 48$ months).
  - **Shock Period in Test Set**: Jan 2022 – Dec 2022 (12 months: Sri Lankan Economic Crisis, currency devaluation, fuel shortages).
  - **Stable Period in Test Set**: Jan 2023 – Dec 2025 (36 months: Tourism recovery and post-crisis normalization).

### Step 3.2 — Auto-ARIMA Fitting
- Configured `pmdarima.auto_arima` on $y_{\text{train}}$ with $m=12$, $d=1$, stepwise search, and AIC scoring.
- Evaluated 37 candidate configurations. Selected $\text{SARIMAX}(2, 1, 0)(1, 0, 0)_{12}$.
- All coefficients were estimated via Maximum Likelihood Estimation (MLE).

### Step 3.3 — Multi-Step Out-of-Sample Forecasting
- Projected the fitted model forward 48 months ($n_{\text{periods}} = 48$).
- Computed 95% confidence intervals:
  $$\hat{y}_t \pm 1.96 \cdot \hat{\sigma}_t$$
- Post-processed lower bounds with `clip(lower=0)` because tourist volumes cannot physically be negative.

### Step 3.4 — Regime-Segmented Metric Evaluation
We calculated Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and the Coefficient of Determination ($R^2$):

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2}$$
$$\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|$$
$$\text{MAPE} = \frac{1}{N} \sum_{t=1}^N \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100\%$$

### Step 3.5 — Visualization (`07_sarima_forecast.png`)
- Plotted historical train actuals, test actuals, the SARIMA forecast trajectory, 95% confidence bounds, and shaded shock regimes.
- Graphically displays how the forecast behaves relative to the ground truth during the 2022 crisis and the subsequent recovery.

### Step 3.6 — Model Persistence & Predictions Export
- Saved the trained model using `joblib` to [`models/sarima_model.pkl`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/models/sarima_model.pkl) (3.8 MB).
- Exported the test predictions to [`results/metrics/sarima_predictions.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/metrics/sarima_predictions.csv) for direct ingestion by Phase 7 evaluation scripts.

---

## 4. Empirical Performance Results

The quantitative evaluation results across test regimes are summarized below:

| Regime | Time Period | Sample Size ($N$) | RMSE | MAE | MAPE (%) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Test Horizon** | Jan 2022 – Dec 2025 | 48 months | 50,850.4 | 40,564.8 | **38.68%** | $+0.361$ |
| **Stable Regime** | Jan 2023 – Dec 2025 | 36 months | 53,985.7 | 41,853.3 | **22.59%** | $-0.188$ |
| **Shock Regime (2022 Crisis)** | Jan 2022 – Dec 2022 | 12 months | 39,996.0 | 36,699.3 | **86.96%** | $\mathbf{-1.248}$ |

---

## 5. Critical Academic Insights: Why SARIMA Fails Under Shocks

These empirical findings provide the core scientific justification for the rest of the dissertation:

### 1. The Shock Error Explosion
- In stable post-crisis months, SARIMA achieves an acceptable baseline MAPE of **22.59%**.
- However, during the 12 months of the 2022 Economic Crisis, SARIMA's error explodes to **86.96% MAPE** (nearly quadruple the error rate).
- Its $R^2$ plunges to **$-1.248$**, meaning the model's predictions are significantly worse than simply predicting the mean arrival rate.

### 2. The Theoretical Cause of Failure
- **Information Blindness**: SARIMA is a **univariate** linear model. Its input space consists exclusively of past arrival values ($y_{t-1}, y_{t-2}, y_{t-12}$).
- When the 2022 Economic Crisis occurred (fuel shortages, daily 10-hour power outages, sovereign debt default, mass protests), the actual tourism demand plummeted to ~30k/month.
- But because the model had no access to macroeconomic indicators (like the USD/LKR exchange rate spiking from 200 to 360) or real-time leading search indicators (Google Trends dropping), it projected a smooth, unperturbed post-COVID recovery curve.
- It continued predicting winter-peak arrivals of ~100k–120k when actual arrivals were barely 30k.

### 3. Justification for Phase 4 (LSTM) and Phase 6 (Hybrid Ensemble)
- This baseline establishes the necessity of:
  - **Recurrent Deep Learning (Phase 4)**: To capture non-linear temporal dependencies and long-term state transitions.
  - **Exogenous Features (Phase 5 & 6)**: Integrating Google Trends and exchange rates so the model can dynamically adjust when real-world crises occur.

---

## 6. How to Explain Phase 3 in Your Project Viva / Examiner Defense

> **Sample Viva / Interview Response:**  
> *"In Phase 3, we implemented the classical econometric benchmark using SARIMA. Using automated stepwise AIC search via `pmdarima`, we identified an optimal $\text{SARIMAX}(2, 1, 0)(1, 0, 0)_{12}$ specification, aligning with our Phase 2 ACF/PACF diagnostics.  
> We evaluated the model out-of-sample over a 48-month horizon (2022–2025). The results revealed a critical regime disparity: while the model achieved a 22.59% MAPE during stable recovery months, its error rate surged to 86.96% MAPE during the 2022 Economic Crisis, with an $R^2$ of $-1.25$.  
> Because SARIMA is univariate and relies solely on historical autoregression, it was completely blind to exogenous macroeconomic and geopolitical disruptions. This quantitative failure validates the need for our proposed Hybrid LSTM-ML ensemble, which incorporates real-time exogenous signals like Google Trends and exchange rates."*
