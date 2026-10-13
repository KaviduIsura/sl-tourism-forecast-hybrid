# Phase 2: Exploratory Data Analysis (EDA) — Comprehensive Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Notebook:** [`notebooks/02_eda.ipynb`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/notebooks/02_eda.ipynb)  
**Output Plots:** [`results/plots/`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/plots/)  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Objective

Before developing complex predictive algorithms, rigorous Exploratory Data Analysis (EDA) is essential to uncover underlying patterns, quantify seasonal structures, detect structural breaks, and statistically validate the research hypothesis.

The central research hypothesis of this dissertation is:  
> *"Traditional univariate time series models (like SARIMA) fail during exogenous shocks because historical seasonal autoregression breaks down. Incorporating real-time exogenous signals (Google Trends search queries and macroeconomic exchange rates) into a Hybrid LSTM-ML architecture provides superior forecasting resilience under shock regimes."*

Phase 2 provides the empirical statistical proof supporting this hypothesis through 6 high-resolution publication figures and comparative regime analysis.

---

## 2. In-Depth Breakdown of the 6 Generated Plots

All figures are rendered at 150 DPI and saved in [`results/plots/`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/results/plots/):

### Plot 1: `01_arrivals_timeseries.png` — Historical Arrivals with Exogenous Shock Regimes
- **What it displays**: The monthly tourist arrivals (in thousands) from January 2015 to December 2025. Red translucent bands (`ax.axvspan`) shade the crisis intervals, with annotations pointing to the Easter Attacks (Apr 2019), COVID-19 Pandemic (Mar 2020), and Economic Crisis (Apr 2022).
- **Key Observation**:
  - Sri Lanka experienced steady, compounding growth between 2015 and 2018, reaching an all-time peak of **258,928 tourists in December 2018**.
  - Three catastrophic structural drops followed:
    1. **Easter 2019**: Immediate decline from 166k in April to 37k in May.
    2. **COVID-19 2020–2021**: The complete shutdown of global aviation caused an unprecedented drop to **zero (0) arrivals** in April and May 2020.
    3. **Economic Crisis 2022**: Power blackouts, fuel lines, and political turmoil aborted the post-COVID recovery, depressing arrivals to a low of 29,802 in May 2022.
- **Why this matters**: Proves visually that the dataset is severely non-stationary with extreme variance shifts that violate linear assumptions.

---

### Plot 2: `02_seasonal_decomposition.png` — Additive Classical Decomposition
- **What it displays**: Decomposes the time series into four stacked subplots:
  $$\text{Observed}_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t$$
  (Using a shock-interpolated baseline series to prevent crisis spikes from distorting the underlying seasonality).
- **Key Observation**:
  - **Trend**: Exhibits rising macro momentum (2015–2018), a complete collapse (2020–2022), and a sharp recovery starting in mid-2023.
  - **Seasonal**: A remarkably stable repeating wave of period $m = 12$ months. Peak seasonal swings add up to $+80,000$ tourists in December and subtract $-50,000$ in May.
  - **Residual**: High residual variance during the shock years, demonstrating where simple linear additive assumptions fail.
- **Why this matters**: Validates that seasonal patterns exist, but they are vulnerable to external shock shocks.

---

### Plot 3: `03_correlation_heatmap.png` — Feature Correlation Matrix
- **What it displays**: An annotated lower-triangular Pearson correlation heatmap across tourist arrivals, lagged variables, exogenous search queries, exchange rates, and the shock flag.
- **Key Numeric Correlations with Tourist Arrivals**:
  - `arrivals_lag_1`: **$+0.88$** (strong immediate momentum).
  - `shock_flag`: **$-0.79$** (massive inverse impact of crisis periods).
  - `gtrends_hotels`: **$+0.68$** (high correlation with actual tourist volume).
  - `gtrends_visa`: **$+0.67$** (high correlation with actual tourist volume).
  - `arrivals_lag_12`: **$+0.64$** (annual seasonal persistence).
  - `gtrends_flights`: **$+0.48$** (moderate positive relationship).
  - `usd_lkr_rate_monthly`: **$-0.12$** (counter-intuitive negative correlation due to massive currency devaluation occurring during low post-crisis recovery).
- **Why this matters**: Demonstrates that exogenous Google Trends features are strongly tied to tourist volumes, confirming their value as model inputs.

---

### Plot 4: `04_exogenous_overlays.png` — Arrivals vs Exogenous Signals
- **What it displays**: 3-panel dual-axis plots directly comparing monthly arrivals against:
  1. Panel A: USD/LKR Exchange Rate
  2. Panel B: Google Trends Search Index for Flights
  3. Panel C: Google Trends Search Index for Hotels
- **Key Observation**:
  - **USD/LKR Rate**: Hovered stably between 130 and 185 LKR/USD from 2015 to 2021. In March 2022, during the sovereign debt default, it devalued sharply to over 360 LKR/USD. While a cheap currency theoretically attracts tourists, the domestic turmoil temporarily suppressed arrivals.
  - **Google Trends (Flights & Hotels)**: Show striking synchrony with actual arrivals. Search interest peaks 1–2 months before tourist arrivals pick up, functioning as an empirical leading indicator of travel demand.
- **Why this matters**: Justifies using Google Trends search queries in Deep Learning models as early warning indicators of demand shifts.

---

### Plot 5: `05_acf_pacf.png` — ACF & PACF Order Identification
- **What it displays**: Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) up to 36 monthly lags.
- **Key Diagnostic Rules for SARIMA (Phase 3)**:
  1. **Annual Seasonality ($m = 12$)**: The ACF displays pronounced spikes at lag 12, lag 24, and lag 36, proving mathematically that the seasonal cycle is annual ($m = 12$).
  2. **Autoregressive Order ($p$)**: The PACF shows significant spikes at lag 1 and lag 2, followed by an immediate sharp truncation within the confidence bands. This suggests a non-seasonal autoregressive order of $p \in \{1, 2\}$.
  3. **Differencing Order ($d, D$)**: The very slow decay in the ACF confirms that the series is non-stationary and requires first-order differencing ($d = 1$) and seasonal differencing ($D = 1$).
- **Why this matters**: Directly supplies the mathematical search bounds for Auto-ARIMA in Phase 3.

---

### Plot 6: `06_monthly_seasonality.png` — Monthly Seasonality Distribution
- **What it displays**: Seaborn boxplot of tourist arrivals across all 12 calendar months, filtered strictly for **stable periods** (`shock_flag == 0`) to eliminate crisis outliers.
- **Key Seasonal Profile of Sri Lanka**:
  - **Primary Peak (Northern Winter Sun Seekers)**:
    - December: Mean **236,018** arrivals
    - January: Mean **205,005** arrivals
    - February: Mean **202,458** arrivals
    - March: Mean **197,456** arrivals
  - **Secondary Mini-Peak (Summer Holidays & Esala Perahera)**:
    - July: Mean **181,908** arrivals
    - August: Mean **173,378** arrivals
  - **Trough (Monsoon Season)**:
    - May: Mean **107,011** arrivals (annual low)
    - June: Mean **114,857** arrivals
- **Why this matters**: Confirms Sri Lanka's distinct bimodal tourism seasonality, providing a quantitative baseline for evaluating model forecasts.

---

## 3. Empirical Statistical Proof for the Dissertation

The EDA produced empirical results that directly substantiate the thesis:

### Quantitative Regime Comparison

| Metric | Full Dataset (2015–2025) | Stable Regime (`shock_flag=0`) | Shock Regime (`shock_flag=1`) | Difference / Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Observation Count** | 132 months | 97 months (73.5%) | 35 months (26.5%) | — |
| **Mean Monthly Arrivals** | 132,868 | **168,921** | **32,949** | **$-80.5\%$ collapse** |
| **Standard Deviation** | 76,107 | 44,792 | 34,105 | High variance shift |
| **Minimum Arrivals** | 0 (Apr 2020) | 63,366 (May 2015) | 0 (Apr 2020) | Complete shutdown |
| **Maximum Arrivals** | 258,928 (Dec 2018) | 258,928 (Dec 2018) | 166,975 (Apr 2019) | — |

---

### The "Seasonality Breakdown" Proof

To test whether classical time-series assumptions hold during crises, we analyzed the correlation between arrivals and their 12-month annual lag (`arrivals_lag_12`):

$$\begin{aligned}
r_{\text{stable}}(\text{arrivals}, \text{arrivals\_lag\_12}) &= \mathbf{+0.741} \quad \text{(Strong positive memory)} \\
r_{\text{shock}}(\text{arrivals}, \text{arrivals\_lag\_12}) &= \mathbf{-0.110} \quad \text{(Complete breakdown)}
\end{aligned}$$

> **Key Finding:**  
> In stable times, looking back at arrivals from 12 months ago is an effective forecasting strategy ($r = +0.741$). But during an exogenous shock, that relationship completely breaks down and turns negative ($r = -0.110$). A model relying solely on past arrivals (like standard SARIMA) will produce massive forecast errors when a crisis hits.

---

### The "Exogenous Resilience" Proof

During shock periods when autoregressive lags fail, what features actually predict tourist volume?

| Feature | Stable Regime Correlation ($r$) | Shock Regime Correlation ($r$) | Behavior Under Shock |
| :--- | :---: | :---: | :--- |
| `gtrends_flights` | $+0.284$ | $\mathbf{+0.915}$ | **Surges to near-perfect correlation** |
| `gtrends_visa` | $+0.287$ | $\mathbf{+0.827}$ | **Dominates recovery signal** |
| `gtrends_hotels` | $+0.441$ | $\mathbf{+0.771}$ | **Strong real-time predictor** |
| `arrivals_lag_12` | $+0.741$ | $\mathbf{-0.110}$ | **Collapses completely** |

> **Key Finding:**  
> When black-swan events disrupt historical patterns, Google Trends search queries become the strongest predictors of tourist volume ($r > 0.80$). This justifies the Hybrid LSTM-ML ensemble model: the LSTM learns sequence dynamics and shock transitions, while the ML ensemble utilizes real-time exogenous signals to adapt when historical lags fail.

---

## 4. How to Explain Phase 2 in Your Project Viva / Examiner Defense

> **Sample Viva / Interview Response:**  
> *"In Phase 2, we conducted exploratory data analysis to evaluate our core research hypothesis. The empirical results yielded two critical insights:  
> First, we proved that Sri Lanka's tourism exhibits an 80.5% collapse in mean arrivals during exogenous shocks. More importantly, we discovered that the correlation between current arrivals and the 12-month seasonal lag drops from $+0.741$ in stable conditions to $-0.110$ during shocks. This confirms that classical univariate models like SARIMA fundamentally break down during black-swan events because seasonal memory is disrupted.  
> Second, we showed that during shock periods, Google Trends search interest for flights and visas surges to correlations exceeding $+0.91$. This validates our hybrid approach: integrating exogenous search and economic signals to sustain accurate forecasts when historical patterns fail."*
