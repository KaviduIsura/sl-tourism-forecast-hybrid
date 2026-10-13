# Phase 1: Data Preparation & Feature Engineering — Comprehensive Guide

**Research Project:** Hybrid LSTM-ML Ensemble for Tourist Arrival Prediction under Exogenous Shocks  
**Student ID:** KIC-BSCCSAI-251P-034  
**Programme:** BSc Computer Science with AI  
**Notebook:** [`notebooks/01_data_prep.ipynb`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/notebooks/01_data_prep.ipynb)  
**Phase Status:** ✅ Fully Completed & Verified  

---

## 1. Executive Summary & Objective

In time-series forecasting, raw data cannot be fed directly into machine learning models. Time series data often contains non-standard timestamp representations, missing values, structural breaks, and non-linear patterns that models cannot comprehend without feature transformation.

The objective of Phase 1 was to take the raw multi-source dataset from [`data/raw/Dataset.xlsx`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/data/raw/Dataset.xlsx), normalize dates, impute and validate exogenous signals, engineer cyclical temporal features and autoregressive lags, flag exogenous shock periods, and export a canonical 18-column feature store to:
- [`data/processed/tourism_dataset_final.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/data/processed/tourism_dataset_final.csv) (used by ML/DL notebooks)
- [`backend/data/historical.csv`](file:///Users/kaviduisura/Desktop/Final%20Research%20/sl-tourism-forecast-hybrid/backend/data/historical.csv) (used by the FastAPI application)

---

## 2. Detailed Step-by-Step Breakdown of What Was Done

### Step 1.1 — Load Master Dataset
- **Action**: Ingested `Dataset.xlsx` from the `Dataset` sheet into a pandas DataFrame.
- **Verification**: Confirmed the dataset contains 132 monthly observations spanning from January 2015 to December 2025.

### Step 1.2 — Resolve Date Serialization (The Excel Date Bug)
- **Problem**: Microsoft Excel stores dates internally as serial integers counting the number of days since December 30, 1899 (e.g., `42005` represents January 1, 2015). If loaded naively, Python treats dates as integers, breaking temporal indexing.
- **Solution**: We detected numeric timestamps and applied Excel base-offset conversion:
  $$\text{datetime} = \text{pd.to\_datetime('1899-12-30') } + \text{pd.to\_timedelta}(\text{serial\_number}, \text{unit='D'})$$
  Each date was normalized to the 1st of each calendar month (`YYYY-MM-01`).

### Step 1.3 — Data Type Conversions & Missing Value Auditing
- **Shock Type Cleaning**: Excel text entries frequently contain string literal `'None'` or `'none'`. These were replaced with `np.nan` so they are properly recognized as null values in Python.
- **Numeric Enforcement**: All numeric features (USD/LKR exchange rate, Google Trends search volumes, tourist arrivals) were cast to explicit `float64` or `int64` types.

### Step 1.4 — Temporal & Cyclical Feature Engineering

#### A. Cyclical Month Encoding ($\sin$ and $\cos$)
- **The Problem with Raw Integers**: If month is represented as a single number $1, 2, \dots, 12$, machine learning algorithms perceive Month 1 (January) and Month 12 (December) as being 11 units apart ($|12 - 1| = 11$), even though in time they are adjacent consecutive months!
- **The Solution**: We projected the calendar month onto a 2-dimensional trigonometric unit circle:
  $$\text{month\_sin} = \sin\left(\frac{2 \pi \cdot \text{month}}{12}\right)$$
  $$\text{month\_cos} = \cos\left(\frac{2 \pi \cdot \text{month}}{12}\right)$$
  This guarantees that January and December have minimal Euclidean distance, preserving cyclical continuity.

#### B. Autoregressive Lag Features
Tourist behavior exhibits both short-term inertia and long-term annual seasonality. We engineered three key lags:
1. `arrivals_lag_1`: Tourist volume from 1 month prior ($t-1$). Captures immediate momentum.
2. `arrivals_lag_3`: Tourist volume from 3 months prior ($t-3$). Captures quarterly trajectory.
3. `arrivals_lag_12`: Tourist volume from 12 months prior ($t-12$). Captures annual seasonal baseline.

#### C. Rolling Window Moving Average
- `arrivals_rolling_mean_3`: The 3-month moving average of tourist arrivals:
  $$\text{rolling\_mean\_3}_t = \frac{1}{3} \sum_{i=0}^{2} \text{arrivals}_{t-i}$$
  Smoothes out short-term idiosyncratic noise.

### Step 1.5 — Validation of Exogenous Shock Regimes
Sri Lanka's tourism industry experienced three massive, distinct black-swan events between 2015 and 2025. Each was verified and categorized:
1. **Easter Sunday Terrorist Attacks**:
   - Shock Flag: `1`
   - Shock Type: `Easter Attack`
   - Occurrence: April 2019 (1 month direct impact, with prolonged aftermath in Q2 2019).
2. **COVID-19 Global Pandemic**:
   - Shock Flag: `1`
   - Shock Type: `COVID-19`
   - Occurrence: March 2020 to December 2021 (22 consecutive months of severe restrictions, including total border closure in Apr–May 2020).
3. **Sri Lankan Economic Crisis & Sovereign Debt Default**:
   - Shock Flag: `1`
   - Shock Type: `Economic Crisis`
   - Occurrence: April 2022 to March 2023 (12 consecutive months of currency devaluation, fuel shortages, and social unrest).

### Step 1.6 — Identification of ML-Ready Observations
- **Warmup Period Necessity**: Because `arrivals_lag_12` requires 12 prior months of data to compute, the first 12 rows of the dataset (Year 2015) naturally have `NaN` for `arrivals_lag_12`.
- **The `ml_ready` Flag**:
  - `0`: For the first 12 warmup rows (2015-01 to 2015-12). Used strictly for historical lookbacks.
  - `1`: For all subsequent 120 rows (2016-01 to 2025-12). These rows have zero null values across all features and are the exact rows used for training, cross-validation, and testing.

### Step 1.7 — Canonical Dataset Export
The final validated dataset was saved in two places:
1. `data/processed/tourism_dataset_final.csv`: Feature store for all Jupyter modeling experiments.
2. `backend/data/historical.csv`: Copy utilized by the FastAPI server to serve historical chart data to the React dashboard.

---

## 3. Complete Feature Dictionary (18 Columns)

| Column Name | Data Type | Source / Calculation | Role in Modeling Pipeline |
| :--- | :---: | :--- | :--- |
| `date` | `datetime64` | Normalized timestamp (`YYYY-MM-01`) | Temporal index |
| `year` | `int64` | Extracted from date (2015–2025) | Macro-temporal reference |
| `month` | `int64` | Extracted from date (1–12) | Calendar month indicator |
| `tourist_arrivals` | `int64` | SLTDA official arrival records | **Target variable ($y$)** |
| `usd_lkr_rate_monthly` | `float64` | Central Bank of Sri Lanka (CBSL) | Exogenous macroeconomic feature |
| `gtrends_flights` | `float64` | Google Search index: "Flights to Sri Lanka" | Exogenous leading demand indicator |
| `gtrends_hotels` | `float64` | Google Search index: "Hotels in Sri Lanka" | Exogenous leading demand indicator |
| `gtrends_visa` | `float64` | Google Search index: "Sri Lanka ETA Visa" | Exogenous leading demand indicator |
| `shock_flag` | `int64` | Binary indicator (1 = Shock, 0 = Stable) | Regime segmentation feature |
| `shock_type` | `object` | Categorical label (`COVID-19`, `Economic Crisis`, etc.) | Metadata for regime analysis |
| `month_sin` | `float64` | $\sin(2\pi \cdot \text{month} / 12)$ | Continuous cyclical feature |
| `month_cos` | `float64` | $\cos(2\pi \cdot \text{month} / 12)$ | Continuous cyclical feature |
| `arrivals_lag_1` | `float64` | Arrivals at $t-1$ | Autoregressive feature |
| `arrivals_lag_3` | `float64` | Arrivals at $t-3$ | Autoregressive feature |
| `arrivals_lag_12` | `float64` | Arrivals at $t-12$ | Seasonal autoregressive feature |
| `arrivals_rolling_mean_3` | `float64` | Mean of $t, t-1, t-2$ | Moving trend feature |
| `dataset_split` | `object` | `train` (2015–2023) or `test` (2024–2025) | Out-of-sample evaluation boundary |
| `ml_ready` | `int64` | 1 if row has all 12 lag features, else 0 | Training filter flag |

---

## 4. How to Explain Phase 1 in Your Project Viva / Examiner Defense

> **Sample Viva / Interview Response:**  
> *"In Phase 1, we converted raw multi-source monthly data into an ML-ready feature matrix. A primary challenge was preventing data leakage: we engineered temporal lags (`lag_1`, `lag_3`, `lag_12`) and marked the first 12 months with `ml_ready=0` as an initialization warmup period. To enable neural networks to perceive seasonal transitions naturally, we converted the discrete month integers into continuous cyclical sinusoids (`month_sin`, `month_cos`). Furthermore, we tagged the three major exogenous disruptions—Easter 2019, COVID-19, and the 2022 Economic Crisis—to facilitate dual-regime evaluation."*
