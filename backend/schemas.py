from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import math

class ForecastRequest(BaseModel):
    month: int = Field(..., ge=1, le=12, description="Forecast target month (1 to 12)", example=3)
    usd_lkr_rate_monthly: float = Field(..., gt=0, description="USD/LKR monthly exchange rate", example=310.5)
    gtrends_flights: float = Field(..., ge=0, le=100, description="Google Trends relative search index for flights (0-100)", example=48.0)
    gtrends_hotels: float = Field(..., ge=0, le=100, description="Google Trends relative search index for hotels (0-100)", example=55.0)
    gtrends_visa: float = Field(..., ge=0, le=100, description="Google Trends relative search index for visa (0-100)", example=38.0)
    shock_flag: int = Field(0, ge=0, le=1, description="Binary shock indicator (0 = Normal, 1 = Crisis / Disruption)", example=0)
    arrivals_lag_1: float = Field(..., gt=0, description="Tourist arrivals in previous month (Lag 1)", example=180000.0)
    arrivals_lag_3: float = Field(..., gt=0, description="Tourist arrivals 3 months ago (Lag 3)", example=165000.0)
    arrivals_lag_12: float = Field(..., gt=0, description="Tourist arrivals 12 months ago (Lag 12 / YoY base)", example=170000.0)
    arrivals_rolling_mean_3: float = Field(..., gt=0, description="3-month moving average of tourist arrivals", example=175000.0)
    lstm_pred: Optional[float] = Field(None, description="Optional pre-computed deep LSTM sequence prediction. If omitted, calculated automatically.", example=178500.0)

    @field_validator('shock_flag')
    @classmethod
    def validate_shock(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError('shock_flag must be either 0 or 1')
        return v

class ScenarioSimulationRequest(BaseModel):
    base_request: ForecastRequest
    shock_flag_override: Optional[int] = Field(None, ge=0, le=1, description="Test turning shock on (1) or off (0)")
    exchange_rate_pct_change: Optional[float] = Field(0.0, description="Simulate currency devaluation (e.g. +20.0 for +20%)")
    gtrends_pct_change: Optional[float] = Field(0.0, description="Simulate search interest surge or collapse (e.g. -40.0 for -40%)")

class FeatureAttribution(BaseModel):
    feature: str
    impact_arrivals: float
    direction: str  # "positive" (increases forecast) or "negative" (decreases forecast)

class ForecastResponse(BaseModel):
    predicted_arrivals: float = Field(..., description="Forecasted tourist arrivals for target month")
    regime: str = Field(..., description="Operating regime ('Stable / Normal' or 'Exogenous Shock Disruption')")
    model_name: str = Field(..., description="Active production model architecture")
    lstm_component: float = Field(..., description="LSTM sequence feature prediction used in stacking")
    confidence_interval_80: Dict[str, float] = Field(..., description="80% confidence interval (lower, upper)")
    confidence_interval_95: Dict[str, float] = Field(..., description="95% confidence interval (lower, upper)")
    feature_attributions: List[FeatureAttribution] = Field(..., description="Local SHAP feature attributions")

class ScenarioSimulationResponse(BaseModel):
    baseline_prediction: float
    simulated_prediction: float
    net_change: float
    percentage_change: float
    baseline_regime: str
    simulated_regime: str
    scenario_description: str

class HistoryRecord(BaseModel):
    date: str
    year: int
    month: int
    tourist_arrivals: float
    usd_lkr_rate_monthly: float
    gtrends_flights: float
    gtrends_hotels: float
    gtrends_visa: float
    shock_flag: int
    shock_type: Optional[str] = None
    dataset_split: str

class HistoryResponse(BaseModel):
    total_records: int
    date_range: Dict[str, str]
    records: List[HistoryRecord]

class HistorySummary(BaseModel):
    total_records: int
    total_arrivals: float
    mean_monthly_arrivals: float
    min_arrivals: float
    max_arrivals: float
    peak_date: str
    shock_months_count: int
    stable_months_count: int

class ShapFeature(BaseModel):
    rank: int
    feature: str
    mean_abs_shap: float
    description: str

class ShapImportanceResponse(BaseModel):
    model_architecture: str
    total_features: int
    rankings: List[ShapFeature]

class ModelMetricRow(BaseModel):
    model: str
    regime: str
    n_samples: int
    rmse: float
    mae: float
    mape_pct: float
    r2_score: float

class ModelInfoResponse(BaseModel):
    active_production_model: str
    status: str
    input_features_count: int
    features: List[str]
    training_period: str
    test_period: str
    candidate_ranking: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    api_version: str
    production_model: str
    models_loaded: Dict[str, bool]
