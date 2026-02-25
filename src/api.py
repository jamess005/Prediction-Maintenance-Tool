"""
FastAPI inference endpoint for the predictive maintenance model.

Endpoints
---------
POST /predict        — predict failure probability for a single reading
POST /predict/batch  — predict for multiple readings
POST /retrain        — retrain the model from the CSV dataset
GET  /health         — liveness check
GET  /model/info     — current model metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import engineer_features, get_feature_columns, get_target_column

ROOT       = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT / "outputs" / "models"
DATA_PATH  = ROOT / "data" / "ai4i2020.csv"

app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time machine failure prediction using a Random Forest model "
                "trained on the AI4I 2020 dataset.",
    version="1.0.0",
)

# ── Global model state ──────────────────────────────────────────────────────
_model = None
_threshold: float = 0.5
_model_type: str = "unknown"


def _load_model() -> None:
    """Load the best available model (RF preferred, XGB fallback)."""
    global _model, _threshold, _model_type

    rf_path  = MODEL_DIR / "rf_model.pkl"
    xgb_path = MODEL_DIR / "xgb_model.pkl"

    path = rf_path if rf_path.exists() else xgb_path
    if not path.exists():
        return

    saved = joblib.load(path)
    _model      = saved["model"]
    _threshold  = float(saved["threshold"])
    _model_type = saved.get("model_type", "unknown")


_load_model()


# ── Schemas ─────────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    """Raw sensor values from a single machine cycle."""
    air_temp_K:      float = Field(..., description="Air temperature [K]")
    proc_temp_K:     float = Field(..., description="Process temperature [K]")
    rot_speed_rpm:   float = Field(..., description="Rotational speed [rpm]")
    torque_Nm:       float = Field(..., description="Torque [Nm]")
    tool_wear_min:   float = Field(..., description="Tool wear [min]")
    product_type:    Literal["L", "M", "H"] = Field(..., description="Product quality type")


class PredictionResponse(BaseModel):
    failure_probability: float
    failure_predicted:   bool
    threshold:           float
    model_type:          str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class RetrainResponse(BaseModel):
    status:     str
    model_type: str
    threshold:  float
    mcc_val:    float
    mcc_test:   float


class ModelInfo(BaseModel):
    model_type: str
    threshold:  float
    features:   list[str]
    loaded:     bool


# ── Helpers ─────────────────────────────────────────────────────────────────
_TYPE_MAP = {"L": 0, "M": 1, "H": 2}


def _reading_to_features(r: SensorReading) -> pd.DataFrame:
    """Convert a raw sensor reading into the 9-feature model input."""
    row = {
        "air_temp_K":    r.air_temp_K,
        "proc_temp_K":   r.proc_temp_K,
        "rot_speed_rpm": r.rot_speed_rpm,
        "torque_Nm":     r.torque_Nm,
        "tool_wear_min": r.tool_wear_min,
        "power_kW":      (r.torque_Nm * r.rot_speed_rpm) / 9550,
        "temp_delta_K":  r.proc_temp_K - r.air_temp_K,
        "torque_wear":   r.torque_Nm * r.tool_wear_min,
        "product_type":  _TYPE_MAP[r.product_type],
    }
    return pd.DataFrame([row])[get_feature_columns()]


def _predict_single(r: SensorReading) -> PredictionResponse:
    if _model is None:
        raise HTTPException(503, "No model loaded. Train one first via POST /retrain")
    X = _reading_to_features(r)
    proba = float(_model.predict_proba(X)[:, 1][0])
    return PredictionResponse(
        failure_probability=round(proba, 4),
        failure_predicted=proba >= _threshold,
        threshold=_threshold,
        model_type=_model_type,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        model_type=_model_type,
        threshold=_threshold,
        features=get_feature_columns(),
        loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    """Predict failure probability for a single sensor reading."""
    return _predict_single(reading)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(readings: list[SensorReading]):
    """Predict failure probability for multiple sensor readings."""
    if not readings:
        raise HTTPException(400, "Empty batch")
    return BatchPredictionResponse(predictions=[_predict_single(r) for r in readings])


@app.post("/retrain", response_model=RetrainResponse)
def retrain(model_type: str = "rf"):
    """
    Retrain the model from the dataset on disk.

    Query param `model_type`: 'rf' (default) or 'xgb'.
    """
    from pipeline import run  # deferred to avoid circular imports at startup

    try:
        run(model_type=model_type)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")

    _load_model()

    # Report metrics from the freshly saved model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import matthews_corrcoef

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["UDI", "Product ID"], errors="ignore")
    df = engineer_features(df)
    X = df[get_feature_columns()]
    y = df[get_target_column()]

    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    assert _model is not None, "Model failed to load after retrain"
    val_pred  = (_model.predict_proba(X_val)[:, 1]  >= _threshold).astype(int)
    test_pred = (_model.predict_proba(X_test)[:, 1] >= _threshold).astype(int)

    return RetrainResponse(
        status="ok",
        model_type=_model_type,
        threshold=_threshold,
        mcc_val=round(float(matthews_corrcoef(y_val, val_pred)), 4),
        mcc_test=round(float(matthews_corrcoef(y_test, test_pred)), 4),
    )
