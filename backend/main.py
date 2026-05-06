"""
main.py — FastAPI backend (fixed CORS + health endpoint + graceful model loading)
"""

import os, time, logging, threading, numpy as np
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Local imports ─────────────────────────────────────────────────────────────
from data_logger      import get_latest_rows, run_logger
from preprocessing    import run_pipeline, inverse_scale_speed, FEATURE_COLS, clean, normalize
from feature_engineering import add_features
from ensemble         import run_ensemble, EnsembleResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
STATE: dict = {
    "pipeline":  None,
    "rf":        None,
    "xgb":       None,
    "lstm":      None,
    "metrics":   None,
    "logs":      [],
    "last_alert": None,
}
ALERT_HISTORY: list = []
SEQ_LEN = 20


def _log(msg: str):
    entry = {"time": datetime.now(timezone.utc).isoformat(), "message": msg}
    STATE["logs"].append(entry)
    if len(STATE["logs"]) > 200:
        STATE["logs"].pop(0)
    logger.info(msg)


def _load_models():
    """Try to load all models — log clearly if missing."""
    _log("Loading preprocessing pipeline...")
    try:
        STATE["pipeline"] = run_pipeline(seq_len=SEQ_LEN, horizon=1)
        _log("Pipeline ready.")
    except Exception as e:
        _log(f"Pipeline error: {e}")

    model_loaders = [
        ("Random Forest", "models_ml",  "load_rf",   "rf"),
        ("XGBoost",       "models_ml",  "load_xgb",  "xgb"),
        ("LSTM",          "models_dl",  "load_lstm",  "lstm"),
    ]
    for name, module_name, fn_name, key in model_loaders:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            STATE[key] = getattr(mod, fn_name)()
            _log(f"{name} loaded OK.")
        except Exception as e:
            STATE[key] = None
            _log(f"WARNING: {name} not loaded — {e}. Run python {module_name}.py first.")


def _compute_metrics():
    """Compute metrics in background — only if all models loaded."""
    p = STATE["pipeline"]
    if not (p and STATE["rf"] and STATE["xgb"] and STATE["lstm"]):
        _log("Skipping metrics — models not ready.")
        return
    try:
        from evaluation import full_evaluation
        from models_dl  import mc_dropout_predict

        X_te, y_te   = p["X_flat_test"],  p["y_flat_test"]
        X_seq, y_seq = p["X_seq_test"],   p["y_seq_test"].ravel()

        rf_preds  = STATE["rf"].predict(X_te)
        xgb_preds = STATE["xgb"].predict(X_te)
        n = min(len(X_te), len(X_seq))
        lstm_preds, _ = mc_dropout_predict(STATE["lstm"], X_seq[:n], n_passes=10)
        lstm_preds = lstm_preds.ravel()
        ens_preds  = 0.30*rf_preds[:n] + 0.35*xgb_preds[:n] + 0.35*lstm_preds

        STATE["metrics"] = full_evaluation(
            y_te[:n], rf_preds[:n], xgb_preds[:n], lstm_preds, ens_preds
        )
        _log("Metrics computed.")
    except Exception as e:
        _log(f"Metrics error: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    threading.Thread(target=_compute_metrics, daemon=True).start()
    threading.Thread(target=run_logger, kwargs={"simulate": False}, daemon=True).start()
    _log("Server started.")
    yield
    _log("Server shutdown.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Solar Wind Prediction API", version="1.0.0", lifespan=lifespan)

# CRITICAL: CORS must allow all origins so React (port 3000) can reach FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    timestamp:            str
    current_speed:        float
    rf_prediction:        Optional[float]
    xgb_prediction:       Optional[float]
    lstm_prediction:      Optional[float]
    ensemble_prediction:  Optional[float]
    confidence:           float
    storm_level:          str
    storm_probability:    float
    ensemble_weights:     dict
    alert:                Optional[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — React frontend uses this to detect backend status."""
    return {
        "status": "ok",
        "models_loaded": {
            "rf":   STATE["rf"]   is not None,
            "xgb":  STATE["xgb"]  is not None,
            "lstm": STATE["lstm"] is not None,
        },
        "pipeline_ready": STATE["pipeline"] is not None,
    }


@app.get("/solar-data")
def solar_data(n: int = 60):
    df = get_latest_rows(n=n)
    if df.empty:
        return []
    return df.fillna(0).to_dict(orient="records")


@app.get("/prediction", response_model=PredictionResponse)
def prediction():
    p = STATE["pipeline"]
    if p is None:
        raise HTTPException(503, "Pipeline not ready. Check backend logs.")

    # Check if any model is loaded
    any_model = STATE["rf"] or STATE["xgb"] or STATE["lstm"]
    if not any_model:
        raise HTTPException(503, detail=(
            "No models loaded. Run: python models_ml.py && python models_dl.py "
            "in your backend folder, then restart the server."
        ))

    # Load and preprocess data
    df_raw = get_latest_rows(n=SEQ_LEN + 60)
    if len(df_raw) < SEQ_LEN + 5:
        raise HTTPException(503, "Not enough data yet. Wait 30 seconds for data logger.")

    df_c = clean(df_raw)
    df_s = normalize(df_c, p["scaler"])
    df_fe = add_features(df_s).dropna().reset_index(drop=True)
    feat_cols = [c for c in df_fe.columns if c not in ["timestamp", "speed"]]
    X_flat = df_fe[feat_cols].values

    last_row = X_flat[-1:]
    last_seq = df_s[FEATURE_COLS].values[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))
    latest   = df_c.iloc[-1]
    speed_kms = float(latest["speed"])
    bz_nt     = float(latest["bz"])

    # Run available models (return 0.0 if model not loaded)
    rf_pred = 0.0
    if STATE["rf"]:
        try:
            rf_pred = float(STATE["rf"].predict(last_row)[0])
        except Exception as e:
            _log(f"RF inference error: {e}")

    xgb_pred = 0.0
    if STATE["xgb"]:
        try:
            xgb_pred = float(STATE["xgb"].predict(last_row)[0])
        except Exception as e:
            _log(f"XGB inference error: {e}")

    lstm_pred, lstm_std_val = 0.0, 0.0
    if STATE["lstm"]:
        try:
            from models_dl import mc_dropout_predict
            lm, ls = mc_dropout_predict(STATE["lstm"], last_seq, n_passes=20)
            lstm_pred    = float(lm[0, 0] if lm.ndim == 2 else lm[0])
            lstm_std_val = float(ls[0, 0] if ls.ndim == 2 else ls[0])
        except Exception as e:
            _log(f"LSTM inference error: {e}")

    # Ensemble
    result: EnsembleResult = run_ensemble(
        rf_pred=rf_pred, xgb_pred=xgb_pred, lstm_pred=lstm_pred,
        lstm_std=lstm_std_val, speed_kms=speed_kms, bz_nt=bz_nt,
        scaler=p["scaler"], use_dynamic_weights=True,
    )

    # Alert logic
    alert = None
    now = time.time()
    prev = STATE.get("last_alert")
    if result.storm_level in ("HIGH", "EXTREME"):
        if not prev or now - prev["time"] > 10 or prev["level"] != result.storm_level:
            alert = f"STORM ALERT: {result.storm_level} — Prob={result.storm_probability:.0%}"
            STATE["last_alert"] = {"time": now, "level": result.storm_level}
            ALERT_HISTORY.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                                   "level": result.storm_level, "message": alert})
            _log(alert)

    # Inverse-scale to get real km/s values
    def to_kms(v):
        if v == 0.0:
            return None
        try:
            return float(inverse_scale_speed(np.array([v]), p["scaler"])[0])
        except:
            return None

    return PredictionResponse(
        timestamp=str(latest.get("timestamp", datetime.now(timezone.utc).isoformat())),
        current_speed=round(speed_kms, 1),
        rf_prediction=to_kms(rf_pred),
        xgb_prediction=to_kms(xgb_pred),
        lstm_prediction=to_kms(lstm_pred),
        ensemble_prediction=round(result.prediction, 1),
        confidence=result.confidence,
        storm_level=result.storm_level,
        storm_probability=result.storm_probability,
        ensemble_weights=result.weights,
        alert=alert,
    )


@app.get("/multistep")
def multistep(horizon: int = 12, background_tasks: BackgroundTasks = None):
    if horizon > 24:
        raise HTTPException(400, "Max horizon is 24.")
    p = STATE["pipeline"]
    if not p:
        raise HTTPException(503, "Pipeline not ready.")

    df_raw = get_latest_rows(n=SEQ_LEN + 30)
    if len(df_raw) < SEQ_LEN + 5:
        raise HTTPException(503, "Not enough data yet.")

    df_c  = clean(df_raw)
    df_s  = normalize(df_c, p["scaler"])
    df_fe = add_features(df_s).dropna().reset_index(drop=True)
    feat_cols = [c for c in df_fe.columns if c not in ["timestamp", "speed"]]
    last_row  = df_fe[feat_cols].values[-1:]
    last_seq  = df_s[FEATURE_COLS].values[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))

    from models_ml import recursive_forecast
    rf_fc  = recursive_forecast(STATE["rf"],  last_row, n_steps=horizon) if STATE["rf"]  else [0]*horizon
    xgb_fc = recursive_forecast(STATE["xgb"], last_row, n_steps=horizon) if STATE["xgb"] else [0]*horizon

    lstm_fc = [0]*horizon
    if STATE["lstm"]:
        from models_dl import dl_multistep_forecast
        lstm_fc = dl_multistep_forecast(STATE["lstm"], last_seq, n_steps=horizon).tolist()

    def inv_arr(arr):
        try:
            return [round(float(inverse_scale_speed(np.array([v]), p["scaler"])[0]), 1) for v in arr]
        except:
            return list(arr)

    rf_kms   = inv_arr(rf_fc)
    xgb_kms  = inv_arr(xgb_fc)
    lstm_kms = inv_arr(lstm_fc)
    ens_kms  = [round(0.30*r + 0.35*x + 0.35*l, 1) for r, x, l in zip(rf_kms, xgb_kms, lstm_kms)]

    return {
        "horizon": horizon,
        "rf_forecast":       rf_kms,
        "xgb_forecast":      xgb_kms,
        "lstm_forecast":     lstm_kms,
        "ensemble_forecast": ens_kms,
    }


@app.get("/metrics")
def metrics(background_tasks: BackgroundTasks):
    if STATE["metrics"] is None:
        background_tasks.add_task(_compute_metrics)
        raise HTTPException(202, "Metrics being computed. Retry in 10 seconds.")
    return {"models": STATE["metrics"], "computed_at": datetime.now(timezone.utc).isoformat()}


@app.get("/logs")
def logs(n: int = 50):
    return {"logs": STATE["logs"][-n:]}


@app.get("/alerts")
def alerts():
    return {"alerts": ALERT_HISTORY[-20:]}


@app.get("/feature-importance")
def feature_importance():
    from models_ml import rf_feature_importance, xgb_feature_importance
    p = STATE["pipeline"]
    if not p or not STATE["rf"]:
        raise HTTPException(503, "RF model not loaded.")
    feat_names = p.get("feature_names", [])
    return {
        "random_forest": rf_feature_importance(STATE["rf"], feat_names),
        "xgboost":       xgb_feature_importance(STATE["xgb"], feat_names) if STATE["xgb"] else {},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)