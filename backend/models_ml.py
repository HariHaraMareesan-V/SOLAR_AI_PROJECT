"""
models_ml.py
Random Forest and XGBoost regressors for solar wind speed prediction.
Supports single-step and recursive multi-step forecasting.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR,"saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

RF_PATH  = os.path.join(MODEL_DIR, "random_forest.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")


# ── Random Forest ─────────────────────────────────────────────────────────────

def build_rf(n_estimators: int = 200, max_depth: int = 12) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Train RF. y_train can be 1-D (single step) or 2-D (multi-step)."""
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(build_rf(), n_jobs=-1)
    else:
        model = build_rf()
        y_train = y_train.ravel()

    model.fit(X_train, y_train)
    joblib.dump(model, RF_PATH)
    return model


def load_rf() -> RandomForestRegressor:
    return joblib.load(RF_PATH)


def rf_feature_importance(model, feature_names: list[str]) -> dict:
    """Extract feature importances from trained RF."""
    if isinstance(model, MultiOutputRegressor):
        importances = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
    else:
        importances = model.feature_importances_
    return dict(sorted(zip(feature_names, importances.tolist()), key=lambda x: -x[1]))


# ── XGBoost ───────────────────────────────────────────────────────────────────

def build_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(build_xgb(), n_jobs=-1)
    else:
        model = build_xgb()
        y_train = y_train.ravel()

    model.fit(X_train, y_train)
    joblib.dump(model, XGB_PATH)
    return model


def load_xgb():
    return joblib.load(XGB_PATH)


def xgb_feature_importance(model, feature_names: list[str]) -> dict:
    if isinstance(model, MultiOutputRegressor):
        scores = np.mean(
            [e.feature_importances_ for e in model.estimators_], axis=0
        )
    else:
        scores = model.feature_importances_
    return dict(sorted(zip(feature_names, scores.tolist()), key=lambda x: -x[1]))


# ── Recursive Multi-Step Forecasting ─────────────────────────────────────────

def recursive_forecast(
    model,
    last_flat_row: np.ndarray,
    n_steps: int = 12,
) -> np.ndarray:
    """
    Recursively predict n_steps into the future using a flat ML model.
    Appends each prediction as the new 'speed' for the next step.
    This is a simplified approximation — a full implementation would
    recompute all lag/rolling features per step.

    Args:
        model: Trained RF or XGB
        last_flat_row: (1, n_features) array of the latest feature row
        n_steps: Forecast horizon

    Returns:
        np.ndarray of shape (n_steps,) — predicted scaled speeds
    """
    preds = []
    row = last_flat_row.copy().reshape(1, -1)

    for _ in range(n_steps):
        pred = model.predict(row)[0]
        if np.ndim(pred) > 0:
            pred = float(pred[0])
        preds.append(float(pred))
        # Shift the first feature (speed-like) by the new prediction
        row[0, 0] = pred

    return np.array(preds)


# ── Confidence via RF variance ────────────────────────────────────────────────

def rf_confidence(rf_model, X: np.ndarray) -> np.ndarray:
    """
    Estimate uncertainty as std of individual tree predictions.
    Lower std → higher confidence.
    Returns confidence in [0, 1] range.
    """
    if isinstance(rf_model, MultiOutputRegressor):
        # Use first output estimator
        estimators = [e for e in rf_model.estimators_]
    else:
        estimators = rf_model.estimators_

    tree_preds = np.array([e.predict(X) for e in estimators])
    std_dev = tree_preds.std(axis=0)

    # Normalize: low std → high confidence
    max_std = std_dev.max() + 1e-8
    confidence = 1.0 - (std_dev / max_std)
    return np.clip(confidence, 0, 1)


if __name__ == "__main__":
    import numpy as np
    from preprocessing import run_pipeline

    data = run_pipeline()
    X_tr, X_te = data["X_flat_train"], data["X_flat_test"]
    y_tr, y_te = data["y_flat_train"], data["y_flat_test"]
    feat_names = data["feature_names"]

    print("Training Random Forest...")
    rf = train_rf(X_tr, y_tr)
    print("RF OOB R²:", round(rf.oob_score_ if hasattr(rf, "oob_score_") else 0, 4))

    print("Training XGBoost...")
    xgb_model = train_xgb(X_tr, y_tr)
    print("XGBoost trained.")

    preds_rf = rf.predict(X_te[:5])
    print("RF sample preds:", preds_rf)

    fi = rf_feature_importance(rf, feat_names)
    print("Top 5 features:", list(fi.items())[:5])