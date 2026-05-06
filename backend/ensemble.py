"""
ensemble.py
Weighted ensemble combining RF, XGBoost, and LSTM predictions.
Supports dynamic weight adjustment based on recent model errors.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EnsembleResult:
    prediction: float
    confidence: float
    weights: dict
    model_predictions: dict
    storm_probability: float
    storm_level: str


DEFAULT_WEIGHTS = {"rf": 0.5, "xgb": 0.4, "lstm": 0.1}

# Running error tracker for dynamic weighting
_error_history: dict[str, list] = {"rf": [], "xgb": [], "lstm": []}
MAX_HISTORY = 50


def update_error_history(model: str, error: float):
    _error_history[model].append(abs(error))
    if len(_error_history[model]) > MAX_HISTORY:
        _error_history[model].pop(0)


def compute_dynamic_weights() -> dict:
    """Inverse-MAE dynamic weighting. Falls back to defaults if no history."""
    means = {k: np.mean(v) if v else None for k, v in _error_history.items()}
    if any(v is None for v in means.values()):
        return DEFAULT_WEIGHTS.copy()

    inv = {k: 1.0 / (v + 1e-6) for k, v in means.items()}
    total = sum(inv.values())
    return {k: round(v / total, 4) for k, v in inv.items()}


def weighted_ensemble(
    rf_pred: float,
    xgb_pred: float,
    lstm_pred: float,
    weights: dict = None,
) -> tuple[float, dict]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    final = (
        weights["rf"]   * rf_pred +
        weights["xgb"]  * xgb_pred +
        weights["lstm"] * lstm_pred
    )
    return round(float(final), 4), weights


def compute_confidence(
    rf_pred: float,
    xgb_pred: float,
    lstm_pred: float,
    lstm_std: float = 0.0,
) -> float:
    """
    Confidence = 1 - normalized spread among model predictions.
    Also penalises high MC-Dropout uncertainty (lstm_std).
    """
    preds = np.array([rf_pred, xgb_pred, lstm_pred])
    spread = preds.std()
    mean   = preds.mean() + 1e-8

    # Coefficient of variation (scale-independent)
    cv = spread / abs(mean)
    spread_confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

    # MC Dropout penalty (normalised against a typical std of 0.1 in scaled space)
    mc_confidence = float(np.clip(1.0 - lstm_std / 0.1, 0.0, 1.0))

    # Combined score
    confidence = 0.6 * spread_confidence + 0.4 * mc_confidence
    return round(float(np.clip(confidence, 0.0, 1.0)), 4)


def classify_storm(
    speed_kms: float,
    bz_nt: float,
    ensemble_pred_kms: float,
) -> tuple[str, float]:
    """
    Rule-based + probabilistic storm classification.
    Returns (level, probability).

    Kp-proxy logic:
      EXTREME  : Bz < -20 or speed > 700
      HIGH     : Bz < -12 or speed > 600
      MODERATE : Bz < -5  or speed > 500
      LOW      : otherwise
    """
    score = 0.0

    # Bz contribution (negative Bz drives geomagnetic storms)
    if bz_nt < -20:
        score += 0.55
    elif bz_nt < -12:
        score += 0.35
    elif bz_nt < -5:
        score += 0.15

    # Speed contribution
    if speed_kms > 700:
        score += 0.45
    elif speed_kms > 600:
        score += 0.30
    elif speed_kms > 500:
        score += 0.15

    # Forecast trend nudge
    if ensemble_pred_kms > speed_kms + 50:
        score += 0.05

    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.75:
        level = "EXTREME"
    elif score >= 0.50:
        level = "HIGH"
    elif score >= 0.25:
        level = "MODERATE"
    else:
        level = "LOW"

    return level, round(score, 4)


def run_ensemble(
    rf_pred: float,
    xgb_pred: float,
    lstm_pred: float,
    lstm_std: float,
    speed_kms: float,
    bz_nt: float,
    scaler=None,
    use_dynamic_weights: bool = True,
) -> EnsembleResult:
    """
    Master ensemble function called by main.py.

    Args:
        rf_pred, xgb_pred, lstm_pred : scaled predictions [0,1]
        lstm_std                      : MC-Dropout std (scaled)
        speed_kms                     : current real speed (km/s) for storm logic
        bz_nt                         : current real Bz (nT)
        scaler                        : fitted MinMaxScaler (to invert predictions)
        use_dynamic_weights           : adapt weights from recent errors

    Returns EnsembleResult dataclass.
    """
    weights = compute_dynamic_weights() if use_dynamic_weights else DEFAULT_WEIGHTS

    final_scaled, weights = weighted_ensemble(rf_pred, xgb_pred, lstm_pred, weights)
    confidence = compute_confidence(rf_pred, xgb_pred, lstm_pred, lstm_std)

    # Inverse-scale to km/s if scaler provided
    def inv(val):
        if scaler is None:
            return val
        from preprocessing import inverse_scale_speed, FEATURE_COLS
        return float(inverse_scale_speed(np.array([val]), scaler)[0])

    final_kms   = inv(final_scaled)
    rf_kms      = inv(rf_pred)
    xgb_kms     = inv(xgb_pred)
    lstm_kms    = inv(lstm_pred)

    storm_level, storm_prob = classify_storm(speed_kms, bz_nt, final_kms)

    return EnsembleResult(
        prediction=round(final_kms, 2),
        confidence=confidence,
        weights=weights,
        model_predictions={
            "rf":   round(rf_kms, 2),
            "xgb":  round(xgb_kms, 2),
            "lstm": round(lstm_kms, 2),
        },
        storm_probability=storm_prob,
        storm_level=storm_level,
    )