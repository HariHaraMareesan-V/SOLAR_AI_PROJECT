"""
evaluation.py
Model evaluation: MAE, RMSE, R², comparison table, and plotting utilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Core metrics ──────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(mean_absolute_error(y_true.ravel(), y_pred.ravel())), 4)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))), 4)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(r2_score(y_true.ravel(), y_pred.ravel())), 4)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Model") -> dict:
    return {
        "model": label,
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "R2":    r2(y_true, y_pred),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Args:
        results: List of dicts from evaluate()
    Returns:
        DataFrame sorted by RMSE ascending
    """
    df = pd.DataFrame(results).set_index("model")
    df = df.sort_values("RMSE")
    return df


def print_comparison(results: list[dict]):
    df = comparison_table(results)
    print("\n" + "=" * 45)
    print("        MODEL PERFORMANCE COMPARISON")
    print("=" * 45)
    print(df.to_string())
    print("=" * 45 + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str = "Actual vs Predicted Solar Wind Speed",
    filename: str = "actual_vs_predicted.png",
    n_points: int = 200,
):
    y_true = y_true.ravel()[:n_points]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label="Actual", color="black", linewidth=1.5, zorder=5)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(pred.ravel()[:n_points], label=name, color=colors[i % len(colors)],
                linestyle="--", linewidth=1.2, alpha=0.85)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Solar Wind Speed (scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_error_distribution(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    filename: str = "error_distribution.png",
):
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4), sharey=True)
    if len(predictions) == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    y_true = y_true.ravel()

    for ax, (name, pred), color in zip(axes, predictions.items(), colors):
        errors = y_true - pred.ravel()
        ax.hist(errors, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"{name}\nμ={errors.mean():.3f}  σ={errors.std():.3f}", fontsize=10)
        ax.set_xlabel("Error")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Frequency")
    plt.suptitle("Prediction Error Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_forecast_horizon(
    forecasts: dict[str, np.ndarray],
    filename: str = "multistep_forecast.png",
):
    """Plot multi-step forecast for each model."""
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    x = np.arange(1, list(forecasts.values())[0].shape[0] + 1)

    for (name, pred), color in zip(forecasts.items(), colors):
        ax.plot(x, pred.ravel(), marker="o", label=name, color=color, linewidth=1.5)

    ax.set_title("Multi-Step Forecast (t+1 … t+12)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("Predicted Speed (scaled)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── Full evaluation run ───────────────────────────────────────────────────────

def full_evaluation(
    y_true: np.ndarray,
    rf_preds: np.ndarray,
    xgb_preds: np.ndarray,
    lstm_preds: np.ndarray,
    ensemble_preds: np.ndarray,
) -> dict:
    results = [
        evaluate(y_true, rf_preds,       "Random Forest"),
        evaluate(y_true, xgb_preds,      "XGBoost"),
        evaluate(y_true, lstm_preds,     "LSTM"),
        evaluate(y_true, ensemble_preds, "Ensemble"),
    ]

    print_comparison(results)

    predictions = {
        "RF":       rf_preds,
        "XGBoost":  xgb_preds,
        "LSTM":     lstm_preds,
        "Ensemble": ensemble_preds,
    }

    plot_actual_vs_predicted(y_true, predictions)
    plot_error_distribution(y_true, predictions)

    return {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in results}