"""
feature_engineering.py
Creates lag, rolling, and physics-motivated features for ML models.
"""

import numpy as np
import pandas as pd

LAG_STEPS = 5
ROLL_WINDOWS = [3, 5, 10]
FEATURE_COLS = ["speed", "density", "bz", "bt", "temperature"]


def add_lag_features(df: pd.DataFrame, cols: list[str], n_lags: int = LAG_STEPS) -> pd.DataFrame:
    for col in cols:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str], windows: list[int] = ROLL_WINDOWS) -> pd.DataFrame:
    for col in cols:
        for w in windows:
            df[f"{col}_rmean{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_rstd{w}"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
    return df


def add_delta_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """First-order differences — captures rate of change."""
    for col in cols:
        df[f"{col}_delta"] = df[col].diff().fillna(0)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-informed interaction terms.
    - Solar wind dynamic pressure proxy: density * speed²
    - Bz * speed: proxy for storm driving energy
    """
    df["dyn_pressure"] = df["density"] * df["speed"] ** 2
    df["bz_speed"]     = df["bz"] * df["speed"]
    df["bt_ratio"]     = df["bz"].abs() / (df["bt"].abs() + 1e-6)  # Bz/Bt coupling
    return df


def add_features(df: pd.DataFrame, cols: list[str] = None) -> pd.DataFrame:
    """
    Master feature engineering function.
    Expects df to already be scaled (or at least cleaned).
    Returns df with all engineered features appended.
    """
    if cols is None:
        cols = [c for c in FEATURE_COLS if c in df.columns]

    df = df.copy()
    df = add_lag_features(df, cols)
    df = add_rolling_features(df, cols)
    df = add_delta_features(df, cols)
    df = add_interaction_features(df)

    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return all engineered feature column names (excludes timestamp and target)."""
    return [c for c in df.columns if c not in ["timestamp", "speed"]]


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Quick smoke test
    mock = pd.DataFrame({
        "speed":       np.random.uniform(350, 600, 100),
        "density":     np.random.uniform(3, 10, 100),
        "bz":          np.random.uniform(-15, 5, 100),
        "bt":          np.random.uniform(2, 20, 100),
        "temperature": np.random.uniform(50000, 150000, 100),
    })
    out = add_features(mock)
    print(f"Input cols : {mock.shape[1]}")
    print(f"Output cols: {out.shape[1]}")
    print(out.tail(3).to_string())