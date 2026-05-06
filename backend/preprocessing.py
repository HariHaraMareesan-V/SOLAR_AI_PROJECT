"""
preprocessing.py
Full preprocessing pipeline: cleaning, normalization, sequence creation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "solar_data.csv")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

FEATURE_COLS = ["speed", "density", "bz", "bt", "temperature"]
TARGET_COL = "speed"
SEQ_LEN = 20  # lookback window for LSTM


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sentinel values, interpolate gaps, drop duplicates."""
    df = df.replace(-9999.99, np.nan)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Interpolate short gaps; forward/back-fill edges
    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .interpolate(method="linear", limit=5)
        .ffill()
        .bfill()
    )

    # Physical bounds filtering
    df = df[df["speed"].between(200, 1200)]
    df = df[df["density"].between(0.1, 100)]
    df = df[df["bz"].between(-80, 40)]

    return df.reset_index(drop=True)


def fit_scaler(df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURE_COLS])
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_scaler() -> MinMaxScaler:
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    raise FileNotFoundError("Scaler not found. Run preprocessing first.")


def normalize(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    scaled = scaler.transform(df[FEATURE_COLS])
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaled
    return df_scaled


def inverse_scale_speed(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse transform speed-only predictions back to km/s."""
    speed_idx = FEATURE_COLS.index(TARGET_COL)
    dummy = np.zeros((len(values), len(FEATURE_COLS)))
    dummy[:, speed_idx] = values
    return scaler.inverse_transform(dummy)[:, speed_idx]


def make_sequences(
    df_scaled: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for LSTM.
    Returns:
        X: (samples, seq_len, n_features)
        y: (samples, horizon)  — target is scaled speed
    """
    data = df_scaled[FEATURE_COLS].values
    speed_idx = FEATURE_COLS.index(TARGET_COL)

    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + horizon, speed_idx])

    return np.array(X), np.array(y)


def make_flat_features(df_scaled: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build flat feature matrix for ML models (after feature engineering).
    Call after feature_engineering.add_features().
    Returns X, y as numpy arrays.
    """
    from feature_engineering import add_features

    df_fe = add_features(df_scaled)
    feature_cols = [c for c in df_fe.columns if c not in ["timestamp", TARGET_COL]]
    df_fe = df_fe.dropna().reset_index(drop=True)

    X = df_fe[feature_cols].values
    y = df_fe[TARGET_COL].values
    return X, y, feature_cols


def run_pipeline(seq_len: int = SEQ_LEN, horizon: int = 1):
    """
    End-to-end preprocessing. Returns everything models need.
    """
    df_raw = load_raw()
    df_clean = clean(df_raw)

    scaler = fit_scaler(df_clean)
    df_scaled = normalize(df_clean, scaler)

    X_seq, y_seq = make_sequences(df_scaled, seq_len=seq_len, horizon=horizon)
    X_flat, y_flat, feat_names = make_flat_features(df_scaled)

    split = int(0.8 * len(X_seq))

    return {
        "scaler": scaler,
        "df_clean": df_clean,
        "df_scaled": df_scaled,
        "X_seq_train": X_seq[:split],
        "X_seq_test": X_seq[split:],
        "y_seq_train": y_seq[:split],
        "y_seq_test": y_seq[split:],
        "X_flat_train": X_flat[:split],
        "X_flat_test": X_flat[split:],
        "y_flat_train": y_flat[:split],
        "y_flat_test": y_flat[split:],
        "feature_names": feat_names,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Seq train shape : {results['X_seq_train'].shape}")
    print(f"Flat train shape: {results['X_flat_train'].shape}")