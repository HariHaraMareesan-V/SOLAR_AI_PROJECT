"""
models_dl.py
LSTM, Bidirectional LSTM, and lightweight Transformer for solar wind forecasting.
Includes Monte Carlo Dropout for uncertainty estimation.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

LSTM_PATH    = os.path.join(MODEL_DIR, "lstm_model.keras")
BILSTM_PATH  = os.path.join(MODEL_DIR, "bilstm_model.keras")
TRANS_PATH   = os.path.join(MODEL_DIR, "transformer_model.keras")


# ── Positional Encoding (for Transformer) ────────────────────────────────────

class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        positions = np.arange(seq_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.encoding = tf.cast(angles[np.newaxis, :, :], tf.float32)

    def call(self, x):
        return x + self.encoding[:, : tf.shape(x)[1], :]


# ── LSTM ──────────────────────────────────────────────────────────────────────

def build_lstm(
    seq_len: int,
    n_features: int,
    horizon: int = 1,
    units: list[int] = [128, 64],
    dropout: float = 0.2,
) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_features))
    x = inp
    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        x = layers.LSTM(u, return_sequences=return_seq, dropout=dropout)(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon)(x)
    model = keras.Model(inp, out, name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ── Bidirectional LSTM ────────────────────────────────────────────────────────

def build_bilstm(
    seq_len: int,
    n_features: int,
    horizon: int = 1,
    units: list[int] = [128, 64],
    dropout: float = 0.2,
) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_features))
    x = inp
    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        x = layers.Bidirectional(
            layers.LSTM(u, return_sequences=return_seq, dropout=dropout)
        )(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon)(x)
    model = keras.Model(inp, out, name="BiLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ── Lightweight Transformer ───────────────────────────────────────────────────

def build_transformer(
    seq_len: int,
    n_features: int,
    horizon: int = 1,
    d_model: int = 64,
    n_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.1,
) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_features))

    # Project to d_model
    x = layers.Dense(d_model)(inp)
    x = PositionalEncoding(seq_len, d_model)(x)

    # Two Transformer encoder blocks
    for _ in range(2):
        attn_out = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(attn_out))
        ff_out = layers.Dense(ff_dim, activation="gelu")(x)
        ff_out = layers.Dense(d_model)(ff_out)
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(ff_out))

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon)(x)

    model = keras.Model(inp, out, name="Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ── Training utility ──────────────────────────────────────────────────────────

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    save_path: str = None,
) -> keras.callbacks.History:

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if save_path:
        model.save(save_path)

    return history


def load_lstm():
    return keras.models.load_model(LSTM_PATH, custom_objects={"PositionalEncoding": PositionalEncoding})


def load_bilstm():
    return keras.models.load_model(BILSTM_PATH, custom_objects={"PositionalEncoding": PositionalEncoding})


def load_transformer():
    return keras.models.load_model(TRANS_PATH, custom_objects={"PositionalEncoding": PositionalEncoding})


# ── Monte Carlo Dropout inference ─────────────────────────────────────────────

def mc_dropout_predict(
    model: keras.Model,
    X: np.ndarray,
    n_passes: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MC Dropout: keep dropout active at inference time.
    Returns mean and std of predictions across passes.
    """
    preds = np.stack(
        [model(X, training=True).numpy() for _ in range(n_passes)], axis=0
    )
    return preds.mean(axis=0), preds.std(axis=0)


def confidence_from_mc(std: np.ndarray) -> np.ndarray:
    """Convert MC Dropout std to a [0,1] confidence score."""
    max_std = std.max() + 1e-8
    return np.clip(1.0 - std / max_std, 0, 1)


# ── Multi-step recursive forecast ────────────────────────────────────────────

def dl_multistep_forecast(
    model: keras.Model,
    last_seq: np.ndarray,
    n_steps: int = 12,
    speed_idx: int = 0,
) -> np.ndarray:
    """
    Recursive multi-step forecast with a sequence model.
    Args:
        last_seq: (1, seq_len, n_features)
        n_steps: Forecast horizon
        speed_idx: Column index of 'speed' in the feature set
    Returns:
        np.ndarray (n_steps,) of predicted scaled speeds
    """
    seq = last_seq.copy()
    preds = []

    for _ in range(n_steps):
        pred = model.predict(seq, verbose=0)  # (1, horizon) or (1,)
        next_val = float(pred[0, 0] if pred.ndim == 2 else pred[0])
        preds.append(next_val)

        # Shift sequence window forward
        new_step = seq[0, -1, :].copy()
        new_step[speed_idx] = next_val
        seq = np.concatenate([seq[:, 1:, :], new_step.reshape(1, 1, -1)], axis=1)

    return np.array(preds)


if __name__ == "__main__":
    from preprocessing import run_pipeline

    data = run_pipeline(seq_len=20, horizon=1)
    X_tr = data["X_seq_train"]
    y_tr = data["y_seq_train"]
    X_te = data["X_seq_test"]
    y_te = data["y_seq_test"]

    seq_len, n_features = X_tr.shape[1], X_tr.shape[2]
    split = int(0.9 * len(X_tr))

    print("Building LSTM...")
    lstm = build_lstm(seq_len, n_features)
    train_model(lstm, X_tr[:split], y_tr[:split], X_tr[split:], y_tr[split:],
                epochs=20, save_path=LSTM_PATH)

    print("Building BiLSTM...")
    bilstm = build_bilstm(seq_len, n_features)
    train_model(bilstm, X_tr[:split], y_tr[:split], X_tr[split:], y_tr[split:],
                epochs=20, save_path=BILSTM_PATH)

    mean_preds, std_preds = mc_dropout_predict(lstm, X_te[:10])
    conf = confidence_from_mc(std_preds)
    print("MC Dropout confidence (first 5):", conf[:5].ravel())