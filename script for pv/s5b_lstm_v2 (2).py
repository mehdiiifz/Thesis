"""Stage 5b: LSTM v2 — fair comparison with the trees, and night masking.

Changes vs scripts/s5_lstm.py (kept intact as LSTM-v1):
  1. The static input gains H_sun(t+h) for each of the 8 horizons —
     deterministic astronomy, exactly the target-time information the tree
     models receive — MinMax-scaled with fit-portion-only stats.
  2. Night masking at prediction time: wherever H_sun(t+h) <= 0 the
     prediction is forced to 0 kW (holdout and test alike).
  3. Capacity doubled: LSTM(128) and Dense(128).
Everything else is identical to v1: Adam(1e-3), MSE, batch 256, <=60 epochs,
EarlyStopping(patience=6, restore_best_weights=True) on the stage-3 holdout,
seed 42, predictions clipped to [0, 1000] kW.

Outputs: results/single_lstm_v2.csv and data/lstm_preds.pkl (overwritten so
the stacking stage uses v2; committed).
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
random.seed(42)
np.random.seed(42)

import tensorflow as tf  # noqa: E402  (seeds must be set before TF loads)

tf.random.set_seed(42)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]
CHANNELS = ["P_kW", "G(i)", "T2m", "WS10m"]
WINDOW = 48
CAP_KW = 1000.0


def build_windows(df: pd.DataFrame, times: pd.DatetimeIndex, lo: pd.Series, hi: pd.Series) -> np.ndarray:
    scaled = ((df[CHANNELS] - lo) / (hi - lo)).to_numpy(dtype=np.float32)
    sw = np.lib.stride_tricks.sliding_window_view(scaled, WINDOW, axis=0)
    pos = df.index.get_indexer(times)
    return sw[pos - (WINDOW - 1)].transpose(0, 2, 1)


def static_vector(splits: dict, key: str, times: pd.DatetimeIndex,
                  hsun_lo: float, hsun_hi: float) -> np.ndarray:
    """Calendar sin/cos as in v1, plus scaled H_sun(t+h) for all 8 horizons."""
    cols = []
    for h in HORIZONS:
        target_hour = (times + pd.Timedelta(hours=h)).hour
        cols += [np.sin(2 * np.pi * target_hour / 24), np.cos(2 * np.pi * target_hour / 24)]
    doy = times.dayofyear
    cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    for h in HORIZONS:
        hsun = splits[h][key]["Hsun_target"].loc[times].to_numpy()
        cols.append((hsun - hsun_lo) / (hsun_hi - hsun_lo))
    return np.stack(cols, axis=1).astype(np.float32)


def build_model(static_dim: int) -> tf.keras.Model:
    seq_in = tf.keras.Input(shape=(WINDOW, len(CHANNELS)), name="past48h")
    cal_in = tf.keras.Input(shape=(static_dim,), name="static")
    x = tf.keras.layers.LSTM(128)(seq_in)
    x = tf.keras.layers.Concatenate()([x, cal_in])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(len(HORIZONS), activation="sigmoid")(x)
    model = tf.keras.Model([seq_in, cal_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    df = pd.read_pickle(ROOT / "data" / "clean.pkl")
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")

    s1 = splits[HORIZONS[0]]
    t_fit, t_hold, t_test = (s1[k].index for k in ["X_train", "X_holdout", "X_test"])

    seen = df.loc[:t_fit[-1]]
    lo, hi = seen[CHANNELS].min(), seen[CHANNELS].max()
    hsun_lo, hsun_hi = float(seen["H_sun"].min()), float(seen["H_sun"].max())

    keys = {"fit": ("X_train", t_fit), "hold": ("X_holdout", t_hold), "test": ("X_test", t_test)}
    X = {name: (build_windows(df, t, lo, hi), static_vector(splits, key, t, hsun_lo, hsun_hi))
         for name, (key, t) in keys.items()}
    y = {name: np.stack([splits[h][ykey].loc[t].to_numpy() for h in HORIZONS], axis=1) / CAP_KW
         for name, ykey, t in [("fit", "y_train", t_fit), ("hold", "y_holdout", t_hold)]}

    model = build_model(X["fit"][1].shape[1])
    model.summary()
    model.fit(
        list(X["fit"]), y["fit"],
        validation_data=(list(X["hold"]), y["hold"]),
        epochs=60, batch_size=256, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True)],
    )

    def predict_masked(name: str, key: str, times: pd.DatetimeIndex) -> np.ndarray:
        """Predict, clip to [0, 1000], and zero out hours where the sun is down."""
        p = np.clip(model.predict(list(X[name]), verbose=0) * CAP_KW, 0, CAP_KW)
        for j, h in enumerate(HORIZONS):
            night = (splits[h][key]["Hsun_target"].loc[times] <= 0).to_numpy()
            p[night, j] = 0.0
        return p

    hold_pred = predict_masked("hold", "X_holdout", t_hold)
    test_pred = predict_masked("test", "X_test", t_test)

    pers24_rmse = (pd.read_csv(ROOT / "results" / "single_trees.csv")
                   .query("model == 'PERS24'").set_index("horizon")["RMSE"])

    preds, rows = {}, []
    for j, h in enumerate(HORIZONS):
        preds[h] = {
            "holdout": pd.Series(hold_pred[:, j], index=t_hold),
            "test": pd.Series(test_pred[:, j], index=t_test),
        }
        y_te = splits[h]["y_test"].to_numpy()
        p = test_pred[:, j]
        day = (splits[h]["X_test"]["Hsun_target"] > 0).to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y_te, p)))
        rows.append({
            "model": "LSTM-v2", "horizon": h,
            "R2": r2_score(y_te, p), "RMSE": rmse,
            "MAE": mean_absolute_error(y_te, p),
            "R2_daytime": r2_score(y_te[day], p[day]),
            "skill_vs_PERS24": 1 - rmse / pers24_rmse[h],
        })

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "single_lstm_v2.csv", index=False)
    pd.to_pickle(preds, ROOT / "data" / "lstm_preds.pkl")
    print("\n" + res.round(4).to_string(index=False))
    print("\nSaved results/single_lstm_v2.csv and data/lstm_preds.pkl (v2 overwrite)")


if __name__ == "__main__":
    main()
