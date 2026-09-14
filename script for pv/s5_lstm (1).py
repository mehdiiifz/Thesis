"""Stage 5: one multi-output LSTM forecasting all 8 horizons jointly.

Inputs per sample (issue time t, same timestamps as the stage-3/4 splits):
  1. the past 48 hours of [P_kW, G(i), T2m, WS10m], MinMax-scaled with
     statistics computed on the fit portion only (never holdout or test);
  2. a calendar vector: sin/cos of each of the 8 target hours + sin/cos of
     the issue day-of-year (18 values, all deterministic).

Architecture: LSTM(64) -> concat calendar -> Dense(64, relu) -> Dense(8,
sigmoid), targets scaled by /1000 to match the sigmoid range. Adam(1e-3),
MSE, batch 256, up to 60 epochs with EarlyStopping(patience=6,
restore_best_weights=True) monitored on the stage-3 holdout block — the test
set is never touched during training. The single trained model produces both
the holdout predictions (meta-learner training data for stacking) and the
test predictions.

Outputs: data/lstm_preds.pkl (committed) and results/single_lstm.csv.
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
    """(n, 48, 4) array of the 48 hours ending at each issue time, MinMax-scaled."""
    scaled = ((df[CHANNELS] - lo) / (hi - lo)).to_numpy(dtype=np.float32)
    sw = np.lib.stride_tricks.sliding_window_view(scaled, WINDOW, axis=0)  # (n-47, 4, 48)
    pos = df.index.get_indexer(times)
    return sw[pos - (WINDOW - 1)].transpose(0, 2, 1)


def calendar_vector(times: pd.DatetimeIndex) -> np.ndarray:
    """sin/cos of each target hour plus sin/cos of the issue day-of-year."""
    cols = []
    for h in HORIZONS:
        target_hour = (times + pd.Timedelta(hours=h)).hour
        cols += [np.sin(2 * np.pi * target_hour / 24), np.cos(2 * np.pi * target_hour / 24)]
    doy = times.dayofyear
    cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    return np.stack(cols, axis=1).astype(np.float32)


def build_model() -> tf.keras.Model:
    seq_in = tf.keras.Input(shape=(WINDOW, len(CHANNELS)), name="past48h")
    cal_in = tf.keras.Input(shape=(2 * len(HORIZONS) + 2,), name="calendar")
    x = tf.keras.layers.LSTM(64)(seq_in)
    x = tf.keras.layers.Concatenate()([x, cal_in])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(len(HORIZONS), activation="sigmoid")(x)
    model = tf.keras.Model([seq_in, cal_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    df = pd.read_pickle(ROOT / "data" / "clean.pkl")
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")

    s1 = splits[HORIZONS[0]]
    t_fit, t_hold, t_test = (s1[k].index for k in ["X_train", "X_holdout", "X_test"])

    # MinMax stats from data visible to the fit portion only (up to its last hour)
    seen = df.loc[:t_fit[-1], CHANNELS]
    lo, hi = seen.min(), seen.max()

    X = {name: (build_windows(df, t, lo, hi), calendar_vector(t))
         for name, t in [("fit", t_fit), ("hold", t_hold), ("test", t_test)]}
    y = {name: np.stack([splits[h][key].loc[t].to_numpy() for h in HORIZONS], axis=1) / CAP_KW
         for name, key, t in [("fit", "y_train", t_fit), ("hold", "y_holdout", t_hold)]}

    model = build_model()
    model.summary()
    model.fit(
        list(X["fit"]), y["fit"],
        validation_data=(list(X["hold"]), y["hold"]),
        epochs=60, batch_size=256, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True)],
    )

    preds, rows = {}, []
    hold_pred = np.clip(model.predict(list(X["hold"]), verbose=0) * CAP_KW, 0, CAP_KW)
    test_pred = np.clip(model.predict(list(X["test"]), verbose=0) * CAP_KW, 0, CAP_KW)

    pers24_rmse = (pd.read_csv(ROOT / "results" / "single_trees.csv")
                   .query("model == 'PERS24'").set_index("horizon")["RMSE"])

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
            "model": "LSTM", "horizon": h,
            "R2": r2_score(y_te, p), "RMSE": rmse,
            "MAE": mean_absolute_error(y_te, p),
            "R2_daytime": r2_score(y_te[day], p[day]),
            "skill_vs_PERS24": 1 - rmse / pers24_rmse[h],
        })

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "single_lstm.csv", index=False)
    pd.to_pickle(preds, ROOT / "data" / "lstm_preds.pkl")
    print("\n" + res.round(4).to_string(index=False))
    print("\nSaved results/single_lstm.csv and data/lstm_preds.pkl")


if __name__ == "__main__":
    main()
