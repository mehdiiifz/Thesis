"""Stage 9: 4-fold rolling-origin seasonal cross-validation.

Folds (chronological, never shuffled; features rebuilt per fold with the
exact stage-3 logic, imported from s3_features):
  fold 0: train 2012-01..2012-12 -> test 2013 Q1
  fold 1: train 2012-01..2013-03 -> test 2013 Q2
  fold 2: train 2012-01..2013-06 -> test 2013 Q3
  fold 3: train 2012-01..2013-09 -> test 2013 Q4
Within each fold's training rows the last 15% is the holdout, used for LSTM
early stopping and for fitting the stacking meta-learner.

Models per fold and horizon: PERS, PERS24, DT, RF, ET, XGB (stage-4 spec),
LSTM-v2 (stage-5b spec, retrained once per fold), RF+XGB and RF+XGB+LSTM
(stage-6-v2 constrained Ridge). Every model's output is night-masked
(prediction = 0 wherever H_sun(t+h) <= 0) and clipped to [0, 1000] kW.

Resumable: each completed fold is appended to results/seasonal_cv.csv and
results/seasonal_cv_weights.csv; already-present folds are skipped on rerun.
After all folds, per-(model, horizon) aggregate rows with fold='mean' and
fold='std' are (re)written.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
random.seed(42)
np.random.seed(42)

import tensorflow as tf  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from s3_features import HORIZONS, issue_time_features, target_time_features  # noqa: E402
from s4_trees import make_models  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CV_CSV = ROOT / "results" / "seasonal_cv.csv"
W_CSV = ROOT / "results" / "seasonal_cv_weights.csv"
CAP_KW = 1000.0
WINDOW = 48
CHANNELS = ["P_kW", "G(i)", "T2m", "WS10m"]

FOLDS = [  # (fold, train_end, test_start, test_end, split label)
    (0, "2012-12-31 23:00", "2013-01-01", "2013-03-31 23:00", "2013Q1"),
    (1, "2013-03-31 23:00", "2013-04-01", "2013-06-30 23:00", "2013Q2"),
    (2, "2013-06-30 23:00", "2013-07-01", "2013-09-30 23:00", "2013Q3"),
    (3, "2013-09-30 23:00", "2013-10-01", "2013-12-31 23:00", "2013Q4"),
]


def mask_clip(pred: np.ndarray, hsun_target: np.ndarray) -> np.ndarray:
    out = np.clip(pred, 0, CAP_KW)
    out[hsun_target <= 0] = 0.0
    return out


def metrics(y, p, day):
    return {
        "R2": r2_score(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": mean_absolute_error(y, p),
        "R2_daytime": r2_score(y[day], p[day]),
    }


# ---- LSTM (stage-5b spec) -------------------------------------------------

def build_windows(df, times, lo, hi):
    scaled = ((df[CHANNELS] - lo) / (hi - lo)).to_numpy(dtype=np.float32)
    sw = np.lib.stride_tricks.sliding_window_view(scaled, WINDOW, axis=0)
    return sw[df.index.get_indexer(times) - (WINDOW - 1)].transpose(0, 2, 1)


def static_vector(df, times, hsun_lo, hsun_hi):
    cols = []
    for h in HORIZONS:
        th = (times + pd.Timedelta(hours=h)).hour
        cols += [np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24)]
    doy = times.dayofyear
    cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    for h in HORIZONS:
        hs = df["H_sun"].shift(-h).loc[times].to_numpy()
        cols.append((hs - hsun_lo) / (hsun_hi - hsun_lo))
    return np.stack(cols, axis=1).astype(np.float32)


def train_lstm(df, t_fit, t_hold, y_fit, y_hold):
    tf.random.set_seed(42)
    seen = df.loc[:t_fit[-1]]
    lo, hi = seen[CHANNELS].min(), seen[CHANNELS].max()
    hsun_lo, hsun_hi = float(seen["H_sun"].min()), float(seen["H_sun"].max())

    def inputs(times):
        return [build_windows(df, times, lo, hi), static_vector(df, times, hsun_lo, hsun_hi)]

    seq_in = tf.keras.Input(shape=(WINDOW, len(CHANNELS)))
    cal_in = tf.keras.Input(shape=(2 * len(HORIZONS) + 2 + len(HORIZONS),))
    x = tf.keras.layers.LSTM(128)(seq_in)
    x = tf.keras.layers.Concatenate()([x, cal_in])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(len(HORIZONS), activation="sigmoid")(x)
    model = tf.keras.Model([seq_in, cal_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(inputs(t_fit), y_fit / CAP_KW,
              validation_data=(inputs(t_hold), y_hold / CAP_KW),
              epochs=60, batch_size=256, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=6, restore_best_weights=True)])
    return model, inputs


# ---- one fold -------------------------------------------------------------

def run_fold(fold, train_end, test_start, test_end, split):
    df = pd.read_pickle(ROOT / "data" / "clean.pkl")
    base = issue_time_features(df)
    valid = df.index[WINDOW:-max(HORIZONS)]
    train_idx = valid[valid <= pd.Timestamp(train_end, tz="UTC")]
    test_idx = valid[(valid >= pd.Timestamp(test_start, tz="UTC"))
                     & (valid <= pd.Timestamp(test_end, tz="UTC"))]
    n_hold = int(round(len(train_idx) * 0.15))
    fit_idx, hold_idx = train_idx[:-n_hold], train_idx[-n_hold:]
    print(f"fold {fold}: fit {len(fit_idx)}, holdout {len(hold_idx)}, test {len(test_idx)} ({split})")

    data = {}
    for h in HORIZONS:
        X = pd.concat([base, target_time_features(df, h)], axis=1)
        y = df["P_kW"].shift(-h)
        data[h] = {
            "X_fit": X.loc[fit_idx], "y_fit": y.loc[fit_idx],
            "X_hold": X.loc[hold_idx], "y_hold": y.loc[hold_idx],
            "X_test": X.loc[test_idx], "y_test": y.loc[test_idx],
            "hs_hold": X.loc[hold_idx, "Hsun_target"].to_numpy(),
            "hs_test": X.loc[test_idx, "Hsun_target"].to_numpy(),
        }

    rows, wrows = [], []
    hold_preds, test_preds = {}, {}  # model -> h -> np.ndarray

    def record(model, h, test_pred, run_time):
        d = data[h]
        day = d["hs_test"] > 0
        rows.append({"model": model, "horizon": h, "fold": fold, "split": split,
                     **metrics(d["y_test"].to_numpy(), test_pred, day),
                     "run_time": round(run_time, 2)})

    # baselines (masked like everything else)
    for name, col in [("PERS", "P_lag0"), ("PERS24", "P_samehour_yday_target")]:
        t0 = time.time()
        hold_preds[name], test_preds[name] = {}, {}
        for h in HORIZONS:
            d = data[h]
            hold_preds[name][h] = mask_clip(d["X_hold"][col].to_numpy(), d["hs_hold"])
            test_preds[name][h] = mask_clip(d["X_test"][col].to_numpy(), d["hs_test"])
            record(name, h, test_preds[name][h], (time.time() - t0) / len(HORIZONS))

    # trees (stage-4 spec: fit on fit -> holdout preds; refit on full train -> test)
    for name, model in make_models().items():
        hold_preds[name], test_preds[name] = {}, {}
        for h in HORIZONS:
            d = data[h]
            t0 = time.time()
            model.fit(d["X_fit"], d["y_fit"])
            hold_preds[name][h] = mask_clip(model.predict(d["X_hold"]), d["hs_hold"])
            model.fit(pd.concat([d["X_fit"], d["X_hold"]]), pd.concat([d["y_fit"], d["y_hold"]]))
            test_preds[name][h] = mask_clip(model.predict(d["X_test"]), d["hs_test"])
            record(name, h, test_preds[name][h], time.time() - t0)
            print(f"  fold {fold} h={h:>2} {name:<4} RMSE={rows[-1]['RMSE']:.1f}")

    # LSTM-v2, retrained once per fold
    t0 = time.time()
    y_fit = np.stack([data[h]["y_fit"].to_numpy() for h in HORIZONS], axis=1)
    y_hold = np.stack([data[h]["y_hold"].to_numpy() for h in HORIZONS], axis=1)
    lstm, inputs = train_lstm(pd.read_pickle(ROOT / "data" / "clean.pkl"),
                              fit_idx, hold_idx, y_fit, y_hold)
    ph = lstm.predict(inputs(hold_idx), verbose=0) * CAP_KW
    pt = lstm.predict(inputs(test_idx), verbose=0) * CAP_KW
    lstm_time = time.time() - t0
    hold_preds["LSTM-v2"], test_preds["LSTM-v2"] = {}, {}
    for j, h in enumerate(HORIZONS):
        d = data[h]
        hold_preds["LSTM-v2"][h] = mask_clip(ph[:, j], d["hs_hold"])
        test_preds["LSTM-v2"][h] = mask_clip(pt[:, j], d["hs_test"])
        record("LSTM-v2", h, test_preds["LSTM-v2"][h], lstm_time / len(HORIZONS))
    print(f"  fold {fold} LSTM-v2 trained in {lstm_time:.0f}s")

    # stacks (stage-6-v2 constrained Ridge on this fold's holdout)
    for name, members in [("RF+XGB", ["RF", "XGB"]), ("RF+XGB+LSTM", ["RF", "XGB", "LSTM-v2"])]:
        test_preds[name] = {}
        for h in HORIZONS:
            d = data[h]
            t0 = time.time()
            H = np.column_stack([hold_preds[m][h] for m in members])
            T = np.column_stack([test_preds[m][h] for m in members])
            meta = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(H, d["y_hold"])
            blend = mask_clip(meta.predict(T), d["hs_test"])
            test_preds[name][h] = blend
            record(name, h, blend, time.time() - t0)
            wrows.append({"model": name, "horizon": h, "fold": fold, "split": split,
                          **{f"w_{m.replace('-v2', '')}": w for m, w in zip(members, meta.coef_)}})

    # skill vs this fold's PERS24
    p24 = {r["horizon"]: r["RMSE"] for r in rows if r["model"] == "PERS24"}
    for r in rows:
        r["skill_vs_PERS24"] = 1 - r["RMSE"] / p24[r["horizon"]]
    return pd.DataFrame(rows), pd.DataFrame(wrows)


def append(df, path):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    done = set()
    if CV_CSV.exists():
        prev = pd.read_csv(CV_CSV)
        done = set(prev[~prev["fold"].isin(["mean", "std"])]["fold"].astype(int))
        print(f"resuming; folds already done: {sorted(done)}")

    for fold, train_end, test_start, test_end, split in FOLDS:
        if fold in done:
            continue
        res, w = run_fold(fold, train_end, test_start, test_end, split)
        append(res, CV_CSV)
        append(w, W_CSV)
        print(f"fold {fold} appended.\n")

    # (re)write aggregate rows
    full = pd.read_csv(CV_CSV)
    full = full[~full["fold"].isin(["mean", "std"])]
    num = ["R2", "RMSE", "MAE", "R2_daytime", "skill_vs_PERS24", "run_time"]
    aggs = []
    for stat in ["mean", "std"]:
        agg = full.groupby(["model", "horizon"])[num].agg(stat).reset_index()
        agg["fold"], agg["split"] = stat, "all"
        aggs.append(agg)
    out = pd.concat([full] + aggs, ignore_index=True)[
        ["model", "horizon", "fold", "split"] + num]
    out.to_csv(CV_CSV, index=False)
    print(f"Saved {CV_CSV.relative_to(ROOT)} ({len(out)} rows) and {W_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
