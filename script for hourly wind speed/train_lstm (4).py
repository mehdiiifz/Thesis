"""LSTM training with rolling-origin evaluation.

Input = sequence of the last 24 hourly timesteps; Dense(H) linear head.
Per fold, a StandardScaler is fit on the training window only. Training
uses Adam/MSE, batch 64, max 100 epochs, EarlyStopping(patience=10,
restore_best_weights=True) on the fold's validation block. Seed 42.

Variants (logged as model="lstm" with the standard features labels):
- "lags_only": timestep feature = [ws100] (wind-only, Mito-comparable).
- "full": naive all-feature input previously HURT the LSTM (overfitting),
  so three fixes are compared on fold 0 validation (at H=6 and H=24) and
  the best mean val RMSE wins (results/lstm_full_selection.md):
    (a) reduced-exogenous channels [ws100, u, v, hour_sin, hour_cos];
    (b) all 9 channels + dropout 0.3, recurrent_dropout 0.2, L2 1e-4;
    (c) all 9 channels + stacked LSTM(128) -> LSTM(64).
  Candidates (a)/(b) use 64 units during selection; if the winner is
  (a)/(b) its units are then tuned per horizon like the wind-only variant.

Units grid 32/64/128 is selected on fold 0's validation only and reused
for all folds (CLAUDE.md). Logging, prediction files and fold-3 model
saving mirror train_trees.py; training curves ->
figures/lstm_training_curves_{features}.png.

Run from project root: python src/train_lstm.py  (full run, hours on CPU)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_hourly  # noqa: E402
from make_dataset import HORIZONS, wind_uv  # noqa: E402
from metrics_utils import (METRICS_CSV, METRICS_PER_STEP_CSV,  # noqa: E402
                           leaderboard_rolling, log_fold_summaries,
                           log_metrics, log_metrics_per_step,
                           log_per_step_summaries, scores)
from rolling_eval import rolling_origin_folds  # noqa: E402

SEED = 42
N_STEPS = 24
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 10
UNITS_GRID = [32, 64, 128]
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"
FIGDIR = ROOT / "figures"

CH_WIND = ["ws100"]
CH_ALL = ["ws100", "temp_c", "pres_hpa", "u", "v",
          "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
CH_REDUCED = ["ws100", "u", "v", "hour_sin", "hour_cos"]

# The three candidate fixes for the "full" variant (selection on fold 0).
CANDIDATES = {
    "a_reduced_exog": dict(channels=CH_REDUCED, arch="single", units=64,
                           dropout=0.0, rec_dropout=0.0, l2=0.0,
                           tune_units=True),
    "b_all_regularized": dict(channels=CH_ALL, arch="single", units=64,
                              dropout=0.3, rec_dropout=0.2, l2=1e-4,
                              tune_units=True),
    "c_all_stacked": dict(channels=CH_ALL, arch="stacked", units=None,
                          dropout=0.0, rec_dropout=0.0, l2=0.0,
                          tune_units=False),
}
WIND_SPEC = dict(channels=CH_WIND, arch="single", units=64, dropout=0.0,
                 rec_dropout=0.0, l2=0.0, tune_units=True)


def channel_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp channel values (all computable at that timestamp)."""
    out = pd.DataFrame(index=df.index)
    out["ws100"] = df["ws100"].astype(float)
    out["temp_c"] = df["temp_c"].astype(float)
    out["pres_hpa"] = df["pres_hpa"].astype(float)
    out["u"], out["v"] = wind_uv(out["ws100"], df["dir"].astype(float))
    hour = df.index.hour.to_numpy()
    doy = df.index.dayofyear.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    return out


def build_sequences(df: pd.DataFrame, horizon: int, channels: list[str]):
    """(Xflat, Y) with issue-time index; Xflat row = 24 steps x channels.

    Column order is timestep-major, oldest step first, so
    Xflat.values.reshape(n, N_STEPS, C) yields sequences ending at t.
    Same no-leakage convention as build_supervised: features shift(k>=0),
    targets shift(-h).
    """
    ch = channel_frame(df)[channels]
    cols = {}
    for k in range(N_STEPS - 1, -1, -1):
        for c in channels:
            cols[f"{c}_tm{k}"] = ch[c].shift(k)
    Xflat = pd.DataFrame(cols, index=df.index)
    Y = pd.DataFrame({f"t+{h}h": df["ws100"].shift(-h).astype(float)
                      for h in range(1, horizon + 1)}, index=df.index)
    valid = Xflat.notna().all(axis=1) & Y.notna().all(axis=1)
    Xflat, Y = Xflat.loc[valid], Y.loc[valid]
    Xflat.index.name = Y.index.name = "timestamp"
    return Xflat, Y


def to_3d(Xflat: pd.DataFrame, n_channels: int) -> np.ndarray:
    return Xflat.values.reshape(len(Xflat), N_STEPS, n_channels)


def build_model(horizon: int, n_channels: int, spec: dict) -> keras.Model:
    """Single or stacked LSTM with a linear Dense(H) head."""
    reg = keras.regularizers.l2(spec["l2"]) if spec["l2"] else None
    inp = keras.Input(shape=(N_STEPS, n_channels))
    if spec["arch"] == "stacked":
        x = keras.layers.LSTM(128, return_sequences=True)(inp)
        x = keras.layers.LSTM(64)(x)
    else:
        x = keras.layers.LSTM(spec["units"], dropout=spec["dropout"],
                              recurrent_dropout=spec["rec_dropout"],
                              kernel_regularizer=reg)(inp)
    out = keras.layers.Dense(horizon)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def fit_on_fold(spec: dict, horizon: int, fold: dict, n_channels: int):
    """Scale (train-only fit), train with early stopping; return
    (model, history, scaled val X, scaled test X)."""
    Xtr, Ytr = fold["train"]
    scaler = StandardScaler().fit(
        to_3d(Xtr, n_channels).reshape(-1, n_channels))

    def prep(Xf):
        a = to_3d(Xf, n_channels)
        return scaler.transform(a.reshape(-1, n_channels)).reshape(a.shape)

    keras.utils.set_random_seed(SEED)
    model = build_model(horizon, n_channels, spec)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True)
    Xv = prep(fold["val"][0])
    hist = model.fit(prep(Xtr), Ytr.values,
                     validation_data=(Xv, fold["val"][1].values),
                     epochs=MAX_EPOCHS, batch_size=BATCH, verbose=0,
                     callbacks=[es])
    return model, hist, Xv, prep(fold["test"][0])


def val_rmse_fold0(spec: dict, horizon: int, fold0: dict,
                   n_channels: int) -> float:
    model, _, Xv, _ = fit_on_fold(spec, horizon, fold0, n_channels)
    pred = model.predict(Xv, verbose=0, batch_size=512)
    return scores(fold0["val"][1].values, pred)["rmse"]


def get_folds(cache: dict, df: pd.DataFrame, horizon: int,
              channels: list[str]) -> list[dict]:
    key = (tuple(channels), horizon)
    if key not in cache:
        Xf, Y = build_sequences(df, horizon, channels)
        cache[key] = list(rolling_origin_folds(Xf, Y, horizon=horizon))
    return cache[key]


def select_full_variant(df: pd.DataFrame, cache: dict) -> str:
    """Pick the best full-variant candidate on fold 0 val at H=6 and H=24."""
    results = {name: {} for name in CANDIDATES}
    for name, spec in CANDIDATES.items():
        for H in (6, 24):
            fold0 = get_folds(cache, df, H, spec["channels"])[0]
            t0 = time.time()
            rmse = val_rmse_fold0(spec, H, fold0, len(spec["channels"]))
            results[name][H] = rmse
            print(f"  candidate {name:18s} H={H:2d}: val RMSE {rmse:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    means = {n: np.mean(list(r.values())) for n, r in results.items()}
    winner = min(means, key=means.get)
    lines = ["# LSTM full-variant selection (fold 0 validation)", "",
             "Naive all-feature input previously hurt the LSTM (overfit).",
             "Candidates compared on fold 0 val RMSE at H=6 and H=24:", "",
             "| candidate | H=6 | H=24 | mean |", "|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r[6]:.3f} | {r[24]:.3f} | {means[n]:.3f} |")
    lines += ["", f"Winner: **{winner}** -> used as the LSTM \"full\" "
              "variant for all horizons/folds."]
    (RESULTS / "lstm_full_selection.md").write_text("\n".join(lines) + "\n")
    return winner


def run_variant(features: str, spec: dict, df: pd.DataFrame, cache: dict,
                histories: dict) -> None:
    """Tune units on fold 0 (if applicable), then run all horizons/folds."""
    C = len(spec["channels"])
    for H in HORIZONS:
        folds = get_folds(cache, df, H, spec["channels"])
        spec_h = dict(spec)
        if spec["tune_units"]:
            best_u, best = None, np.inf
            for u in UNITS_GRID:
                r = val_rmse_fold0({**spec_h, "units": u}, H, folds[0], C)
                if r < best:
                    best_u, best = u, r
            spec_h["units"] = best_u
            print(f"  lstm {features} H={H}: units={best_u} "
                  f"(fold-0 val RMSE {best:.3f})", flush=True)
        test_scores, val_scores, steps, run_times = [], [], [], []
        for f in folds:
            k = f["fold"]
            t0 = time.time()
            model, hist, Xv, Xte = fit_on_fold(spec_h, H, f, C)
            pred_val = model.predict(Xv, verbose=0, batch_size=512)
            pred = model.predict(Xte, verbose=0, batch_size=512)
            rt = time.time() - t0
            sc_val = scores(f["val"][1].values, pred_val)
            sc_test = scores(f["test"][1].values, pred)
            log_metrics("lstm", features, H, k, "val", sc_val, rt)
            log_metrics("lstm", features, H, k, "test", sc_test, rt)
            steps.append(log_metrics_per_step("lstm", features, H, k, "test",
                                              f["test"][1].values, pred, rt))
            out = pd.DataFrame(index=f["test"][1].index)
            for j, col in enumerate(f["test"][1].columns):
                out[f"true_{col}"] = f["test"][1][col].values
                out[f"pred_{col}"] = pred[:, j]
            out.to_csv(RESULTS / f"predictions_lstm_{features}_h{H}_fold{k}.csv")
            histories[(features, H, k)] = hist.history
            val_scores.append(sc_val)
            test_scores.append(sc_test)
            run_times.append(rt)
            if k == len(folds) - 1:
                model.save(MODELS_DIR / f"lstm_{features}_h{H}.keras")
        log_fold_summaries("lstm", features, H, "val", val_scores, run_times)
        s = log_fold_summaries("lstm", features, H, "test", test_scores,
                               run_times)
        log_per_step_summaries("lstm", features, H, "test", steps, run_times)
        print(f"  lstm {features:9s} H={H:2d}: test RMSE "
              f"{s['rmse_mean']:.3f} +/- {s['rmse_std']:.3f}  "
              f"R {s['r_mean']:.3f}  ({sum(run_times):.0f}s)", flush=True)


def plot_training_curves(histories: dict) -> None:
    """Loss curves per horizon (one figure per variant, all folds)."""
    for features in ["lags_only", "full"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for ax, H in zip(axes.ravel(), HORIZONS):
            for k in range(4):
                h = histories.get((features, H, k))
                if h is None:
                    continue
                ax.plot(h["loss"], color=f"C{k}", alpha=0.35, lw=1)
                ax.plot(h["val_loss"], color=f"C{k}", lw=1.5,
                        label=f"fold {k} val")
            ax.set_title(f"H = {H} h")
            ax.set_xlabel("epoch")
            ax.set_ylabel("MSE loss")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"LSTM training curves ({features}); faint = train, "
                     "solid = validation")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"lstm_training_curves_{features}.png", dpi=150)
        plt.close(fig)


def purge_previous() -> None:
    """Drop earlier lstm rows/files so re-runs don't duplicate."""
    for path in (METRICS_CSV, METRICS_PER_STEP_CSV):
        if path.exists():
            d = pd.read_csv(path)
            d[d["model"] != "lstm"].to_csv(path, index=False)
    for p in RESULTS.glob("predictions_lstm_*.csv"):
        p.unlink()
    for p in MODELS_DIR.glob("lstm_*.keras"):
        p.unlink()


def main() -> None:
    """Full LSTM experiment: variant selection, tuning, all horizons/folds."""
    t0 = time.time()
    print(f"tensorflow {tf.__version__}, seed {SEED}", flush=True)
    purge_previous()
    df = load_hourly()
    cache, histories = {}, {}
    print("Selecting full-variant fix on fold 0:", flush=True)
    winner = select_full_variant(df, cache)
    print(f"full-variant winner: {winner}", flush=True)
    run_variant("lags_only", WIND_SPEC, df, cache, histories)
    run_variant("full", CANDIDATES[winner], df, cache, histories)
    plot_training_curves(histories)
    print(f"\nTotal {time.time() - t0:.0f}s. Leaderboard:", flush=True)
    print(leaderboard_rolling().to_string())


if __name__ == "__main__":
    main()
