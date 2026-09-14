"""Tree-ensemble training with rolling-origin evaluation.

For each horizon in HORIZONS and each feature set (lags_only, full):
- Baselines (lags_only only, no fitting): persistence (last observed value
  repeated over the lead times) and seasonal-persistence (value at the same
  hour 24 h earlier, i.e. ws100(t+h-24) = lag 24-h).
- Models: RandomForest, ExtraTrees, XGBoost, LightGBM (all multi-output;
  LightGBM via MultiOutputRegressor).

Protocol (CLAUDE.md):
- Hyperparameters: small documented grid, selected once on fold 0's
  validation block by overall RMSE, reused for all 4 folds.
- Per fold: fit on the training window (validation block excluded so the
  logged val metrics stay meaningful), log val+test metrics per fold plus
  fold="mean"/"std" summary rows; per-step test metrics likewise.
- Per-fold test predictions -> results/predictions_{model}_{features}_
  h{H}_fold{k}.csv; the fold-3 model (largest training window) -> models/.
- Datasets are built once per (horizon, feature set) and cached in memory.
- Figure: cross-fold mean test RMSE vs lead time with +/- std shading,
  one subplot per horizon -> figures/rmse_vs_leadtime_trees.png.

Run from project root: python src/train_trees.py  (full run, ~1-2 h)
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_hourly  # noqa: E402
from make_dataset import HORIZONS, build_supervised  # noqa: E402
from metrics_utils import (METRICS_CSV, METRICS_PER_STEP_CSV,  # noqa: E402
                           leaderboard_rolling, log_fold_summaries,
                           log_metrics, log_metrics_per_step,
                           log_per_step_summaries, scores)
from rolling_eval import rolling_origin_folds  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"
FIGDIR = ROOT / "figures"
FEATURE_SETS = ["lags_only", "full"]
BASELINES = ["persistence", "seasonal_persistence"]
N_TREES = 300

# Small documented grids (4 candidates each), scored on fold 0 validation.
GRIDS = {
    "random_forest": [{"max_features": mf, "min_samples_leaf": msl}
                      for mf, msl in product([0.33, 0.6], [2, 5])],
    "extra_trees": [{"max_features": mf, "min_samples_leaf": msl}
                    for mf, msl in product([0.33, 0.6], [2, 5])],
    "xgboost": [{"max_depth": d, "learning_rate": lr}
                for d, lr in product([4, 8], [0.05, 0.1])],
    "lightgbm": [{"num_leaves": nl, "learning_rate": lr}
                 for nl, lr in product([31, 63], [0.05, 0.1])],
}


def make_model(name: str, params: dict):
    """Instantiate a model with fixed sane defaults + grid params."""
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=N_TREES, random_state=SEED,
                                     n_jobs=-1, **params)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=N_TREES, random_state=SEED,
                                   n_jobs=-1, **params)
    if name == "xgboost":
        return XGBRegressor(n_estimators=N_TREES, subsample=0.9,
                            colsample_bytree=0.9, tree_method="hist",
                            random_state=SEED, n_jobs=-1, verbosity=0,
                            **params)
    if name == "lightgbm":
        base = LGBMRegressor(n_estimators=N_TREES, subsample=0.9,
                             subsample_freq=1, colsample_bytree=0.9,
                             random_state=SEED, verbose=-1, n_jobs=-1,
                             **params)
        return MultiOutputRegressor(base)
    raise ValueError(name)


def fit_predict(name: str, params: dict, Xtr, Ytr, Xte, horizon: int):
    """Fit a fresh model and return (model, test predictions (n, H))."""
    model = make_model(name, params)
    y = Ytr.values
    if name != "lightgbm" and horizon == 1:
        y = y.ravel()  # avoid sklearn column-vector warning
    model.fit(Xtr.values, y)
    pred = np.asarray(model.predict(Xte.values)).reshape(len(Xte), horizon)
    return model, pred


def baseline_predictions(name: str, X: pd.DataFrame, horizon: int) -> np.ndarray:
    """Predictions for the two no-fit baselines from the lag features."""
    if name == "persistence":
        return np.tile(X["ws100_lag0"].values[:, None], (1, horizon))
    if name == "seasonal_persistence":
        cols = [f"ws100_lag{(24 - h) % 24}" for h in range(1, horizon + 1)]
        return X[cols].values
    raise ValueError(name)


def save_predictions(name, features, horizon, k, Yte, pred) -> None:
    out = pd.DataFrame(index=Yte.index)
    for j, col in enumerate(Yte.columns):
        out[f"true_{col}"] = Yte[col].values
        out[f"pred_{col}"] = pred[:, j]
    out.to_csv(RESULTS / f"predictions_{name}_{features}_h{horizon}_fold{k}.csv")


def tune_on_fold0(name: str, fold0: dict, horizon: int) -> dict:
    """Pick grid params with lowest overall RMSE on fold 0's validation."""
    best, best_rmse = None, np.inf
    for params in GRIDS[name]:
        _, pred = fit_predict(name, params, fold0["train"][0],
                              fold0["train"][1], fold0["val"][0], horizon)
        rmse = scores(fold0["val"][1].values, pred)["rmse"]
        if rmse < best_rmse:
            best, best_rmse = params, rmse
    return best


def run_config(name, params, features, horizon, folds) -> None:
    """Fit per fold, log metrics/predictions, save the fold-3 model."""
    is_baseline = name in BASELINES
    test_scores, val_scores, steps, run_times = [], [], [], []
    for f in folds:
        k = f["fold"]
        t0 = time.time()
        if is_baseline:
            pred_val = baseline_predictions(name, f["val"][0], horizon)
            pred = baseline_predictions(name, f["test"][0], horizon)
            model = None
        else:
            model, pred = fit_predict(name, params, f["train"][0],
                                      f["train"][1], f["test"][0], horizon)
            pred_val = np.asarray(model.predict(f["val"][0].values)
                                  ).reshape(len(f["val"][0]), horizon)
        rt = time.time() - t0
        sc_val = scores(f["val"][1].values, pred_val)
        sc_test = scores(f["test"][1].values, pred)
        log_metrics(name, features, horizon, k, "val", sc_val, rt)
        log_metrics(name, features, horizon, k, "test", sc_test, rt)
        steps.append(log_metrics_per_step(name, features, horizon, k, "test",
                                          f["test"][1].values, pred, rt))
        save_predictions(name, features, horizon, k, f["test"][1], pred)
        val_scores.append(sc_val)
        test_scores.append(sc_test)
        run_times.append(rt)
        if model is not None and k == len(folds) - 1:
            joblib.dump(model, MODELS_DIR / f"{name}_{features}_h{horizon}.joblib",
                        compress=3)
    log_fold_summaries(name, features, horizon, "val", val_scores, run_times)
    s = log_fold_summaries(name, features, horizon, "test", test_scores,
                           run_times)
    log_per_step_summaries(name, features, horizon, "test", steps, run_times)
    print(f"  {name:21s} {features:9s} H={horizon:2d}: "
          f"test RMSE {s['rmse_mean']:.3f} +/- {s['rmse_std']:.3f}  "
          f"R {s['r_mean']:.3f}  ({sum(run_times):.0f}s)", flush=True)


def purge_previous(models: list[str]) -> None:
    """Drop old rows/files of these models so re-runs don't duplicate."""
    for path in (METRICS_CSV, METRICS_PER_STEP_CSV):
        if path.exists():
            df = pd.read_csv(path)
            df[~df["model"].isin(models)].to_csv(path, index=False)
    for m in models:
        for p in RESULTS.glob(f"predictions_{m}_*.csv"):
            p.unlink()
        for p in MODELS_DIR.glob(f"{m}_*.joblib"):
            p.unlink()


def make_figure() -> None:
    """Mean test RMSE vs lead time with +/- std bands, one subplot per H."""
    df = pd.read_csv(METRICS_PER_STEP_CSV)
    df = df[df["split"] == "test"]
    colors = {"persistence": "0.45", "seasonal_persistence": "0.7",
              "random_forest": "tab:blue", "extra_trees": "tab:orange",
              "xgboost": "tab:green", "lightgbm": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, H in zip(axes.ravel(), HORIZONS):
        sub = df[df["horizon"] == H]
        for (model, feats), g in sub.groupby(["model", "features"]):
            mean = g[g["fold"] == "mean"].sort_values("step")
            std = g[g["fold"] == "std"].sort_values("step")
            ls = "--" if feats == "lags_only" else "-"
            if model in BASELINES:
                ls = ":" if model == "seasonal_persistence" else "--"
            ax.plot(mean["step"], mean["rmse"], ls, color=colors[model],
                    marker="o" if H == 1 else None, ms=4,
                    label=f"{model} ({feats})")
            if H > 1 and model not in BASELINES:
                ax.fill_between(mean["step"], mean["rmse"] - std["rmse"].values,
                                mean["rmse"] + std["rmse"].values,
                                color=colors[model], alpha=0.12)
        ax.set_title(f"H = {H} h")
        ax.set_xlabel("lead time (h)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("RMSE (m/s)")
    axes[1, 0].set_ylabel("RMSE (m/s)")
    handles, labels = axes.ravel()[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Cross-fold mean test RMSE vs lead time (bands: +/- 1 std)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGDIR / "rmse_vs_leadtime_trees.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Full tree-model experiment over all horizons and feature sets."""
    t_start = time.time()
    RESULTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    purge_previous(BASELINES + list(GRIDS))
    df = load_hourly()
    cache = {}
    selected = {}
    for features in FEATURE_SETS:
        for H in HORIZONS:
            key = (H, features)
            if key not in cache:
                X, Y, _ = build_supervised(df, horizon=H, features=features)
                cache[key] = list(rolling_origin_folds(X, Y, horizon=H))
            folds = cache[key]
            print(f"[{time.time() - t_start:6.0f}s] {features} H={H}",
                  flush=True)
            if features == "lags_only":
                for b in BASELINES:
                    run_config(b, None, features, H, folds)
            for name in GRIDS:
                params = tune_on_fold0(name, folds[0], H)
                selected[f"{name}_{features}_h{H}"] = params
                run_config(name, params, features, H, folds)
    (RESULTS / "selected_params.json").write_text(
        json.dumps(selected, indent=2) + "\n")
    make_figure()
    print(f"\nTotal {time.time() - t_start:.0f}s. Leaderboard (test, "
          "rolling-origin mean across folds):", flush=True)
    print(leaderboard_rolling().to_string())


if __name__ == "__main__":
    main()
