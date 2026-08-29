"""Compute the coefficient of determination R^2 for every model and horizon.

R^2 = 1 - SS_res / SS_tot is a different metric from the Pearson correlation
r that scores() already reports: unlike r, it penalises bias and scale
errors, so it is generally lower. It is computed here exactly from the saved
per-fold test predictions, using the SAME conventions as every other metric
in the project:
- POOLED per fold: all test samples and all lead-time steps are flattened
  together before computing R^2 (matching the pooled-RMSE hard rule);
- SS_tot uses each fold's own test mean;
- reported as mean +/- std across the 4 rolling-origin folds.

Because CLAUDE.md fixes the columns of results/metrics.csv, R^2 is written to
a separate file results/r2_by_model.csv (same row layout: per-fold rows plus
fold="mean"/"std"). This script also refreshes results/report_tables.json so
the Word report's appendix can show R^2 alongside rmse/mae/nrmse/r.

Run from project root: python src/compute_r2.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HORIZONS = [1, 2, 4, 6, 12, 24]
N_FOLDS = 4

# (model, features) pairs and the horizons each was run for. Stack uses a
# different meta-label per horizon, so it is resolved by globbing.
STACK_META = {1: "ridge", 2: "mlp", 4: "mlp", 6: "mlp", 12: "perstep",
              24: "perstep"}
BASE_CONFIGS = [
    ("persistence", "lags_only"), ("seasonal_persistence", "lags_only"),
    ("random_forest", "lags_only"), ("random_forest", "full"),
    ("extra_trees", "lags_only"), ("extra_trees", "full"),
    ("xgboost", "lags_only"), ("xgboost", "full"),
    ("lightgbm", "lags_only"), ("lightgbm", "full"),
    ("lstm", "lags_only"), ("lstm", "full"),
]


def r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination over flattened (samples x steps) arrays."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _fold_r2(model: str, features: str, horizon: int, fold: int):
    """R^2 for one saved prediction file, or None if it is missing."""
    path = (RESULTS /
            f"predictions_{model}_{features}_h{horizon}_fold{fold}.csv")
    if not path.exists():
        return None
    d = pd.read_csv(path)
    tc = [c for c in d.columns if c.startswith("true_")]
    pc = [c for c in d.columns if c.startswith("pred_")]
    return r2_pooled(d[tc].values, d[pc].values)


def compute() -> pd.DataFrame:
    """Return a tidy frame of per-fold + mean/std R^2 for every config."""
    rows = []
    for horizon in HORIZONS:
        configs = list(BASE_CONFIGS) + [("stack", STACK_META[horizon])]
        for model, features in configs:
            vals = []
            for k in range(N_FOLDS):
                v = _fold_r2(model, features, horizon, k)
                if v is None:
                    break
                vals.append(v)
                rows.append({"model": model, "features": features,
                             "horizon": horizon, "fold": str(k),
                             "split": "test", "r2": v})
            if len(vals) == N_FOLDS:
                arr = np.array(vals)
                rows.append({"model": model, "features": features,
                             "horizon": horizon, "fold": "mean",
                             "split": "test", "r2": float(arr.mean())})
                rows.append({"model": model, "features": features,
                             "horizon": horizon, "fold": "std", "split": "test",
                             "r2": float(arr.std(ddof=1))})
    return pd.DataFrame(rows)


def refresh_report_tables(r2: pd.DataFrame) -> None:
    """Rebuild results/report_tables.json with all five metrics incl. R^2."""
    m = pd.read_csv(RESULTS / "metrics.csv")
    t = m[m["split"] == "test"]
    mean = t[t["fold"] == "mean"]
    std = t[t["fold"] == "std"]
    key = ["model", "features", "horizon"]
    d = mean.merge(std[key + ["rmse", "mae", "r", "nrmse"]], on=key,
                   suffixes=("", "_sd"))
    r2m = r2[r2["fold"] == "mean"].set_index(key)["r2"]
    r2s = r2[r2["fold"] == "std"].set_index(key)["r2"]

    label = {
        ("persistence", "lags_only"): "Persistence",
        ("seasonal_persistence", "lags_only"): "Seasonal persistence",
        ("random_forest", "lags_only"): "Random Forest (lags-only)",
        ("random_forest", "full"): "Random Forest (full)",
        ("extra_trees", "lags_only"): "Extra Trees (lags-only)",
        ("extra_trees", "full"): "Extra Trees (full)",
        ("xgboost", "lags_only"): "XGBoost (lags-only)",
        ("xgboost", "full"): "XGBoost (full)",
        ("lightgbm", "lags_only"): "LightGBM (lags-only)",
        ("lightgbm", "full"): "LightGBM (full)",
        ("lstm", "lags_only"): "LSTM (wind-only)",
        ("lstm", "full"): "LSTM (reduced-exog)",
        ("stack", "ridge"): "Stacking ensemble",
        ("stack", "mlp"): "Stacking ensemble",
        ("stack", "perstep"): "Stacking ensemble",
    }
    order = list(dict.fromkeys(label.values()))
    out = {}
    for horizon in HORIZONS:
        sub = d[d["horizon"] == horizon]
        rows = {}
        for _, r in sub.iterrows():
            lab = label[(r.model, r.features)]
            idx = (r.model, r.features, horizon)
            rows[lab] = {
                "rmse": [round(r.rmse, 3), round(r.rmse_sd, 3)],
                "mae": [round(r.mae, 3), round(r.mae_sd, 3)],
                "nrmse": [round(r.nrmse, 3), round(r.nrmse_sd, 3)],
                "r": [round(r.r, 3), round(r.r_sd, 3)],
                "r2": [round(float(r2m.loc[idx]), 3),
                       round(float(r2s.loc[idx]), 3)],
            }
        out[str(horizon)] = {"order": [l for l in order if l in rows],
                             "rows": rows}
    (RESULTS / "report_tables.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    r2 = compute()
    r2.to_csv(RESULTS / "r2_by_model.csv", index=False)
    refresh_report_tables(r2)

    mean = r2[r2["fold"] == "mean"]
    std = r2[r2["fold"] == "std"]
    key = ["model", "features", "horizon"]
    d = mean.merge(std[key + ["r2"]], on=key, suffixes=("", "_sd"))
    d["label"] = d.model + " (" + d.features + ")"
    piv = d.pivot_table(index="label", columns="horizon", values="r2")
    order = ["persistence (lags_only)", "seasonal_persistence (lags_only)",
             "random_forest (lags_only)", "random_forest (full)",
             "extra_trees (lags_only)", "extra_trees (full)",
             "xgboost (lags_only)", "xgboost (full)",
             "lightgbm (lags_only)", "lightgbm (full)",
             "lstm (lags_only)", "lstm (full)", "stack (ridge)",
             "stack (mlp)", "stack (perstep)"]
    piv = piv.reindex([o for o in order if o in piv.index])
    print("R^2 (coefficient of determination), mean over 4 folds:")
    print(piv.round(3).to_string())
    print("\nWrote results/r2_by_model.csv and refreshed "
          "results/report_tables.json")


if __name__ == "__main__":
    main()
