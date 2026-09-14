"""Stage 4: persistence baselines and single tree models across all horizons.

Two training-free baselines are evaluated on the same test set as the models:
  PERS   — prediction for t+h is P(t)          (the P_lag0 column)
  PERS24 — prediction for t+h is P(t+h-24)     (the P_samehour_yday_target column)

Then four tree models (DT, RF, ET, XGB) per horizon:
  1. fit on the fit portion  -> predict the holdout (meta-learner training data
     for later stacking stages)
  2. refit on fit+holdout    -> predict the test set
All predictions are clipped to [0, 1000] kW. Metrics: R2, RMSE, MAE, daytime
R2 (rows with H_sun at the target time > 0), and the skill score vs PERS24
(1 - RMSE_model / RMSE_PERS24).

Outputs: results/single_trees.csv and data/tree_preds.pkl (committed, so later
stages can reference the predictions without refitting).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]
CAP_KW = 1000.0
SEED = 42

BASELINE_COLS = {"PERS": "P_lag0", "PERS24": "P_samehour_yday_target"}


def make_models() -> dict:
    return {
        "DT": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=SEED),
        "RF": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        "ET": ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        "XGB": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                            subsample=0.9, colsample_bytree=0.9,
                            n_jobs=-1, random_state=SEED),
    }


def metrics(y_true: pd.Series, y_pred: np.ndarray, daytime: pd.Series) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2_daytime": r2_score(y_true[daytime], y_pred[daytime.to_numpy()]),
    }


def main() -> None:
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    rows, preds = [], {name: {} for name in [*BASELINE_COLS, *make_models()]}

    for h in HORIZONS:
        s = splits[h]
        X_sub, y_sub = s["X_train"], s["y_train"]          # fit portion
        X_hold, y_hold = s["X_holdout"], s["y_holdout"]
        X_tr = pd.concat([X_sub, X_hold])                  # full train
        y_tr = pd.concat([y_sub, y_hold])
        X_te, y_te = s["X_test"], s["y_test"]
        daytime = X_te["Hsun_target"] > 0

        for name, col in BASELINE_COLS.items():            # training-free baselines
            hold_pred = X_hold[col].clip(0, CAP_KW)
            test_pred = X_te[col].clip(0, CAP_KW)
            preds[name][h] = {"holdout": hold_pred, "test": test_pred}
            rows.append({"model": name, "horizon": h,
                         **metrics(y_te, test_pred.to_numpy(), daytime)})

        for name, model in make_models().items():
            t0 = time.time()
            model.fit(X_sub, y_sub)
            hold_pred = pd.Series(model.predict(X_hold), index=X_hold.index).clip(0, CAP_KW)
            model.fit(X_tr, y_tr)
            test_pred = pd.Series(model.predict(X_te), index=X_te.index).clip(0, CAP_KW)
            preds[name][h] = {"holdout": hold_pred, "test": test_pred}
            rows.append({"model": name, "horizon": h,
                         **metrics(y_te, test_pred.to_numpy(), daytime)})
            print(f"h={h:>2} {name:<4} fitted in {time.time() - t0:5.1f}s  "
                  f"test R2={rows[-1]['R2']:.4f} RMSE={rows[-1]['RMSE']:.1f}")

    res = pd.DataFrame(rows)
    pers24_rmse = res[res["model"] == "PERS24"].set_index("horizon")["RMSE"]
    res["skill_vs_PERS24"] = 1 - res["RMSE"] / res["horizon"].map(pers24_rmse)

    res.to_csv(ROOT / "results" / "single_trees.csv", index=False)
    pd.to_pickle(preds, ROOT / "data" / "tree_preds.pkl")
    print("\n" + res.round(4).to_string(index=False))
    print("\nSaved results/single_trees.csv and data/tree_preds.pkl")


if __name__ == "__main__":
    main()
