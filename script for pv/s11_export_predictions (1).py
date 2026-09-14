"""Stage 11: export actual vs predicted values of the recommended forecaster.

The recommended model per the seasonal cross-validation is the RF+XGB+LSTM
hybrid for horizons 1-12 h and Random Forest for 24 h. This script rebuilds
those predictions on the untouched single-split test set (2013-07-01 ..
2013-12-30) with the stage-7 procedure — constrained Ridge meta-learner fitted
on the holdout block, clipping to [0, 1000] kW, night masking — and writes
them next to the measured values.

Outputs (both in results/):
  best_model_predictions.csv       tidy: one row per horizon and issue time
  best_model_predictions_wide.csv  one row per issue time, actual/predicted
                                   column pair per horizon
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]
CAP_KW = 1000.0
HYBRID_MEMBERS = ["RF", "XGB", "LSTM"]


def main() -> None:
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    tree = pd.read_pickle(ROOT / "data" / "tree_preds.pkl")
    lstm = pd.read_pickle(ROOT / "data" / "lstm_preds.pkl")
    base = lambda m, h, blk: (lstm[h] if m == "LSTM" else tree[m][h])[blk]

    frames = []
    for h in HORIZONS:
        s = splits[h]
        hsun = s["X_test"]["Hsun_target"].to_numpy()
        model = "RF+XGB+LSTM hybrid" if h <= 12 else "Random Forest"

        if h <= 12:  # rebuild the hybrid exactly as in stage 7
            H = np.column_stack([base(m, h, "holdout") for m in HYBRID_MEMBERS])
            T = np.column_stack([base(m, h, "test") for m in HYBRID_MEMBERS])
            meta = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(H, s["y_holdout"])
            pred = meta.predict(T)
        else:
            pred = tree["RF"][h]["test"].to_numpy()
        pred = np.clip(pred, 0, CAP_KW)
        pred[hsun <= 0] = 0.0

        actual = s["y_test"].to_numpy()
        issue = s["y_test"].index
        frames.append(pd.DataFrame({
            "issue_time_utc": issue.strftime("%Y-%m-%d %H:%M"),
            "target_time_utc": (issue + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M"),
            "horizon_h": h,
            "model": model,
            "actual_kW": actual.round(3),
            "predicted_kW": pred.round(3),
            "error_kW": (pred - actual).round(3),
            "sun_up": hsun > 0,
        }))
        print(f"h={h:>2} {model:<20} R2={r2_score(actual, pred):.4f} "
              f"RMSE={np.sqrt(mean_squared_error(actual, pred)):.2f} "
              f"MAE={mean_absolute_error(actual, pred):.2f}  n={len(actual)}")

    tidy = pd.concat(frames, ignore_index=True)
    tidy.to_csv(ROOT / "results" / "best_model_predictions.csv", index=False)

    wide = None
    for h in HORIZONS:
        blk = (tidy[tidy["horizon_h"] == h]
               .set_index("issue_time_utc")[["actual_kW", "predicted_kW"]]
               .rename(columns={"actual_kW": f"actual_h{h}_kW",
                                "predicted_kW": f"predicted_h{h}_kW"}))
        wide = blk if wide is None else wide.join(blk)
    wide.reset_index().to_csv(ROOT / "results" / "best_model_predictions_wide.csv", index=False)

    print(f"\nRows: {len(tidy)} tidy ({len(wide)} issue times x {len(HORIZONS)} horizons)")
    print("Saved results/best_model_predictions.csv and results/best_model_predictions_wide.csv")


if __name__ == "__main__":
    main()
