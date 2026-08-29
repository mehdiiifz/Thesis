"""Stage 6 (v2): two-model blending per horizon, reusing stage-4 predictions.

For each pair (RF+XGB, ET+XGB) and horizon: a constrained Ridge meta-learner
(alpha=1.0, fit_intercept=False, positive=True) is fit on the pair's holdout
predictions against the holdout truth, then applied to the pair's test
predictions. No intercept and non-negative weights mean the blend is a pure
weighted mix of the members — when both members predict 0 kW (night), the
blend is exactly 0, never phantom power. On top of that, night masking forces
the output to 0 wherever H_sun(t+h) <= 0, the same rule as the LSTM-v2 fix.
Blended predictions are clipped to [0, 1000] kW.

The v1 run (plain Ridge with intercept, no masking) is preserved in
results/stacks.csv; this script now writes results/stacks_v2.csv, including
each pair's best single member's RMSE/MAE and the stack-minus-member deltas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]
PAIRS = [("RF", "XGB"), ("ET", "XGB")]
CAP_KW = 1000.0


def main() -> None:
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    preds = pd.read_pickle(ROOT / "data" / "tree_preds.pkl")
    singles = pd.read_csv(ROOT / "results" / "single_trees.csv").set_index(["model", "horizon"])
    pers24_rmse = singles.xs("PERS24")["RMSE"]

    rows = []
    for a, b in PAIRS:
        name = f"{a}+{b}"
        print(f"\n{name} meta-weights (w_{a}, w_{b}):")
        for h in HORIZONS:
            X_hold = np.column_stack([preds[a][h]["holdout"], preds[b][h]["holdout"]])
            X_test = np.column_stack([preds[a][h]["test"], preds[b][h]["test"]])
            y_hold = splits[h]["y_holdout"].to_numpy()
            y_test = splits[h]["y_test"].to_numpy()

            meta = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X_hold, y_hold)
            blend = np.clip(meta.predict(X_test), 0, CAP_KW)
            night = (splits[h]["X_test"]["Hsun_target"] <= 0).to_numpy()
            blend[night] = 0.0

            day = ~night
            rmse = float(np.sqrt(mean_squared_error(y_test, blend)))
            mae = mean_absolute_error(y_test, blend)

            # best single member of this pair, by test RMSE (from stage 4)
            best = min((a, b), key=lambda m: singles.loc[(m, h), "RMSE"])
            best_rmse = singles.loc[(best, h), "RMSE"]
            best_mae = singles.loc[(best, h), "MAE"]

            rows.append({
                "model": name, "horizon": h,
                "R2": r2_score(y_test, blend), "RMSE": rmse, "MAE": mae,
                "R2_daytime": r2_score(y_test[day], blend[day]),
                "skill_vs_PERS24": 1 - rmse / pers24_rmse[h],
                "w_first": meta.coef_[0], "w_second": meta.coef_[1],
                "best_member": best, "best_member_RMSE": best_rmse,
                "best_member_MAE": best_mae,
                "delta_RMSE": rmse - best_rmse, "delta_MAE": mae - best_mae,
            })
            print(f"  h={h:>2}: {meta.coef_[0]:+.3f} {meta.coef_[1]:+.3f}   "
                  f"stack RMSE={rmse:6.2f} vs {best} {best_rmse:6.2f} "
                  f"(dRMSE={rmse - best_rmse:+5.2f}, dMAE={mae - best_mae:+5.2f})")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "stacks_v2.csv", index=False)
    print("\n" + res.round(4).to_string(index=False))
    print("\nSaved results/stacks_v2.csv")


if __name__ == "__main__":
    main()
