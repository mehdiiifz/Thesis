"""Stage 7: hybrid stacking — tree ensembles blended with the LSTM-v2.

Trios [RF+XGB+LSTM] and [ET+XGB+LSTM] per horizon, using the corrected
blending procedure from stage 6 v2: Ridge(alpha=1.0, fit_intercept=False,
positive=True) fit on the holdout block's base predictions, applied to the
test predictions, clipped to [0, 1000] kW and night-masked (prediction = 0
wherever H_sun(t+h) <= 0). Rows where the LSTM prediction is NaN are dropped
consistently from meta-training and evaluation, with the count reported.

Also computes the diagnostic behind the result: the Pearson correlation
matrix of the base models' holdout ERRORS (prediction - actual) for RF, ET,
XGB, LSTM at h=1 and h=24 -> results/error_correlations.csv.

Outputs: results/hybrid.csv (metrics, meta-weights, and explicit comparisons
against the trio's best single member and the corresponding stage-6-v2
two-tree stack) and results/error_correlations.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]
TRIOS = [("RF", "XGB", "LSTM"), ("ET", "XGB", "LSTM")]
CAP_KW = 1000.0


def main() -> None:
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    tree_preds = pd.read_pickle(ROOT / "data" / "tree_preds.pkl")
    lstm_preds = pd.read_pickle(ROOT / "data" / "lstm_preds.pkl")

    def base(model: str, h: int, block: str) -> pd.Series:
        return (lstm_preds[h] if model == "LSTM" else tree_preds[model][h])[block]

    singles = pd.concat([
        pd.read_csv(ROOT / "results" / "single_trees.csv"),
        pd.read_csv(ROOT / "results" / "single_lstm_v2.csv").assign(model="LSTM"),
    ]).set_index(["model", "horizon"])
    stacks2 = pd.read_csv(ROOT / "results" / "stacks_v2.csv").set_index(["model", "horizon"])
    pers24_rmse = singles.xs("PERS24")["RMSE"]

    # --- hybrid stacks -----------------------------------------------------
    rows = []
    for trio in TRIOS:
        name = "+".join(trio)
        pair_name = f"{trio[0]}+{trio[1]}"
        print(f"\n{name} meta-weights ({', '.join('w_' + m for m in trio)}):")
        for h in HORIZONS:
            H = np.column_stack([base(m, h, "holdout") for m in trio])
            T = np.column_stack([base(m, h, "test") for m in trio])
            y_hold = splits[h]["y_holdout"].to_numpy()
            y_test = splits[h]["y_test"].to_numpy()

            keep_h, keep_t = ~np.isnan(H).any(axis=1), ~np.isnan(T).any(axis=1)
            n_drop = int((~keep_h).sum() + (~keep_t).sum())
            H, y_hold = H[keep_h], y_hold[keep_h]
            T, y_test = T[keep_t], y_test[keep_t]

            meta = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(H, y_hold)
            blend = np.clip(meta.predict(T), 0, CAP_KW)
            night = (splits[h]["X_test"]["Hsun_target"] <= 0).to_numpy()[keep_t]
            blend[night] = 0.0

            rmse = float(np.sqrt(mean_squared_error(y_test, blend)))
            mae = mean_absolute_error(y_test, blend)

            best = min(trio, key=lambda m: singles.loc[(m, h), "RMSE"])
            b_rmse, b_mae = singles.loc[(best, h), ["RMSE", "MAE"]]
            s_rmse, s_mae = stacks2.loc[(pair_name, h), ["RMSE", "MAE"]]

            rows.append({
                "model": name, "horizon": h, "n_nan_dropped": n_drop,
                "R2": r2_score(y_test, blend), "RMSE": rmse, "MAE": mae,
                "R2_daytime": r2_score(y_test[~night], blend[~night]),
                "skill_vs_PERS24": 1 - rmse / pers24_rmse[h],
                **{f"w_{m}": w for m, w in zip(trio, meta.coef_)},
                "best_member": best, "best_member_RMSE": b_rmse, "best_member_MAE": b_mae,
                "delta_RMSE_vs_best": rmse - b_rmse, "delta_MAE_vs_best": mae - b_mae,
                "stack2_RMSE": s_rmse, "stack2_MAE": s_mae,
                "delta_RMSE_vs_stack2": rmse - s_rmse, "delta_MAE_vs_stack2": mae - s_mae,
            })
            print(f"  h={h:>2}: " + " ".join(f"{w:+.3f}" for w in meta.coef_) +
                  f"   RMSE={rmse:6.2f} (best {best} {b_rmse:6.2f}, "
                  f"2-stack {s_rmse:6.2f})  nan_dropped={n_drop}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "hybrid.csv", index=False)

    # --- error decorrelation diagnostic (holdout) --------------------------
    corr_frames = []
    for h in [1, 24]:
        y_hold = splits[h]["y_holdout"]
        errs = pd.DataFrame({m: base(m, h, "holdout") - y_hold for m in ["RF", "ET", "XGB", "LSTM"]})
        corr = errs.corr().round(4)
        corr_frames.append(corr.assign(horizon=h).rename_axis("model").reset_index())
        print(f"\nHoldout error correlations, h={h}:")
        print(corr.to_string())
    pd.concat(corr_frames).to_csv(ROOT / "results" / "error_correlations.csv", index=False)

    print("\n" + res.round(4).to_string(index=False))
    print("\nSaved results/hybrid.csv and results/error_correlations.csv")


if __name__ == "__main__":
    main()
