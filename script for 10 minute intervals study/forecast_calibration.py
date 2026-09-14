"""Calibrate the forecast uncertainty bands used by the scheduling decision tree.

Run from the project root:

    python src/forecast_calibration.py

Writes results/forecast_residual_quantiles.csv

Why empirical quantiles and not mean +/- z * RMSE
-------------------------------------------------
The decision tree needs a pessimistic wind trajectory for the turn-ON branch and
an optimistic one for the turn-OFF branch. A Gaussian band built from the RMSE
would be wrong in exactly the place that matters: section 5 of the results
report measured that the predicted spread shrinks with lead time (std ratio
0,993 at 10 minutes falling to 0,685 at 480) and that the predicted minimum
rises from 0,29 to 1,98 m/s while the measured minimum stays at 0,23. In other
words the model stops forecasting calms at long lead times, so a symmetric band
would under-warn about precisely the lulls that empty the product storage.

This script therefore takes the residuals e = y_true - y_pred from the saved
per-fold test predictions and reports their empirical quantiles per lead step.
The tree adds those quantiles to the point forecast, so the bands inherit the
real asymmetry instead of assuming it away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_data import load_10min  # noqa: E402
from report_utils import HORIZON_STEPS, RESULTS, STEP_MINUTES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRED = RESULTS / "predictions"
OUT = RESULTS / "forecast_residual_quantiles.csv"

MODEL = "lstm"
FEATURES = "reduced"
N_FOLDS = 4
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def residuals(target: pd.Series, horizon: int) -> np.ndarray:
    """Stack (n, horizon) residuals over all folds for one horizon model."""
    path = PRED / f"{MODEL}_{FEATURES}_h{horizon}.npz"
    blob = np.load(path, allow_pickle=True)
    position = {t: i for i, t in enumerate(target.index)}
    values = target.to_numpy(np.float64)

    stacked = []
    for fold in range(N_FOLDS):
        pred = blob[f"fold{fold}"].astype(np.float64)
        issued = pd.to_datetime(blob[f"index{fold}"])
        rows = np.array([position[t] for t in issued])
        # Column k holds the forecast for the issue time plus k+1 steps.
        steps = np.arange(1, horizon + 1)
        truth = values[rows[:, None] + steps[None, :]]
        stacked.append(truth - pred)
    return np.vstack(stacked)


def main() -> None:
    frame = load_10min()
    target = frame["ws100"]

    records = []
    for horizon in HORIZON_STEPS:
        err = residuals(target, horizon)
        for k in range(horizon):
            column = err[:, k]
            row = {
                "horizon_steps": horizon,
                "horizon_min": horizon * STEP_MINUTES,
                "step": k + 1,
                "lead_min": (k + 1) * STEP_MINUTES,
                "n": int(column.size),
                "bias": float(column.mean()),
                "rmse": float(np.sqrt((column ** 2).mean())),
            }
            for q in QUANTILES:
                row[f"q{int(q * 100):02d}"] = float(np.quantile(column, q))
            records.append(row)

    out = pd.DataFrame(records)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(out)} rows)")

    longest = out[out["horizon_steps"] == max(HORIZON_STEPS)]
    print("\nresidual band at the 480-minute model, m/s")
    print(f"{'lead':>6} {'q05':>8} {'q50':>8} {'q95':>8} {'rmse':>8} "
          f"{'skew':>8}")
    for lead in (10, 60, 120, 240, 480):
        r = longest[longest["lead_min"] == lead].iloc[0]
        skew = abs(r["q05"]) - abs(r["q95"])
        print(f"{int(lead):>6} {r['q05']:>8.3f} {r['q50']:>8.3f} "
              f"{r['q95']:>8.3f} {r['rmse']:>8.3f} {skew:>8.3f}")
    print("\nA negative skew column means the model over-forecasts more often "
          "than it under-forecasts,\ni.e. the pessimistic band has to be the "
          "wider one.")


if __name__ == "__main__":
    main()
