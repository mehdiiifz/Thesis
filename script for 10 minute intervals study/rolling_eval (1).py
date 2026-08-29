"""Rolling-origin (expanding-window) evaluation for time series forecasting.

Why: with one year of data, a single chronological 70/15/15 split tests ONLY
on winter (Nov-Jan). Rolling-origin evaluation retrains on an expanding window
and tests on several blocks spread across seasons, reporting mean +/- std of
the metrics. This is the standard, examiner-proof protocol (Tashman 2000;
Bergmeir & Benitez 2012).

The splitting is purely positional (by row index), so the same code works
unchanged at any sampling resolution -- hourly, 10-minute, or otherwise.
All window sizes below are expressed in rows/steps, never in wall-clock hours.

Design for ~1 year of data at a fixed resolution:
- n_folds test blocks, each `test_frac` of the series (default 4 x 12.5%),
  placed back-to-back after `min_train_frac` (default 40%). With this data
  the blocks cover roughly summer, early autumn, late autumn, and winter.
  (Spring cannot be tested with an expanding window on a single year -> state
  this as a limitation; it disappears once multi-year data is available.)
- A `gap` of `horizon` rows is left between the training window and each
  test block so no training sample has target values inside the test block
  (prevents boundary leakage). One row = one step of the series resolution.
- Within each training window, the last `val_frac` is used as validation.

Drop-in usage (replaces chrono_split):

    from rolling_eval import rolling_origin_folds
    for fold in rolling_origin_folds(X, Y, horizon=H):
        model.fit(fold["train"][0], fold["train"][1])
        ...evaluate on fold["val"], fold["test"]...

Each fold dict: {"fold": k, "train": (X,Y), "val": (X,Y), "test": (X,Y),
                 "test_period": (start, end)}
"""
from typing import Iterator
import numpy as np
import pandas as pd


def rolling_origin_folds(X: pd.DataFrame, Y: pd.DataFrame, horizon: int,
                         n_folds: int = 4, min_train_frac: float = 0.40,
                         test_frac: float = 0.125,
                         val_frac: float = 0.15) -> Iterator[dict]:
    """Yield expanding-window folds with a leakage gap of `horizon` rows.

    X, Y must share a DatetimeIndex of forecast-issue times (as produced by
    build_supervised). Targets of issue time t span the next `horizon` steps
    after t, i.e. rows (t, t+horizon] in row positions.

    Splitting is by row position only, so `horizon` and every window size are
    counted in rows/steps of the series, whatever its resolution.
    """
    assert (X.index == Y.index).all(), "X and Y must be aligned"
    n = len(X)
    used = min_train_frac + n_folds * test_frac
    assert used <= 1.0 + 1e-9, (
        f"min_train_frac + n_folds*test_frac = {used:.3f} > 1; reduce folds "
        "or fractions")

    for k in range(n_folds):
        test_start = int(n * (min_train_frac + k * test_frac))
        test_end = min(int(n * (min_train_frac + (k + 1) * test_frac)), n)
        # training issue times must have ALL targets strictly before the
        # first test issue time -> last train issue index = test_start - horizon - 1
        train_end = max(test_start - horizon, 1)
        val_start = int(train_end * (1 - val_frac))
        yield {
            "fold": k,
            "train": (X.iloc[:val_start], Y.iloc[:val_start]),
            "val": (X.iloc[val_start:train_end], Y.iloc[val_start:train_end]),
            "test": (X.iloc[test_start:test_end], Y.iloc[test_start:test_end]),
            "test_period": (X.index[test_start], X.index[test_end - 1]),
        }


def summarize_folds(fold_scores: list[dict]) -> dict:
    """Mean and std of each metric across folds.

    fold_scores: list of dicts like {"rmse": ..., "mae": ..., "r": ..., "nrmse": ...}
    Returns {"rmse_mean": ..., "rmse_std": ..., ...}
    """
    out = {}
    keys = fold_scores[0].keys()
    for key in keys:
        vals = np.array([f[key] for f in fold_scores], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def nrmse(rmse: float, y_true_test: np.ndarray) -> float:
    """Normalized RMSE = RMSE / mean of observed test values.

    Makes results comparable across sites with different mean wind speed
    (e.g., this site ~6.7 m/s vs Mito's Bahrain site ~3.85 m/s).
    """
    return float(rmse / np.asarray(y_true_test).mean())


if __name__ == "__main__":
    # Smoke test on real data with a fast model, H=6
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load_data import load_hourly
    from make_dataset import build_supervised
    from metrics_utils import scores
    from sklearn.ensemble import ExtraTreesRegressor

    df = load_hourly()
    H = 6
    X, Y, _ = build_supervised(df, horizon=H, features="lags_only")
    per_fold = []
    for fold in rolling_origin_folds(X, Y, horizon=H):
        m = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        m.fit(fold["train"][0].values, fold["train"][1].values)
        yt = fold["test"][1].values
        sc = scores(yt, m.predict(fold["test"][0].values))
        sc["nrmse"] = nrmse(sc["rmse"], yt)
        per_fold.append(sc)
        t0, t1 = fold["test_period"]
        print(f"fold {fold['fold']}: test {t0.date()} -> {t1.date()}  "
              f"RMSE={sc['rmse']:.3f}  NRMSE={sc['nrmse']:.3f}  R={sc['r']:.3f}")
    s = summarize_folds(per_fold)
    print(f"\nET H={H}: RMSE {s['rmse_mean']:.3f} +/- {s['rmse_std']:.3f}, "
          f"NRMSE {s['nrmse_mean']:.3f} +/- {s['nrmse_std']:.3f}, "
          f"R {s['r_mean']:.3f} +/- {s['r_std']:.3f}")
