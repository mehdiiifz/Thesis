"""Stage 3: feature engineering and chronological splits, direct multi-horizon strategy.

For each horizon h in HORIZONS a separate supervised dataset is built with
target y = P_kW.shift(-h) (the "direct" strategy: one model per horizon, no
recursive feeding of predictions). Every feature is known at forecast-issue
time t: measured variables (P, G(i), T2m, WS10m) enter only as lags or
rolling windows ending at t, while target-time features are restricted to
deterministic quantities (sun height, calendar) plus P(t+h-24), which for
h <= 24 lies in the past.

All horizons share one row index (rows valid for every horizon), so sample
counts are identical across horizons and results are directly comparable.
Saves data/splits.pkl (git-ignored; regenerates in seconds).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]
TRAIN_END = pd.Timestamp("2013-06-30 23:00", tz="UTC")   # train: start .. 2013-06-30
TEST_START = pd.Timestamp("2013-07-01 00:00", tz="UTC")  # test: 2013-07-01 .. end
HOLDOUT_FRAC = 0.15                                      # last 15% of train


def issue_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that depend only on data up to and including issue time t."""
    p, gi = df["P_kW"], df["G(i)"]
    X = pd.DataFrame(index=df.index)
    for lag in [0, 1, 2, 3, 23, 24, 25, 48]:                      # A) P lags
        X[f"P_lag{lag}"] = p.shift(lag)
    X["P_roll3_mean"] = p.rolling(3).mean()                       # B) rolling, windows end at t
    X["P_roll24_mean"] = p.rolling(24).mean()
    X["P_roll24_max"] = p.rolling(24).max()
    X["Gi_lag0"] = gi                                             # C) measured weather at/before t
    X["Gi_lag1"] = gi.shift(1)
    X["Gi_lag24"] = gi.shift(24)
    X["T2m_lag0"] = df["T2m"]
    X["WS10m_lag0"] = df["WS10m"]
    return X


def target_time_features(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """D) Deterministic features evaluated at the target time t+h, plus P(t+h-24).

    Legal because sun position and calendar are computable in advance, and
    because t+h-24 is in the past (or exactly t) for every h <= 24.
    """
    assert h <= 24 and 24 - h >= 0, f"P(t+h-24) would be in the future for h={h}"
    X = pd.DataFrame(index=df.index)
    X["Hsun_target"] = df["H_sun"].shift(-h)
    target_time = df.index + pd.Timedelta(hours=h)
    hour, doy = target_time.hour, target_time.dayofyear
    X["sin_hour_target"] = np.sin(2 * np.pi * hour / 24)
    X["cos_hour_target"] = np.cos(2 * np.pi * hour / 24)
    X["sin_doy_target"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy_target"] = np.cos(2 * np.pi * doy / 365.25)
    X["P_samehour_yday_target"] = df["P_kW"].shift(24 - h)
    return X


def main() -> None:
    df = pd.read_pickle(ROOT / "data" / "clean.pkl")
    base = issue_time_features(df)

    # One shared row index across horizons: drop the warm-up rows (deepest lag,
    # 48 h) at the start and max(HORIZONS) rows at the end, where the farthest
    # target/H_sun(t+h) is unknown. Keeps all horizons on identical samples.
    valid = df.index[48:-max(HORIZONS)]

    splits: dict[int, dict[str, pd.DataFrame | pd.Series]] = {}
    for h in HORIZONS:
        X = pd.concat([base, target_time_features(df, h)], axis=1).loc[valid]
        y = df["P_kW"].shift(-h).rename(f"P_kW_t+{h}").loc[valid]
        assert not X.isna().any().any() and not y.isna().any()

        train_idx = X.index[X.index <= TRAIN_END]
        test_idx = X.index[X.index >= TEST_START]
        n_holdout = int(round(len(train_idx) * HOLDOUT_FRAC))
        fit_idx, holdout_idx = train_idx[:-n_holdout], train_idx[-n_holdout:]

        splits[h] = {
            "X_train": X.loc[fit_idx], "y_train": y.loc[fit_idx],
            "X_holdout": X.loc[holdout_idx], "y_holdout": y.loc[holdout_idx],
            "X_test": X.loc[test_idx], "y_test": y.loc[test_idx],
        }

    pd.to_pickle(splits, ROOT / "data" / "splits.pkl")

    s = splits[HORIZONS[0]]
    n_fit, n_hold, n_test = len(s["X_train"]), len(s["X_holdout"]), len(s["X_test"])
    print(f"Horizons: {HORIZONS}")
    print(f"Features ({s['X_train'].shape[1]}): {list(s['X_train'].columns)}")
    print(f"\nSamples per horizon (identical across horizons):")
    print(f"  train total : {n_fit + n_hold}  ({s['X_train'].index[0]} .. {s['X_holdout'].index[-1]})")
    print(f"    fit       : {n_fit}")
    print(f"    holdout   : {n_hold}  (last {HOLDOUT_FRAC:.0%} of train)")
    print(f"  test        : {n_test}  ({s['X_test'].index[0]} .. {s['X_test'].index[-1]})")
    for h in HORIZONS:  # sanity: same shapes everywhere
        assert len(splits[h]["X_train"]) == n_fit and len(splits[h]["X_test"]) == n_test
    print("\nSaved data/splits.pkl")


if __name__ == "__main__":
    main()
