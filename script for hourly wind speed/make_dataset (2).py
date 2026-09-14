"""Build supervised matrices for direct multi-horizon wind forecasting.

For issue time t, features use only data at times <= t and targets are
ws100 at t+1h ... t+Hh (strictly after t). One (X, Y) pair per horizon
(direct multi-output strategy, HORIZONS = [1, 2, 4, 6, 12, 24]).

Feature sets (see CLAUDE.md):
- "lags_only": past `n_lags` hourly ws100 values (lag 0 = value at t).
- "full": lags + hour/day-of-year sin-cos at t + lags 0-2 of temp_c,
  pres_hpa, u, v (u/v computed from ws100 and meteorological direction).
ws107, ws95, gust3s, n_obs are never used.

Run from project root: python src/make_dataset.py  (smoke demo on real data)
"""
import numpy as np
import pandas as pd

HORIZONS = [1, 2, 4, 6, 12, 24]
N_EXOG_LAGS = 3


def wind_uv(ws: pd.Series, dir_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Zonal/meridional components from speed and meteorological direction.

    `dir_deg` is the direction the wind blows FROM, clockwise from north,
    so u = -ws*sin(theta), v = -ws*cos(theta).
    """
    theta = np.deg2rad(dir_deg)
    return -ws * np.sin(theta), -ws * np.cos(theta)


def build_supervised(df: pd.DataFrame, horizon: int, n_lags: int = 24,
                     features: str = "lags_only",
                     target_col: str = "ws100"):
    """Return (X, Y, feature_cols) for one forecast horizon.

    X: DataFrame of features, one row per issue time t (only times <= t).
    Y: DataFrame with columns t+1h ... t+{horizon}h (always 2-D, also for
       horizon=1), values = target_col at those future times.
    Rows with any incomplete lag window or incomplete target window are
    dropped. Index = forecast issue times, shared by X and Y.
    """
    assert features in ("lags_only", "full"), features
    assert df.index.is_monotonic_increasing, "index must be sorted"
    s = df[target_col].astype(float)

    cols = {f"{target_col}_lag{k}": s.shift(k) for k in range(n_lags)}
    if features == "full":
        hour = df.index.hour.to_numpy()
        doy = df.index.dayofyear.to_numpy()
        cols["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        cols["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        cols["doy_sin"] = np.sin(2 * np.pi * doy / 366)
        cols["doy_cos"] = np.cos(2 * np.pi * doy / 366)
        u, v = wind_uv(s, df["dir"].astype(float))
        for name, series in [("temp_c", df["temp_c"].astype(float)),
                             ("pres_hpa", df["pres_hpa"].astype(float)),
                             ("u", u), ("v", v)]:
            for k in range(N_EXOG_LAGS):
                cols[f"{name}_lag{k}"] = series.shift(k)

    X = pd.DataFrame(cols, index=df.index)
    Y = pd.DataFrame({f"t+{h}h": s.shift(-h) for h in range(1, horizon + 1)},
                     index=df.index)
    valid = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X.loc[valid], Y.loc[valid]
    X.index.name = Y.index.name = "timestamp"
    return X, Y, list(X.columns)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load_data import load_hourly

    df = load_hourly()
    for feats in ("lags_only", "full"):
        for H in HORIZONS:
            X, Y, names = build_supervised(df, horizon=H, features=feats)
            print(f"{feats:9s} H={H:2d}: X {X.shape}, Y {Y.shape}, "
                  f"issue times {X.index.min()} -> {X.index.max()}")
        print(f"  features ({len(names)}): {names[:3]} ... {names[-3:]}")
