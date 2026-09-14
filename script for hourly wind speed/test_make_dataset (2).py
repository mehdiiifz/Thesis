"""Tests for build_supervised: no leakage, exogenous lag alignment, shapes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_dataset import build_supervised  # noqa: E402


def _toy(n=200):
    """Strictly increasing series: value == position in the time index."""
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return pd.DataFrame({"ws100": np.arange(n, dtype=float)}, index=idx)


def _toy_exog(n=200):
    df = _toy(n)
    df["temp_c"] = np.arange(n, dtype=float) + 1000.0
    df["pres_hpa"] = np.arange(n, dtype=float) + 2000.0
    df["dir"] = 270.0  # wind from due west -> u = +ws100, v = 0
    return df


def test_no_leakage_features_vs_targets():
    """Newest feature is the value at t; earliest target is the value at t+1."""
    df = _toy()
    X, Y, _ = build_supervised(df, horizon=6, n_lags=8)
    vals = df["ws100"]
    # value increases with time, so max over features = most recent feature
    assert (X.max(axis=1).to_numpy() == vals.loc[X.index].to_numpy()).all()
    # and min over targets = earliest target
    t_plus_1 = vals.shift(-1).loc[Y.index].to_numpy()
    assert (Y.min(axis=1).to_numpy() == t_plus_1).all()
    # every feature strictly precedes every target
    assert (X.max(axis=1).to_numpy() < Y.min(axis=1).to_numpy()).all()


def test_exog_lag0_at_issue_time_never_later():
    """Exogenous lag-0 columns equal the value AT the issue time t."""
    df = _toy_exog()
    X, _, _ = build_supervised(df, horizon=4, n_lags=8, features="full")
    assert (X["temp_c_lag0"].to_numpy()
            == df["temp_c"].loc[X.index].to_numpy()).all()
    assert (X["pres_hpa_lag1"].to_numpy()
            == df["pres_hpa"].shift(1).loc[X.index].to_numpy()).all()
    # dir=270 (from west): u = +ws100 at the same time, v = 0
    assert np.allclose(X["u_lag0"].to_numpy(),
                       df["ws100"].loc[X.index].to_numpy())
    assert np.allclose(X["v_lag0"].to_numpy(), 0.0, atol=1e-9)
    # no exogenous column may correlate with FUTURE values: lag-0 must not
    # equal the value one hour after the issue time
    assert not (X["temp_c_lag0"].to_numpy()
                == df["temp_c"].shift(-1).loc[X.index].to_numpy()).any()


@pytest.mark.parametrize("H", [1, 2, 4, 6, 12, 24])
def test_shapes_match_horizon(H):
    n, n_lags = 300, 24
    X, Y, feats = build_supervised(_toy(n), horizon=H, n_lags=n_lags)
    assert isinstance(Y, pd.DataFrame)  # 2-D also for H=1
    assert Y.shape == (n - n_lags + 1 - H, H)
    assert list(Y.columns) == [f"t+{h}h" for h in range(1, H + 1)]
    assert X.shape == (Y.shape[0], n_lags)
    assert feats == list(X.columns)
    assert (X.index == Y.index).all()
