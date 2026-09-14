"""Tests for the supervised dataset builder: no leakage, correct shapes.

The central trick is a toy frame in which ws100 equals the row position. Every
feature value is then directly readable as "which row did this come from",
so leakage becomes a numeric assertion rather than a matter of inspection:
no feature may exceed the issue position, and no target may fall at or below it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_dataset import (  # noqa: E402
    COARSE_BLOCKS, EXOG_FULL, EXOG_LAGS, EXOG_TURB, FEATURE_SETS, FINE_LAGS,
    TARGET, build_supervised, classify_feature, max_lag,
    multiresolution_lag_design,
)

HORIZON_STEPS = [1, 2, 3, 6, 9, 12, 24, 48]
DEEPEST = max_lag()          # 143 with the documented design


def _toy(n=600):
    """Toy frame where ws100 == row position, with all exogenous channels."""
    idx = pd.date_range("2020-01-01", periods=n, freq="10min")
    pos = np.arange(n, dtype=float)
    return pd.DataFrame({
        TARGET: pos,
        # Distinct offsets so a mix-up between channels is visible.
        "dir": (pos * 0.7) % 360.0,
        "sd": pos + 1000.0,
        "dsd": pos + 2000.0,
        "ri": pos + 3000.0,
        "vert_ws": pos + 4000.0,
        "temp_c": pos + 5000.0,
        "pres_hpa": pos + 6000.0,
        # Banned channels are present in the frame and must stay out of X.
        "ws107": pos + 7000.0,
        "ws95": pos + 8000.0,
        "gust3s": pos + 9000.0,
    }, index=idx)


def _positions(df, X):
    """Row position of each issue time in X."""
    return np.array([df.index.get_loc(t) for t in X.index])


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------
@pytest.mark.parametrize("features", FEATURE_SETS)
def test_no_feature_comes_from_after_the_issue_time(features):
    """Every ws100-derived feature must originate at or before t."""
    df = _toy()
    H = 6
    X, Y, _ = build_supervised(df, horizon_steps=H, features=features)
    pos = _positions(df, X)

    wind_cols = [c for c in X.columns
                 if c.startswith(f"{TARGET}_lag") or c.startswith(f"{TARGET}_mean_lag")]
    # ws100 == row position, so any wind feature is an average of row positions
    # and can never exceed the issue position without reading the future.
    for col in wind_cols:
        assert (X[col].to_numpy() <= pos).all(), f"{col} contains values from after t"


@pytest.mark.parametrize("features", FEATURE_SETS)
def test_targets_are_strictly_in_the_future(features):
    df = _toy()
    H = 6
    X, Y, _ = build_supervised(df, horizon_steps=H, features=features)
    pos = _positions(df, X)
    for step in range(1, H + 1):
        assert (Y[f"h{step}"].to_numpy() == pos + step).all()


def test_fine_lags_are_exact_row_offsets():
    df = _toy()
    X, _, _ = build_supervised(df, horizon_steps=1)
    pos = _positions(df, X)
    for lag in range(FINE_LAGS):
        assert (X[f"{TARGET}_lag{lag}"].to_numpy() == pos - lag).all()


def test_banned_channels_never_enter_the_feature_matrix():
    df = _toy()
    for features in FEATURE_SETS:
        X, _, names = build_supervised(df, horizon_steps=3, features=features)
        for banned in ("ws107", "ws95", "gust3s"):
            assert not [c for c in names if c.startswith(banned)]
        # The banned columns carry values >= 7000; nothing in X may reach that.
        assert X.to_numpy().max() < 7000.0


# --------------------------------------------------------------------------
# coarse blocks
# --------------------------------------------------------------------------
def test_coarse_blocks_contain_no_values_from_after_t():
    """A block [a, b] must average exactly rows t-a .. t-b, nothing newer."""
    df = _toy()
    X, _, _ = build_supervised(df, horizon_steps=1)
    pos = _positions(df, X)

    for lag_from, lag_to in COARSE_BLOCKS:
        col = f"{TARGET}_mean_lag{lag_from}_{lag_to}"
        got = X[col].to_numpy()
        # Mean of the consecutive positions p-lag_to .. p-lag_from.
        expected = pos - (lag_from + lag_to) / 2.0
        assert np.allclose(got, expected), f"{col} does not match its lag window"
        # The newest row any block may touch is p - lag_from.
        assert (got <= pos - lag_from).all(), f"{col} reaches past t-{lag_from}"


def test_coarse_blocks_are_contiguous_and_cover_the_full_range():
    design = multiresolution_lag_design()
    assert design[0].kind == "fine"
    assert design[0].lag_from == 0 and design[0].lag_to == FINE_LAGS - 1
    for previous, block in zip(design, design[1:]):
        assert block.lag_from == previous.lag_to + 1, "design has a gap or overlap"
    assert design[-1].lag_to == DEEPEST


def test_lag_design_rejects_non_contiguous_blocks():
    with pytest.raises(ValueError):
        multiresolution_lag_design(fine_lags=12, coarse_blocks=((13, 17),))
    with pytest.raises(ValueError):
        multiresolution_lag_design(fine_lags=12, coarse_blocks=((12, 17), (20, 23)))
    with pytest.raises(ValueError):
        multiresolution_lag_design(fine_lags=12, coarse_blocks=((17, 12),))


# --------------------------------------------------------------------------
# exogenous channels
# --------------------------------------------------------------------------
def test_exogenous_lag0_equals_value_at_issue_time():
    """Lag 0 of an exogenous channel is the observation made AT t, not after."""
    df = _toy()
    X, _, _ = build_supervised(df, horizon_steps=4, features="full_turb")
    for name in ("temp_c", "pres_hpa", *EXOG_TURB):
        expected = df[name].reindex(X.index).to_numpy()
        assert np.allclose(X[f"{name}_lag0"].to_numpy(), expected), \
            f"{name}_lag0 does not equal the value at the issue time"


def test_exogenous_lags_step_back_one_row_at_a_time():
    df = _toy()
    X, _, _ = build_supervised(df, horizon_steps=4, features="full_turb")
    pos = _positions(df, X)
    for name, offset in (("temp_c", 5000.0), ("pres_hpa", 6000.0),
                         ("sd", 1000.0), ("ri", 3000.0)):
        for lag in range(EXOG_LAGS):
            assert (X[f"{name}_lag{lag}"].to_numpy() == pos - lag + offset).all()


def test_uv_components_follow_the_meteorological_convention():
    df = _toy()
    X, _, _ = build_supervised(df, horizon_steps=1, features="full")
    rad = np.deg2rad(df["dir"].reindex(X.index).to_numpy())
    ws = df[TARGET].reindex(X.index).to_numpy()
    assert np.allclose(X["u_lag0"].to_numpy(), -ws * np.sin(rad))
    assert np.allclose(X["v_lag0"].to_numpy(), -ws * np.cos(rad))


def test_turbulence_intensity_is_sd_over_ws_at_the_issue_time():
    df = _toy()
    X, _, _ = build_supervised(df, horizon_steps=1, features="full_turb")
    sd = df["sd"].reindex(X.index).to_numpy()
    ws = df[TARGET].reindex(X.index).to_numpy()
    assert np.allclose(X["ti_lag0"].to_numpy(), sd / ws)


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("H", HORIZON_STEPS)
@pytest.mark.parametrize("features", FEATURE_SETS)
def test_shapes_for_every_horizon(H, features):
    n = 600
    df = _toy(n)
    X, Y, names = build_supervised(df, horizon_steps=H, features=features)

    assert len(X) == len(Y)
    assert (X.index == Y.index).all()
    assert Y.shape[1] == H
    assert list(Y.columns) == [f"h{s}" for s in range(1, H + 1)]
    assert list(X.columns) == names
    # Rows lost: DEEPEST at the start for the lags, H at the end for targets.
    assert len(X) == n - DEEPEST - H


@pytest.mark.parametrize("H", HORIZON_STEPS)
def test_horizon_one_and_all_others_produce_no_nans(H):
    X, Y, _ = build_supervised(_toy(), horizon_steps=H, features="full_turb")
    assert not X.isna().any().any()
    assert not Y.isna().any().any()
    assert len(X) > 0


def test_feature_counts_match_the_documented_design():
    df = _toy()
    expected_wind = FINE_LAGS + len(COARSE_BLOCKS)
    n_time = 4
    counts = {}
    for features in FEATURE_SETS:
        _, _, names = build_supervised(df, horizon_steps=1, features=features)
        counts[features] = len(names)

    assert counts["lags_only"] == expected_wind
    assert counts["full"] == expected_wind + n_time + len(EXOG_FULL) * EXOG_LAGS
    assert counts["full_turb"] == (counts["full"]
                                   + len(EXOG_TURB) * EXOG_LAGS
                                   + 1)          # + TI at lag 0


def test_feature_sets_are_nested():
    df = _toy()
    _, _, lags_only = build_supervised(df, horizon_steps=1, features="lags_only")
    _, _, full = build_supervised(df, horizon_steps=1, features="full")
    _, _, full_turb = build_supervised(df, horizon_steps=1, features="full_turb")
    assert set(lags_only) < set(full) < set(full_turb)


def test_every_feature_is_classifiable():
    df = _toy()
    for features in FEATURE_SETS:
        _, _, names = build_supervised(df, horizon_steps=1, features=features)
        for name in names:
            classify_feature(name)          # raises if unclassified


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------
def test_invalid_arguments_are_rejected():
    df = _toy()
    with pytest.raises(ValueError):
        build_supervised(df, horizon_steps=0)
    with pytest.raises(ValueError):
        build_supervised(df, horizon_steps=1, features="nonsense")
    with pytest.raises(ValueError):
        build_supervised(df.drop(columns=[TARGET]), horizon_steps=1)
