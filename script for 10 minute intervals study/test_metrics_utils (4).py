"""Tests for the error metrics and result logging.

The r/r2 distinction is the point of several of these tests: a forecast can
track the truth perfectly (r = 1) while being badly wrong in level or scale
(r2 far below 1, possibly negative). Both must be reported, so both must be
computed correctly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics_utils import (  # noqa: E402
    METRIC_COLUMNS, PER_STEP_COLUMNS, fold_record, leaderboard_rolling,
    log_metrics, per_step_r2, per_step_records, per_step_rmse, per_step_scores,
    scores, summarize_records,
)

RNG = np.random.default_rng(42)


def _y(n=500):
    """A positive, varying truth series resembling wind speed."""
    return 7.0 + 3.0 * np.sin(np.linspace(0, 12, n)) + RNG.normal(0, 0.4, n)


# --------------------------------------------------------------------------
# r2 anchor cases
# --------------------------------------------------------------------------
def test_perfect_prediction_gives_r2_of_one():
    y = _y()
    s = scores(y, y.copy())
    assert s["r2"] == pytest.approx(1.0)
    assert s["rmse"] == pytest.approx(0.0)
    assert s["mae"] == pytest.approx(0.0)
    assert s["nrmse"] == pytest.approx(0.0)
    assert s["r"] == pytest.approx(1.0)


def test_constant_mean_prediction_gives_r2_of_zero():
    """Predicting the mean of the truth explains exactly none of its variance."""
    y = _y()
    pred = np.full_like(y, y.mean())
    s = scores(y, pred)
    assert s["r2"] == pytest.approx(0.0, abs=1e-12)
    # rmse then equals the population standard deviation of the truth.
    assert s["rmse"] == pytest.approx(y.std())
    # A constant forecast has no variance, so Pearson r is undefined.
    assert np.isnan(s["r"])


def test_half_scale_prediction_has_perfect_r_but_poor_r2():
    """The headline case: r cannot see a scale error, r2 can."""
    y = _y()
    pred = 0.5 * y
    s = scores(y, pred)

    assert s["r"] == pytest.approx(1.0), "r is invariant to affine rescaling"
    assert s["r2"] < 0.0, "halving the forecast must be worse than the mean"
    assert s["r2"] < s["r"]
    # It is also worse than predicting the mean, by construction.
    mean_pred = scores(y, np.full_like(y, y.mean()))
    assert s["rmse"] > mean_pred["rmse"]
    # r2 is emphatically NOT r squared here.
    assert not np.isclose(s["r2"], s["r"] ** 2)


def test_biased_prediction_keeps_r_but_loses_r2():
    """A pure offset leaves co-movement intact and still costs explained variance."""
    y = _y()
    s = scores(y, y + 2.0)
    assert s["r"] == pytest.approx(1.0)
    assert s["r2"] < 1.0
    assert s["mae"] == pytest.approx(2.0)


def test_r2_matches_its_definition_exactly():
    y = _y()
    pred = y + RNG.normal(0, 1.0, y.size)
    expected = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert scores(y, pred)["r2"] == pytest.approx(expected)


# --------------------------------------------------------------------------
# the other metrics
# --------------------------------------------------------------------------
def test_nrmse_on_a_known_array():
    """Hand-checkable: errors of +1 and -1 about a mean of 4."""
    y = np.array([2.0, 4.0, 4.0, 6.0])          # mean 4,0
    pred = np.array([3.0, 3.0, 5.0, 5.0])       # errors -1, +1, -1, +1
    s = scores(y, pred)
    assert s["rmse"] == pytest.approx(1.0)
    assert s["mae"] == pytest.approx(1.0)
    assert s["nrmse"] == pytest.approx(0.25)     # 1,0 / 4,0
    # sse = 4, sst = (4+0+0+4) = 8  ->  r2 = 0,5
    assert s["r2"] == pytest.approx(0.5)


def test_rmse_and_mae_differ_when_errors_are_uneven():
    y = np.zeros(4)
    pred = np.array([0.0, 0.0, 0.0, 4.0])
    s = scores(y, pred)
    assert s["mae"] == pytest.approx(1.0)
    assert s["rmse"] == pytest.approx(2.0), "rmse must punish the single large error"


def test_metrics_pool_over_lead_time_steps():
    """A 2-D (n, H) pair is flattened, so pooling equals scoring the flat arrays."""
    y = _y(600).reshape(200, 3)
    pred = y + RNG.normal(0, 0.5, y.shape)
    assert scores(y, pred) == pytest.approx(scores(y.ravel(), pred.ravel()))


# --------------------------------------------------------------------------
# per-step curves
# --------------------------------------------------------------------------
def test_per_step_curves_have_one_value_per_lead_time():
    y = _y(600).reshape(200, 3)
    pred = y + RNG.normal(0, 0.5, y.shape)
    assert per_step_rmse(y, pred).shape == (3,)
    assert per_step_r2(y, pred).shape == (3,)

    table = per_step_scores(y, pred)
    assert list(table["step"]) == [1, 2, 3]
    assert list(table["step_min"]) == [10, 20, 30]


def test_per_step_values_match_scoring_each_column():
    y = _y(600).reshape(200, 3)
    pred = y + RNG.normal(0, 0.5, y.shape)
    curve = per_step_rmse(y, pred)
    for j in range(3):
        assert curve[j] == pytest.approx(scores(y[:, j], pred[:, j])["rmse"])


def test_per_step_rmse_grows_with_added_noise():
    y = _y(600).reshape(200, 3)
    pred = y.copy()
    pred[:, 1] += 1.0
    pred[:, 2] += 2.0
    curve = per_step_rmse(y, pred)
    assert curve[0] < curve[1] < curve[2]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_shape_mismatch_and_empty_input_are_rejected():
    with pytest.raises(ValueError):
        scores(np.zeros(5), np.zeros(6))
    with pytest.raises(ValueError):
        scores(np.array([]), np.array([]))


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError):
        scores(np.array([1.0, np.nan]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        scores(np.array([1.0, 2.0]), np.array([1.0, np.inf]))


def test_constant_truth_gives_undefined_r2():
    s = scores(np.full(10, 5.0), np.full(10, 5.0))
    assert np.isnan(s["r2"]), "zero total sum of squares leaves r2 undefined"


# --------------------------------------------------------------------------
# records, summarising and logging
# --------------------------------------------------------------------------
def _four_folds():
    records, per_step = [], []
    for fold in range(4):
        y = _y(300).reshape(100, 3)
        pred = y + RNG.normal(0, 0.3 + 0.1 * fold, y.shape)
        records.append(fold_record("xgb", "full_turb", 3, fold, "test",
                                   y, pred, run_time=1.5))
        per_step.extend(per_step_records("xgb", "full_turb", 3, fold, "test",
                                         y, pred, run_time=1.5))
    return records, per_step


def test_fold_record_carries_both_horizon_columns():
    rec = fold_record("persistence", "lags_only", 48, 0, "test",
                      _y(96).reshape(2, 48), _y(96).reshape(2, 48))
    assert rec["horizon_steps"] == 48
    assert rec["horizon_min"] == 480
    assert set(("rmse", "mae", "r", "r2", "nrmse")) <= set(rec)


def test_summarize_records_adds_mean_and_std_rows_for_every_metric():
    records, _ = _four_folds()
    summary = summarize_records(records)
    assert len(summary) == 2
    by_fold = {row["fold"]: row for row in summary}
    assert set(by_fold) == {"mean", "std"}
    for metric in ("rmse", "mae", "r", "r2", "nrmse"):
        assert metric in by_fold["mean"] and metric in by_fold["std"]
        assert np.isfinite(by_fold["mean"][metric])
    # The mean must equal the mean of the four fold values.
    expected = np.mean([r["r2"] for r in records])
    assert by_fold["mean"]["r2"] == pytest.approx(expected)
    expected_std = np.std([r["r2"] for r in records], ddof=1)
    assert by_fold["std"]["r2"] == pytest.approx(expected_std)


def test_log_metrics_writes_both_files_with_the_required_columns(tmp_path):
    records, per_step = _four_folds()
    m_path, s_path = tmp_path / "metrics.csv", tmp_path / "per_step.csv"
    metrics, steps = log_metrics(records, per_step, metrics_path=m_path,
                                 per_step_path=s_path)

    assert list(metrics.columns) == METRIC_COLUMNS
    assert list(steps.columns) == PER_STEP_COLUMNS
    assert m_path.exists() and s_path.exists()

    # 4 folds + mean + std
    assert len(metrics) == 6
    assert set(metrics["fold"].astype(str)) == {"0", "1", "2", "3", "mean", "std"}
    # per-step: 3 lead times x (4 folds + mean + std)
    assert len(steps) == 18
    assert set(steps["step"]) == {1, 2, 3}


def test_log_metrics_replaces_rather_than_duplicates_on_rerun(tmp_path):
    records, per_step = _four_folds()
    m_path, s_path = tmp_path / "metrics.csv", tmp_path / "per_step.csv"
    first, _ = log_metrics(records, per_step, metrics_path=m_path,
                           per_step_path=s_path)
    second, _ = log_metrics(records, per_step, metrics_path=m_path,
                            per_step_path=s_path)
    assert len(second) == len(first), "re-running must refresh, not duplicate"


def test_log_metrics_accumulates_across_models(tmp_path):
    m_path = tmp_path / "metrics.csv"
    y = _y(300).reshape(100, 3)
    for model in ("persistence", "xgb"):
        rows = [fold_record(model, "lags_only", 3, k, "test", y,
                            y + RNG.normal(0, 0.3, y.shape)) for k in range(4)]
        frame, _ = log_metrics(rows, metrics_path=m_path)
    assert set(frame["model"]) == {"persistence", "xgb"}
    assert len(frame) == 12          # 2 models x (4 folds + mean + std)


# --------------------------------------------------------------------------
# leaderboard
# --------------------------------------------------------------------------
def test_leaderboard_pivots_models_against_horizons(tmp_path):
    m_path = tmp_path / "metrics.csv"
    frame = None
    for model in ("persistence", "xgb"):
        for horizon in (1, 6):
            y = _y(100 * horizon).reshape(100, horizon)
            rows = [fold_record(model, "lags_only", horizon, k, "test", y,
                                y + RNG.normal(0, 0.3, y.shape)) for k in range(4)]
            frame, _ = log_metrics(rows, metrics_path=m_path)

    boards = leaderboard_rolling(frame)
    assert set(boards) == {"rmse", "nrmse", "r", "r2"}
    assert list(boards["rmse"].columns) == [10, 60]
    assert len(boards["rmse"]) == 2
    # rmse cells carry "mean +/- std" with decimal commas.
    cell = boards["rmse"].iloc[0, 0]
    assert "+/-" in cell and "," in cell
    # the other boards carry plain means
    assert np.isfinite(boards["r2"].iloc[0, 0])


def test_leaderboard_needs_mean_rows():
    y = _y(100).reshape(100, 1)
    rows = [fold_record("xgb", "lags_only", 1, 0, "test", y, y)]
    with pytest.raises(ValueError):
        leaderboard_rolling(pd.DataFrame(rows))
