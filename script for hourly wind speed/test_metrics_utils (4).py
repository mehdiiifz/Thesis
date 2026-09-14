"""Tests for metric computation: nrmse, per-step RMSE, logging schema."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics_utils import (COLUMNS, COLUMNS_PER_STEP, leaderboard_rolling,  # noqa: E402
                           log_fold_summaries, log_metrics,
                           log_metrics_per_step, per_step_rmse, scores)


def test_nrmse_known_values():
    # constant error of 1 on a series with mean 3 -> rmse 1, nrmse 1/3
    y_true = np.array([3.0, 3.0, 3.0, 3.0])
    y_pred = np.array([4.0, 4.0, 4.0, 4.0])
    sc = scores(y_true, y_pred)
    assert sc["rmse"] == 1.0
    assert sc["mae"] == 1.0
    assert np.isclose(sc["nrmse"], 1.0 / 3.0)
    # perfect forecast -> rmse = nrmse = 0
    sc0 = scores(y_true, y_true)
    assert sc0["rmse"] == 0.0 and sc0["nrmse"] == 0.0


def test_per_step_rmse_known_values():
    n = 50
    rng = np.random.default_rng(42)
    y_true = rng.uniform(2, 12, size=(n, 3))
    # constant bias per lead time -> per-step RMSE = |bias|
    y_pred = y_true + np.array([1.0, 2.0, -3.0])
    out = per_step_rmse(y_true, y_pred)
    assert out.shape == (3,)
    assert np.allclose(out, [1.0, 2.0, 3.0])
    # 1-D input behaves as single-step
    out1 = per_step_rmse(y_true[:, 0], y_true[:, 0] + 2.0)
    assert out1.shape == (1,) and np.isclose(out1[0], 2.0)
    # overall rmse aggregates the per-step values: rmse^2 = mean(step_rmse^2)
    sc = scores(y_true, y_pred)
    assert np.isclose(sc["rmse"] ** 2, np.mean(out ** 2))


def test_logging_schema_and_leaderboard(tmp_path):
    path = tmp_path / "metrics.csv"
    path_step = tmp_path / "metrics_per_step.csv"
    rng = np.random.default_rng(42)
    for model in ["persistence", "extratrees"]:
        for horizon in [1, 6]:
            fold_scores, run_times = [], []
            for fold in range(4):
                yt = rng.uniform(2, 12, size=(100, horizon))
                yp = yt + rng.normal(0, 1, size=yt.shape)
                sc = scores(yt, yp)
                log_metrics(model, "lags_only", horizon, fold, "test",
                            sc, 0.5, path)
                log_metrics_per_step(model, "lags_only", horizon, fold,
                                     "test", yt, yp, 0.5, path_step)
                fold_scores.append(sc)
                run_times.append(0.5)
            log_fold_summaries(model, "lags_only", horizon, "test",
                               fold_scores, run_times, path)

    df = pd.read_csv(path)
    assert list(df.columns) == COLUMNS
    # 4 per-fold + mean + std rows per (model, horizon)
    assert len(df) == 2 * 2 * 6
    assert set(df["fold"].unique()) == {"0", "1", "2", "3", "mean", "std"}

    df_step = pd.read_csv(path_step)
    assert list(df_step.columns) == COLUMNS_PER_STEP
    assert len(df_step) == 2 * 4 * (1 + 6)  # models x folds x steps

    lb = leaderboard_rolling(path)
    assert lb.shape == (2, 6)  # 2 models x (2 horizons * 3 metrics)
    cell = lb.loc[("extratrees", "lags_only"), (6, "rmse")]
    assert "+/-" in cell
