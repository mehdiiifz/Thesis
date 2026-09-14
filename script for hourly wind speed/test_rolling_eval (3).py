"""Tests for rolling-origin fold generator: no boundary leakage, correct order."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_dataset import build_supervised  # noqa: E402
from rolling_eval import rolling_origin_folds  # noqa: E402


def _toy(n=2000):
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return pd.DataFrame({"ws100": np.arange(n, dtype=float)}, index=idx)


def test_gap_prevents_boundary_leakage():
    H = 12
    X, Y, _ = build_supervised(_toy(), horizon=H, n_lags=8)
    for fold in rolling_origin_folds(X, Y, horizon=H):
        last_train_issue = fold["train"][0].index.max()
        first_test_issue = fold["test"][0].index.min()
        # last training target time = issue + H hours; must be < first test issue
        last_train_target = last_train_issue + pd.Timedelta(hours=H)
        assert last_train_target <= first_test_issue


def test_chronology_and_expansion():
    H = 6
    X, Y, _ = build_supervised(_toy(), horizon=H, n_lags=8)
    folds = list(rolling_origin_folds(X, Y, horizon=H))
    assert len(folds) == 4
    train_sizes = [len(f["train"][0]) for f in folds]
    assert train_sizes == sorted(train_sizes)          # expanding window
    for f in folds:
        assert f["val"][0].index.max() < f["test"][0].index.min()
    # test blocks are consecutive, non-overlapping
    for a, b in zip(folds, folds[1:]):
        assert a["test"][0].index.max() < b["test"][0].index.min()
