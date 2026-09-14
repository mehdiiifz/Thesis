"""Metric computation and CSV logging for the forecasting experiments.

Schemas (CLAUDE.md):
- results/metrics.csv:          model, features, horizon, fold, split,
                                rmse, mae, r, nrmse, run_time
- results/metrics_per_step.csv: same + step (lead time 1..H, rows per step)

fold is 0..3 for per-fold rows, plus "mean" and "std" summary rows added by
the log_*_summaries wrappers (via rolling_eval.summarize_folds).

Run from project root: python src/metrics_utils.py  (prints leaderboard if
results/metrics.csv exists, else a demo on synthetic data)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rolling_eval import summarize_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "results" / "metrics.csv"
METRICS_PER_STEP_CSV = ROOT / "results" / "metrics_per_step.csv"

METRIC_KEYS = ["rmse", "mae", "r", "nrmse"]
COLUMNS = ["model", "features", "horizon", "fold", "split",
           "rmse", "mae", "r", "nrmse", "run_time"]
COLUMNS_PER_STEP = ["model", "features", "horizon", "fold", "split", "step",
                    "rmse", "mae", "r", "nrmse", "run_time"]


def scores(y_true, y_pred) -> dict:
    """rmse, mae, r (Pearson), nrmse = rmse / mean(y_true).

    Accepts 1-D or 2-D arrays; 2-D inputs are scored over all elements,
    i.e. RMSE is POOLED across samples and lead-time steps jointly
    (= sqrt(mean(per-step RMSE^2))), per the CLAUDE.md hard rule.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    assert yt.shape == yp.shape, (yt.shape, yp.shape)
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    degenerate = yt.std() == 0 or yp.std() == 0
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "r": np.nan if degenerate else float(np.corrcoef(yt, yp)[0, 1]),
        "nrmse": rmse / float(yt.mean()) if yt.mean() != 0 else np.nan,
    }


def per_step_rmse(y_true, y_pred) -> np.ndarray:
    """RMSE per lead time: column j of (n, H) arrays -> element j (length H).

    1-D inputs are treated as a single-step (n, 1) forecast.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        yt, yp = yt[:, None], yp[:, None]
    assert yt.shape == yp.shape, (yt.shape, yp.shape)
    return np.sqrt(np.mean((yp - yt) ** 2, axis=0))


def _append(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows to a CSV, writing the header only if the file is new."""
    path.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def log_metrics(model: str, features: str, horizon: int, fold, split: str,
                sc: dict, run_time: float, path: Path = None) -> None:
    """Append one row (per-fold or summary) to results/metrics.csv."""
    row = {"model": model, "features": features, "horizon": horizon,
           "fold": fold, "split": split, "run_time": run_time}
    row.update({k: sc[k] for k in METRIC_KEYS})
    _append(path or METRICS_CSV, [row], COLUMNS)


def log_metrics_per_step(model: str, features: str, horizon: int, fold,
                         split: str, y_true, y_pred, run_time: float,
                         path: Path = None) -> list[dict]:
    """Append one row per lead time to results/metrics_per_step.csv.

    Returns the per-step score dicts (step order 1..H) so callers can
    collect them across folds for log_per_step_summaries.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        yt, yp = yt[:, None], yp[:, None]
    step_scores, rows = [], []
    for j in range(yt.shape[1]):
        sc = scores(yt[:, j], yp[:, j])
        step_scores.append(sc)
        row = {"model": model, "features": features, "horizon": horizon,
               "fold": fold, "split": split, "step": j + 1,
               "run_time": run_time}
        row.update(sc)
        rows.append(row)
    _append(path or METRICS_PER_STEP_CSV, rows, COLUMNS_PER_STEP)
    return step_scores


def log_fold_summaries(model: str, features: str, horizon: int, split: str,
                       fold_scores: list[dict], run_times: list[float],
                       path: Path = None) -> dict:
    """Add fold="mean" and fold="std" rows to metrics.csv via summarize_folds."""
    s = summarize_folds(fold_scores)
    rt = np.asarray(run_times, dtype=float)
    mean_sc = {k: s[f"{k}_mean"] for k in METRIC_KEYS}
    std_sc = {k: s[f"{k}_std"] for k in METRIC_KEYS}
    log_metrics(model, features, horizon, "mean", split, mean_sc,
                float(rt.mean()), path)
    log_metrics(model, features, horizon, "std", split, std_sc,
                float(rt.std(ddof=1)) if len(rt) > 1 else 0.0, path)
    return s


def log_per_step_summaries(model: str, features: str, horizon: int,
                           split: str, per_fold_steps: list[list[dict]],
                           run_times: list[float],
                           path: Path = None) -> None:
    """Add fold="mean"/"std" rows per step to metrics_per_step.csv.

    per_fold_steps: one list of per-step score dicts per fold, as returned
    by log_metrics_per_step.
    """
    rt = np.asarray(run_times, dtype=float)
    rt_mean = float(rt.mean())
    rt_std = float(rt.std(ddof=1)) if len(rt) > 1 else 0.0
    n_steps = len(per_fold_steps[0])
    mean_rows, std_rows = [], []
    for j in range(n_steps):
        s = summarize_folds([fold[j] for fold in per_fold_steps])
        base = {"model": model, "features": features, "horizon": horizon,
                "split": split, "step": j + 1}
        mean_rows.append({**base, "fold": "mean", "run_time": rt_mean,
                          **{k: s[f"{k}_mean"] for k in METRIC_KEYS}})
        std_rows.append({**base, "fold": "std", "run_time": rt_std,
                         **{k: s[f"{k}_std"] for k in METRIC_KEYS}})
    _append(path or METRICS_PER_STEP_CSV, mean_rows + std_rows,
            COLUMNS_PER_STEP)


def leaderboard_rolling(path: Path = None) -> pd.DataFrame:
    """Model x horizon leaderboard from rolling-origin test summary rows.

    Keeps fold=="mean" test rows (with fold=="std" for the +/-) and pivots
    to (model, features) x horizon with columns rmse ("mean +/- std"),
    nrmse and r (means across folds). All rmse/nrmse values are POOLED
    over samples and lead-time steps jointly (see scores()).
    """
    df = pd.read_csv(path or METRICS_CSV)
    df = df[df["split"] == "test"]
    keys = ["model", "features", "horizon"]
    mean = df[df["fold"] == "mean"]
    std = df[df["fold"] == "std"]
    m = mean.merge(std[keys + ["rmse"]], on=keys, how="left",
                   suffixes=("", "_sd"))
    m["rmse_str"] = (m["rmse"].round(3).astype(str) + " +/- "
                     + m["rmse_sd"].fillna(0).round(3).astype(str))
    out = m.pivot(index=["model", "features"], columns="horizon",
                  values=["rmse_str", "nrmse", "r"])
    out = out.rename(columns={"rmse_str": "rmse"}, level=0)
    out.loc[:, ("nrmse", slice(None))] = out.loc[:, ("nrmse", slice(None))].round(3)
    out.loc[:, ("r", slice(None))] = out.loc[:, ("r", slice(None))].round(3)
    # horizon as the outer column level, metrics inner
    out = out.swaplevel(axis=1).sort_index(axis=1, level=0)
    return out


if __name__ == "__main__":
    if METRICS_CSV.exists():
        print(leaderboard_rolling().to_string())
    else:
        rng = np.random.default_rng(42)
        yt = rng.uniform(2, 12, size=(500, 6))
        yp = yt + rng.normal(0, 1, size=yt.shape)
        print("demo scores:", {k: round(v, 3) for k, v in scores(yt, yp).items()})
        print("demo per-step rmse:", per_step_rmse(yt, yp).round(3))
        print(f"(no {METRICS_CSV.relative_to(ROOT)} yet - run training first)")
