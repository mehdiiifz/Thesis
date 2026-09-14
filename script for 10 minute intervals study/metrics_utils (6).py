"""Error metrics and result logging for the rolling-origin evaluation.

All metrics are POOLED over all test samples and all lead-time steps jointly,
as required by CLAUDE.md: for a direct multi-output model with horizon H, the
(n, H) matrices of truth and prediction are flattened before the metric is
computed, so every lead time contributes equally to one number per fold.

Metrics
    rmse   root mean square error, m/s
    mae    mean absolute error, m/s
    r      Pearson correlation, dimensionless
    r2     coefficient of determination, dimensionless, MAY BE NEGATIVE
    nrmse  rmse / mean(observed test values), dimensionless

r and r2 are NOT interchangeable and both are reported. See scores().

Result files
    results/metrics.csv            one row per model/feature set/horizon/fold,
                                   plus fold="mean" and fold="std" rows
    results/metrics_per_step.csv   the same, broken out per lead-time step

Run from the project root to regenerate the evaluation section:

    python src/metrics_utils.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rolling_eval import summarize_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS_CSV = RESULTS / "metrics.csv"
PER_STEP_CSV = RESULTS / "metrics_per_step.csv"
FOLDS_CSV = RESULTS / "folds.csv"

STEP_MINUTES = 10

METRIC_NAMES = ("rmse", "mae", "r", "r2", "nrmse")

METRIC_COLUMNS = ["model", "features", "horizon_steps", "horizon_min", "fold",
                  "split", "rmse", "mae", "r", "r2", "nrmse", "run_time"]
PER_STEP_COLUMNS = ["model", "features", "horizon_steps", "horizon_min", "fold",
                    "split", "step", "rmse", "mae", "r", "r2", "nrmse", "run_time"]

# Rows are identified by these keys, so re-running a training script replaces
# its own rows instead of appending duplicates.
METRIC_KEYS = ["model", "features", "horizon_steps", "fold", "split"]
PER_STEP_KEYS = METRIC_KEYS + ["step"]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _flatten(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Validate and flatten truth/prediction to pooled 1-D arrays."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    yt, yp = yt.ravel(), yp.ravel()
    if yt.size == 0:
        raise ValueError("empty arrays -- nothing to score")
    if not np.isfinite(yt).all():
        raise ValueError("y_true contains non-finite values")
    if not np.isfinite(yp).all():
        raise ValueError("y_pred contains non-finite values")
    return yt, yp


def scores(y_true, y_pred) -> dict[str, float]:
    """Pooled error metrics for one fold.

    y_true, y_pred may be 1-D (n,) or 2-D (n, H). A 2-D pair is flattened
    first, so the result pools all samples and all lead-time steps jointly.

    Returns {"rmse", "mae", "r", "r2", "nrmse"}.

    On r versus r2 -- these are NOT interchangeable and both are reported:

        r   Pearson correlation. Measures only CO-MOVEMENT: whether prediction
            and truth rise and fall together. r is invariant to any affine
            rescaling of the prediction, so a forecast of half the true wind
            speed still scores r = 1,0 despite being wrong everywhere.

        r2  Coefficient of determination,
                r2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
            Measures EXPLAINED VARIANCE against the test-period mean as the
            reference, and therefore penalises bias and scale errors that r
            ignores. r2 MAY BE NEGATIVE: a negative value means the forecast
            has larger squared error than simply predicting the mean of the
            observed test values. It is not the square of r here, because the
            prediction is not fitted by least squares to this sample.

    r is nan when either series is constant (correlation undefined); r2 is nan
    when y_true is constant (zero total sum of squares).
    """
    yt, yp = _flatten(y_true, y_pred)

    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    # Detect a constant series exactly. Comparing std() to 0 is not reliable:
    # an array of identical values can still report a standard deviation of
    # order 1e-16, which would let a degenerate case through as a real number.
    constant_true = yt.min() == yt.max()
    constant_pred = yp.min() == yp.max()

    sse = float(np.sum(err ** 2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float("nan") if constant_true else 1.0 - sse / sst

    if constant_true or constant_pred:
        r = float("nan")
    else:
        r = float(np.corrcoef(yt, yp)[0, 1])

    mean_obs = float(yt.mean())
    nrmse = float("nan") if mean_obs == 0 else rmse / mean_obs

    return {"rmse": rmse, "mae": mae, "r": r, "r2": r2, "nrmse": nrmse}


def _as_2d(y) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def per_step_scores(y_true, y_pred) -> pd.DataFrame:
    """All metrics for each lead-time step of a direct multi-output forecast.

    Column j of the (n, H) matrices is lead time j+1, i.e. t+(j+1) steps.
    Returns one row per step with step, step_min and every metric.
    """
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    rows = []
    for j in range(yt.shape[1]):
        row = {"step": j + 1, "step_min": (j + 1) * STEP_MINUTES}
        row.update(scores(yt[:, j], yp[:, j]))
        rows.append(row)
    return pd.DataFrame(rows)


def per_step_rmse(y_true, y_pred) -> np.ndarray:
    """RMSE at each lead-time step -- the lead-time error curve."""
    return per_step_scores(y_true, y_pred)["rmse"].to_numpy()


def per_step_r2(y_true, y_pred) -> np.ndarray:
    """r2 at each lead-time step -- the lead-time skill curve.

    Each step is scored against the mean of that step's own observations, so
    values may be negative where the forecast is worse than that mean.
    """
    return per_step_scores(y_true, y_pred)["r2"].to_numpy()


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def fold_record(model: str, features: str, horizon_steps: int, fold,
                split: str, y_true, y_pred, run_time: float = float("nan"),
                ) -> dict:
    """One pooled metrics row for a single fold and split."""
    record = {
        "model": model,
        "features": features,
        "horizon_steps": int(horizon_steps),
        "horizon_min": int(horizon_steps) * STEP_MINUTES,
        "fold": fold,
        "split": split,
        "run_time": float(run_time),
    }
    record.update(scores(y_true, y_pred))
    return record


def per_step_records(model: str, features: str, horizon_steps: int, fold,
                     split: str, y_true, y_pred,
                     run_time: float = float("nan")) -> list[dict]:
    """One metrics row per lead-time step for a single fold and split."""
    table = per_step_scores(y_true, y_pred)
    rows = []
    for _, row in table.iterrows():
        rows.append({
            "model": model,
            "features": features,
            "horizon_steps": int(horizon_steps),
            "horizon_min": int(horizon_steps) * STEP_MINUTES,
            "fold": fold,
            "split": split,
            "step": int(row["step"]),
            "run_time": float(run_time),
            **{m: float(row[m]) for m in METRIC_NAMES},
        })
    return rows


def summarize_records(records: list[dict]) -> list[dict]:
    """Add fold="mean" and fold="std" rows for every metric, including r2.

    Records are grouped by model, features, horizon_steps, split (and step, if
    present). Numeric fold labels are summarised via summarize_folds() from
    rolling_eval; any existing "mean"/"std" rows are ignored so the function is
    safe to apply to an already-summarised list.
    """
    if not records:
        return []

    frame = pd.DataFrame(records)
    numeric_folds = frame[~frame["fold"].astype(str).isin(("mean", "std"))]
    if numeric_folds.empty:
        return []

    group_keys = ["model", "features", "horizon_steps", "horizon_min", "split"]
    if "step" in frame.columns:
        group_keys.append("step")

    value_keys = [c for c in (*METRIC_NAMES, "run_time") if c in frame.columns]

    out: list[dict] = []
    for key_values, group in numeric_folds.groupby(group_keys, dropna=False):
        fold_scores = [{k: float(row[k]) for k in value_keys}
                       for _, row in group.iterrows()]
        summary = summarize_folds(fold_scores)
        for stat in ("mean", "std"):
            row = dict(zip(group_keys, key_values))
            row["fold"] = stat
            for k in value_keys:
                row[k] = summary[f"{k}_{stat}"]
            out.append(row)
    return out


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
def _write(records: list[dict], path: Path, columns: list[str],
           keys: list[str], append: bool) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    missing = [c for c in columns if c not in frame.columns]
    for col in missing:
        frame[col] = np.nan
    frame = frame[columns]

    if append and path.exists():
        previous = pd.read_csv(path)
        for col in columns:
            if col not in previous.columns:
                previous[col] = np.nan
        frame = pd.concat([previous[columns], frame], ignore_index=True)

    # A round trip through CSV turns the fold label 0 into "0", which would
    # otherwise never match the int 0 of a fresh record and so append a
    # duplicate on every re-run. Normalise to text before de-duplicating.
    frame["fold"] = frame["fold"].astype(str)

    # Later rows win, so re-running a script refreshes its own rows.
    frame = frame.drop_duplicates(subset=keys, keep="last")
    frame = frame.sort_values(
        ["model", "features", "horizon_steps", "split", "fold"]
        + (["step"] if "step" in columns else []),
        kind="stable").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def log_metrics(records: list[dict], per_step: list[dict] | None = None, *,
                summarize: bool = True, append: bool = True,
                metrics_path: Path | str = METRICS_CSV,
                per_step_path: Path | str = PER_STEP_CSV,
                ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Write results/metrics.csv and results/metrics_per_step.csv.

    records   pooled rows, e.g. from fold_record()
    per_step  optional per-lead-time rows, e.g. from per_step_records()

    With summarize=True (the default) fold="mean" and fold="std" rows are
    appended for every metric before writing. With append=True existing rows
    are kept, so several training scripts can accumulate into one file; rows
    matching on (model, features, horizon_steps, fold, split[, step]) are
    replaced rather than duplicated.

    Returns the two written frames (the second is None if per_step is omitted).
    """
    records = list(records)
    if summarize:
        records = records + summarize_records(records)
    metrics = _write(records, Path(metrics_path), METRIC_COLUMNS,
                     METRIC_KEYS, append)

    per_step_frame = None
    if per_step is not None:
        rows = list(per_step)
        if summarize:
            rows = rows + summarize_records(rows)
        per_step_frame = _write(rows, Path(per_step_path), PER_STEP_COLUMNS,
                                PER_STEP_KEYS, append)
    return metrics, per_step_frame


# --------------------------------------------------------------------------
# leaderboard
# --------------------------------------------------------------------------
def leaderboard_rolling(metrics: pd.DataFrame | None = None,
                        split: str = "test",
                        metrics_path: Path | str = METRICS_CSV,
                        ) -> dict[str, pd.DataFrame]:
    """Model x horizon leaderboard from the fold="mean" / fold="std" rows.

    Returns four pivots, all indexed by (model, features) with one column per
    horizon_min:

        "rmse"   formatted "rmse_mean +/- rmse_std"
        "nrmse"  nrmse_mean
        "r"      r_mean
        "r2"     r2_mean

    Only rows for `split` (default the test split) are used.
    """
    from report_utils import fmt

    if metrics is None:
        path = Path(metrics_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- train models before building a leaderboard")
        metrics = pd.read_csv(path)

    rows = metrics[metrics["split"] == split].copy()
    rows["fold"] = rows["fold"].astype(str)
    means = rows[rows["fold"] == "mean"]
    stds = rows[rows["fold"] == "std"]
    if means.empty:
        raise ValueError(f"no fold=='mean' rows for split {split!r}")

    index = ["model", "features"]
    merged = means.merge(
        stds[index + ["horizon_min", *METRIC_NAMES]],
        on=index + ["horizon_min"], how="left", suffixes=("_mean", "_std"))

    out: dict[str, pd.DataFrame] = {}
    merged["rmse_label"] = [
        f"{fmt(m)} +/- {fmt(s)}" if pd.notna(s) else fmt(m)
        for m, s in zip(merged["rmse_mean"], merged["rmse_std"])
    ]
    out["rmse"] = merged.pivot_table(index=index, columns="horizon_min",
                                     values="rmse_label", aggfunc="first")
    for metric in ("nrmse", "r", "r2"):
        out[metric] = merged.pivot_table(index=index, columns="horizon_min",
                                         values=f"{metric}_mean", aggfunc="first")
    for frame in out.values():
        frame.columns.name = "horizon_min"
    return out


# --------------------------------------------------------------------------
# fold boundaries + report section
# --------------------------------------------------------------------------
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}


def _season_span(start: pd.Timestamp, end: pd.Timestamp) -> str:
    months = pd.period_range(start=start, end=end, freq="M")
    seen: list[str] = []
    for period in months:
        name = SEASONS[period.month]
        if name not in seen:
            seen.append(name)
    return " / ".join(seen)


def fold_boundaries(horizons: list[int] | None = None) -> pd.DataFrame:
    """Actual rolling-origin fold boundaries on the real data set.

    Computed by running the fold generator over the built design matrices, so
    the report never states a boundary that the code does not produce.
    """
    from load_data import load_10min
    from make_dataset import build_supervised
    from rolling_eval import rolling_origin_folds
    from report_utils import HORIZON_STEPS

    horizons = horizons or HORIZON_STEPS
    df = load_10min()

    rows = []
    for horizon in horizons:
        X, Y, _ = build_supervised(df, horizon_steps=horizon, features="lags_only")
        for fold in rolling_origin_folds(X, Y, horizon=horizon):
            train_X, val_X, test_X = fold["train"][0], fold["val"][0], fold["test"][0]
            t0, t1 = fold["test_period"]
            rows.append({
                "horizon_steps": horizon,
                "horizon_min": horizon * STEP_MINUTES,
                "fold": fold["fold"],
                "n_train": len(train_X),
                "n_val": len(val_X),
                "n_test": len(test_X),
                "train_start": train_X.index.min(),
                "train_end": train_X.index.max(),
                "val_start": val_X.index.min(),
                "val_end": val_X.index.max(),
                "test_start": t0,
                "test_end": t1,
                "gap_rows": horizon,
                "seasons": _season_span(t0, t1),
            })
    return pd.DataFrame(rows)


def write_evaluation_section() -> None:
    """Write report/03_methodology_evaluation.md from results/folds.csv."""
    from report_utils import (HORIZON_MIN, REPORT, fmt, fmt_int, log_progress,
                              md_table, write_section)

    RESULTS.mkdir(parents=True, exist_ok=True)
    folds = fold_boundaries()
    folds.to_csv(FOLDS_CSV, index=False)
    print(f"wrote {FOLDS_CSV.name}")

    # Re-read from disk: the section must be generated from the result file.
    folds = pd.read_csv(FOLDS_CSV, parse_dates=["train_start", "train_end",
                                                "val_start", "val_end",
                                                "test_start", "test_end"])

    n_folds = int(folds["fold"].nunique())
    reference_h = 6                      # 60 minutes, mid-range
    ref = folds[folds["horizon_steps"] == reference_h].sort_values("fold")

    # How far the block edges move across the eight horizons, in rows.
    spread = folds.groupby("fold")["test_start"].agg(lambda s: s.max() - s.min())
    max_shift_min = int(spread.max().total_seconds() // 60)

    tbl_folds = pd.DataFrame({
        "Fold": ref["fold"].astype(int) + 1,
        "Train rows": ref["n_train"].astype(int),
        "Val. rows": ref["n_val"].astype(int),
        "Test rows": ref["n_test"].astype(int),
        "Test period start": ref["test_start"].dt.strftime("%d.%m.%Y %H:%M"),
        "Test period end": ref["test_end"].dt.strftime("%d.%m.%Y %H:%M"),
        "Season(s)": ref["seasons"],
    })

    metric_defs = pd.DataFrame({
        "Metric": ["rmse", "mae", "r", "r2", "nrmse"],
        "Formula": [
            "sqrt( mean( (y - yhat)^2 ) )",
            # md_table escapes the pipes itself -- do not pre-escape here.
            "mean( |y - yhat| )",
            "cov(y, yhat) / (sd(y) sd(yhat))",
            "1 - sum((y - yhat)^2) / sum((y - mean(y))^2)",
            "rmse / mean(y)",
        ],
        "Unit": ["m/s", "m/s", "-", "-", "-"],
        "Range": ["[0, inf)", "[0, inf)", "[-1, 1]", "(-inf, 1]", "[0, inf)"],
        "Interpretation": [
            "Typical error magnitude, quadratic, so large errors dominate. "
            "The headline accuracy figure.",
            "Typical error magnitude, linear, less sensitive to single large "
            "errors than rmse. Reported alongside rmse to show whether an "
            "error gap comes from many small errors or a few large ones.",
            "Co-movement only. Invariant to affine rescaling of the forecast, "
            "so it does NOT detect bias or a wrong amplitude.",
            "Share of test-period variance explained, relative to predicting "
            "the test-period mean. MAY BE NEGATIVE, meaning worse than that "
            "mean. Penalises exactly the bias and scale errors that r ignores.",
            "rmse normalised by the mean observed wind speed, so results are "
            "comparable across sites with different wind regimes.",
        ],
    })

    why_rolling = (
        "## 3.1 Why not a single chronological split\n\n"
        "The record covers one year. A single chronological split — the usual "
        "70/15/15 — would place the test window at the end of that year, so "
        "every reported number would describe a single season. With a "
        "record starting on 10.01.2012 the test block would fall in "
        "November to January: the model would be selected and judged on winter "
        "alone, and the result would silently confound model quality with the "
        "particular wind regime of those weeks. Section 1 showed the diurnal "
        "profile changes shape markedly between seasons, so this is not a "
        "hypothetical concern. A single split also yields exactly one number "
        "per configuration, with no way to tell a real difference between two "
        "models from ordinary sampling variation.\n\n"
        "Rolling-origin evaluation with an expanding window addresses both "
        f"problems. The series is cut into {n_folds} consecutive test blocks "
        "placed after a minimum training period. For each block the model is "
        "retrained on all data preceding it and evaluated on the block itself, "
        "so the blocks fall in different parts of the year and every "
        "configuration is scored several times. Results are reported as mean "
        "+/- standard deviation across folds, which makes the spread visible "
        "instead of hiding it. This is the standard protocol for time-series "
        "model comparison (Tashman 2000; Bergmeir and Benitez 2012).\n\n"
        "Two properties are preserved throughout. Training data always precede "
        "the test block in time, so no model ever sees the future. And the "
        "window only ever expands, so later folds are trained on more data — "
        "which mirrors how the system would actually be operated, retraining "
        "as the record grows."
    )

    gap_prose = (
        "## 3.2 The horizon gap\n\n"
        "Ordering the data is not by itself sufficient. Under the direct "
        "multi-output formulation, a training sample issued at time t carries "
        "targets reaching to t+H. If training ran right up to the first test "
        "issue time, the last training samples would carry targets lying "
        "*inside* the test block, and the model would have been fitted on the "
        "very values it is about to be scored on. This is boundary leakage, "
        "and it inflates results at exactly the long horizons where the study "
        "is most interested in the answer.\n\n"
        "`src/rolling_eval.py` therefore leaves a gap of `horizon` rows between "
        "the end of the training window and the start of the test block. The "
        "gap scales with the horizon — 1 row at 10 minutes, 48 rows at 480 "
        "minutes — so that the last training target always falls strictly "
        "before the first test issue time, whatever the horizon. "
        "`tests/test_rolling_eval.py` asserts this property directly rather "
        "than trusting the arithmetic.\n\n"
        "Within each training window the final 15 % is held out as a "
        "validation set, used once on fold 0 to select hyperparameters. Those "
        "hyperparameters are then reused unchanged for all folds, so the "
        "later folds' test blocks play no part in model selection."
    )

    fold_prose = (
        f"Table 3-1 lists the {n_folds} folds as the code actually produces "
        f"them, for a horizon of {reference_h * STEP_MINUTES} minutes. Because "
        "the number of usable issue times depends slightly on the horizon — "
        "deeper lags and longer targets each cost rows — the block edges shift "
        f"a little across the eight horizons, by at most {fmt_int(max_shift_min)} "
        "minutes. The seasonal coverage is identical in all cases.\n\n"
        "One limitation follows directly from having a single year and an "
        "expanding window: the earliest 40 % of the record is reserved as the "
        "minimum training window and is therefore never tested. Spring is "
        "consequently not represented among the test blocks. This is a property "
        "of the data, not of the protocol, and it disappears as soon as a "
        "second year becomes available; it should be stated as a limitation "
        "when the results are read."
    )

    metrics_prose = (
        "## 3.3 Error metrics\n\n"
        "All five metrics are pooled over all test samples and all lead-time "
        "steps jointly: for a horizon of H steps the (n, H) matrices of "
        "observation and forecast are flattened before the metric is computed, "
        "so each lead time contributes equally to the fold's score. Per-step "
        "values are logged separately to `results/metrics_per_step.csv`, which "
        "is what the lead-time error curves are drawn from.\n\n"
        "**r and r2 are not interchangeable and both are reported.** Pearson's "
        "r measures co-movement only and is invariant to any affine rescaling "
        "of the forecast: a prediction of exactly half the true wind speed "
        "still scores r = 1,0 while being wrong at every single point. r2 "
        "measures explained variance against the test-period mean and so "
        "penalises precisely the bias and scale errors that r cannot see. For "
        "that half-scale forecast r2 is strongly negative. Reporting r alone "
        "would therefore be misleading, and r2 is not the square of r here, "
        "because the forecasts are not least-squares fits to the test sample.\n\n"
        "**r2 may be negative.** A negative r2 means the forecast has a larger "
        "squared error than simply predicting the mean of the observed test "
        "values. This is a real possibility at the longer horizons, where "
        "Section 1 showed the autocorrelation has decayed substantially, and "
        "it is reported as it falls rather than being clipped at zero."
    )

    write_section(REPORT / "03_methodology_evaluation.md",
                  "3 Evaluation Protocol and Error Metrics", [
        "This section states how models are evaluated and how their errors are "
        "measured. The fold boundaries in Table 3-1 are read from "
        "`results/folds.csv`, which is produced by running the fold generator "
        "over the actual design matrices.",
        why_rolling,
        gap_prose,
        fold_prose,
        md_table(tbl_folds, f"Rolling-origin folds at a horizon of "
                            f"{reference_h * STEP_MINUTES} minutes, with an "
                            f"expanding training window and a gap of "
                            f"{reference_h} rows before each test block", "3-1"),
        metrics_prose,
        md_table(metric_defs, "Error metric definitions and interpretation", "3-2"),
        "Every metric is logged per fold to `results/metrics.csv`, together "
        "with fold=\"mean\" and fold=\"std\" rows computed across the "
        f"{n_folds} folds for all five metrics.",
    ])
    print("wrote report/03_methodology_evaluation.md")

    log_progress(
        3, f"Evaluation protocol fixed: {n_folds} expanding-window rolling-origin "
        f"folds with a gap of `horizon` rows before each test block; metrics "
        "rmse, mae, r, r2 and nrmse pooled over samples and lead times, with r2 "
        "reported as it falls including negative values. Test blocks cover "
        f"{', '.join(ref['seasons'].unique())}; spring is untested with one "
        "year of data.")
    print("appended to report/PROGRESS.md")


if __name__ == "__main__":
    write_evaluation_section()
