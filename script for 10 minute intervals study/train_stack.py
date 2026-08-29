"""Stacking ensemble over the tree models and the LSTM, plus the final comparison.

Run from the project root:

    python src/train_stack.py
    python src/train_stack.py --report-only    # rebuild sections 6 and 7

Protocol
--------
Stacking is done strictly inside each rolling-origin fold, so no information
crosses a fold boundary:

1. Base models are fitted on the fold's training window with the
   hyperparameters and feature sets chosen on fold-0 validation in Sections 4
   and 5 -- never re-tuned here, and never chosen on test results.
2. Out-of-fold base predictions covering the training window come from
   TimeSeriesSplit(5). A plain KFold would place future rows in the training
   part of an inner split and leak backwards; TimeSeriesSplit only ever trains
   on rows preceding the block it predicts.
3. The meta-learner is fitted on those out-of-fold predictions and SELECTED on
   the fold's validation block, which no base model was fitted on.
4. The selected meta-learner is applied to base predictions for the test block,
   produced by base models refitted on the whole training window.

Meta-learner (lesson from the hourly study)
-------------------------------------------
For horizons >= 12 steps the PER-STEP formulation is used: one Ridge per
lead-time step, seeing only the base models' predictions for that same step,
with alpha chosen from [0.1, 1, 10, 100] on the validation block. MLP
meta-models are BANNED at these horizons -- a dense meta-model over all
H x n_base inputs has enough freedom to fit the validation block's particular
shape and generalises poorly over long vectors.

For horizons <= 9 steps a small MLP is admitted as a candidate and compared
against the per-step Ridge on the validation block; the winner is kept.

Runtime
-------
The LSTM dominates. Inside the stacking loop it is trained with a cheaper
configuration (stride 4, at most 40 epochs, patience 6) than in Section 5,
applied IDENTICALLY to the out-of-fold, validation and test fits so the
meta-learner is calibrated on the same distribution it later sees. The
consequence is that the stack's LSTM parent is slightly weaker than the
Section 5 LSTM; both are reported, and the acceptance criterion is evaluated
against the stack's own base models so that like is compared with like.
"""
from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_data import load_10min  # noqa: E402
from make_dataset import build_supervised  # noqa: E402
from metrics_utils import (  # noqa: E402
    METRIC_NAMES, fold_record, log_metrics, per_step_records, scores,
)
from report_utils import (  # noqa: E402
    FIGURES, HORIZON_STEPS, REPORT, RESULTS, STEP_MINUTES, fmt, fmt_int,
    log_progress, md_table, write_section,
)
from rolling_eval import rolling_origin_folds  # noqa: E402
import train_lstm as L  # noqa: E402
import train_trees as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PREDICTIONS = RESULTS / "predictions"

SEED = 42
INNER_SPLITS = 5
PER_STEP_FROM = 12                 # horizons >= this use the per-step meta only
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
ACCEPTANCE_TOLERANCE = 0.01        # stack must be <= best base + 1 %

# Cheaper LSTM inside the stacking loop; see the module docstring.
STACK_LSTM_STRIDE = 4
STACK_LSTM_EPOCHS = 40
STACK_LSTM_PATIENCE = 6

TREE_MODELS = ("rf", "et", "xgb", "lgbm")
STACK_NAME = "stack"
STACK_FEATURES = "ensemble"

BASE_METRICS_CSV = RESULTS / "stack_base_metrics.csv"
META_CHOICE_CSV = RESULTS / "stack_meta_choice.csv"
ACCEPTANCE_CSV = RESULTS / "stack_acceptance.csv"
DIAGNOSTIC_CSV = RESULTS / "stack_oof_vs_test_distributions.csv"
TEX_TABLE = RESULTS / "final_results_table.tex"
RECOMMENDATION_MD = RESULTS / "operational_recommendation.md"
HOURLY_CSV = RESULTS / "hourly_reference.csv"

SECTION_6 = REPORT / "06_results_ensemble.md"
SECTION_7 = REPORT / "07_comparison_to_hourly.md"

FIG_FINAL_RMSE = "final_rmse_vs_leadtime.png"
FIG_FINAL_NRMSE = "final_nrmse_by_horizon.png"
FIG_R2 = "r2_vs_horizon.png"
FIG_FEATURES = "feature_set_comparison.png"
FIG_CASES = "case_study_windows.png"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
})


# --------------------------------------------------------------------------
# base-model configuration, taken from the fold-0 validation records
# --------------------------------------------------------------------------
def base_configuration() -> dict:
    """Feature set and hyperparameters per tree model and horizon.

    Read from results/tuning.csv, which holds fold-0 VALIDATION scores. The
    feature set is therefore chosen on validation data, never on the test
    results -- picking it from results/metrics.csv would be selection on the
    very blocks the models are about to be scored on.
    """
    tuning = pd.read_csv(RESULTS / "tuning.csv")
    best = (tuning.loc[tuning.groupby(["model", "features", "horizon_steps"])
                       ["val_rmse"].idxmin()])
    config = {}
    for model in TREE_MODELS:
        for horizon in HORIZON_STEPS:
            rows = best[(best["model"] == model)
                        & (best["horizon_steps"] == horizon)]
            if rows.empty:
                continue
            pick = rows.loc[rows["val_rmse"].idxmin()]
            config[(model, horizon)] = {
                "features": pick["features"],
                "params": ast.literal_eval(pick["params"]),
            }
    return config


def lstm_configuration() -> tuple[str, int]:
    """Winning LSTM variant and width, from the fold-0 selection records."""
    selection = pd.read_csv(RESULTS / "lstm_variant_selection.csv")
    variant = selection.groupby("variant")["val_rmse"].mean().idxmin()
    units = pd.read_csv(RESULTS / "lstm_units_selection.csv")
    sub = units[units["variant"] == variant]
    return variant, int(sub.groupby("units")["val_rmse"].mean().idxmin())


# --------------------------------------------------------------------------
# base model fitting
# --------------------------------------------------------------------------
def fit_predict_tree(model: str, params: dict, horizon: int,
                     Xtr: np.ndarray, Ytr: np.ndarray,
                     targets: list[np.ndarray]) -> list[np.ndarray]:
    spec = T.SPEC_BY_NAME[model]
    fitted, _ = T.fit_model(spec, params, horizon, Xtr, Ytr, eval_set=None)
    return [T.predict(spec, fitted, X, horizon) for X in targets]


def fit_predict_lstm(units: int, horizon: int, Xtr, Ytr, Xva, Yva,
                     targets: list[np.ndarray]) -> list[np.ndarray]:
    """Fit the stacking LSTM and predict each requested block."""
    stride, epochs, patience = L.TRAIN_STRIDE, L.MAX_EPOCHS, L.PATIENCE
    L.TRAIN_STRIDE, L.MAX_EPOCHS, L.PATIENCE = (
        STACK_LSTM_STRIDE, STACK_LSTM_EPOCHS, STACK_LSTM_PATIENCE)
    try:
        model, scaler, _, _ = L.fit_lstm(units, horizon, Xtr, Ytr, Xva, Yva)
        return [L.predict_lstm(model, scaler, X) for X in targets]
    finally:
        L.TRAIN_STRIDE, L.MAX_EPOCHS, L.PATIENCE = stride, epochs, patience


# --------------------------------------------------------------------------
# meta-learners
# --------------------------------------------------------------------------
def fit_per_step_ridge(oof: np.ndarray, y_oof: np.ndarray,
                       val: np.ndarray, y_val: np.ndarray):
    """One Ridge per lead-time step, alpha chosen on the validation block.

    oof/val have shape (n, H, n_base). Step j sees only oof[:, j, :], i.e. the
    base models' predictions for that same lead time -- not the whole vector.
    """
    horizon = oof.shape[1]
    models, alphas = [], []
    for step in range(horizon):
        best, best_rmse, best_alpha = None, np.inf, None
        for alpha in RIDGE_ALPHAS:
            ridge = Ridge(alpha=alpha, random_state=SEED)
            ridge.fit(oof[:, step, :], y_oof[:, step])
            rmse = float(np.sqrt(np.mean(
                (ridge.predict(val[:, step, :]) - y_val[:, step]) ** 2)))
            if rmse < best_rmse:
                best, best_rmse, best_alpha = ridge, rmse, alpha
        models.append(best)
        alphas.append(best_alpha)
    return models, alphas


def predict_per_step_ridge(models, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X[:, j, :]) for j, m in enumerate(models)])


def fit_mlp_meta(oof: np.ndarray, y_oof: np.ndarray):
    """Small dense meta-model over the flattened base predictions."""
    n = oof.shape[0]
    mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=400,
                       early_stopping=True, n_iter_no_change=10,
                       random_state=SEED)
    # A single-output target must be passed 1-D: sklearn ravels an (n, 1)
    # column vector anyway and only warns about it.
    target = y_oof.ravel() if y_oof.shape[1] == 1 else y_oof
    mlp.fit(oof.reshape(n, -1), target)
    return mlp


def predict_mlp_meta(mlp, X: np.ndarray) -> np.ndarray:
    out = np.asarray(mlp.predict(X.reshape(X.shape[0], -1)), dtype=np.float64)
    # Restore the (n, H) shape: with a single output sklearn returns (n,),
    # which would not align with the (n, 1) targets at horizon 1.
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def select_meta(horizon: int, oof: np.ndarray, y_oof: np.ndarray,
                val: np.ndarray, y_val: np.ndarray):
    """Fit candidate meta-learners and keep the one that wins on validation.

    At horizons >= PER_STEP_FROM steps only the per-step Ridge is admitted;
    the MLP is banned there, following the hourly study.
    """
    ridge_models, alphas = fit_per_step_ridge(oof, y_oof, val, y_val)
    ridge_rmse = scores(y_val, predict_per_step_ridge(ridge_models, val))["rmse"]

    if horizon >= PER_STEP_FROM:
        return ("ridge_per_step", ridge_models, ridge_rmse,
                {"alphas": alphas, "mlp_rmse": None, "ridge_rmse": ridge_rmse})

    mlp = fit_mlp_meta(oof, y_oof)
    mlp_rmse = scores(y_val, predict_mlp_meta(mlp, val))["rmse"]
    if mlp_rmse < ridge_rmse:
        return ("mlp", mlp, mlp_rmse,
                {"alphas": alphas, "mlp_rmse": mlp_rmse, "ridge_rmse": ridge_rmse})
    return ("ridge_per_step", ridge_models, ridge_rmse,
            {"alphas": alphas, "mlp_rmse": mlp_rmse, "ridge_rmse": ridge_rmse})


def apply_meta(kind: str, model, X: np.ndarray) -> np.ndarray:
    if kind == "mlp":
        return predict_mlp_meta(model, X)
    return predict_per_step_ridge(model, X)


# --------------------------------------------------------------------------
# main stacking run
# --------------------------------------------------------------------------
def run_stack():
    df = load_10min()
    channels = L.channel_frame(df)
    config = base_configuration()
    lstm_variant, lstm_units = lstm_configuration()
    print(f"LSTM base: variant={lstm_variant}, units={lstm_units}", flush=True)

    records, step_records = [], []
    base_rows, meta_rows, diagnostics = [], [], []

    for horizon in HORIZON_STEPS:
        minutes = horizon * STEP_MINUTES
        needed = sorted({config[(m, horizon)]["features"] for m in TREE_MODELS})
        data = {f: T.get_dataset(df, horizon, f) for f in needed}
        index = data[needed[0]][0].index
        positions = np.array([channels.index.get_loc(t) for t in index])
        sequences = L.build_sequences(channels, lstm_variant, positions)
        X_ref, Y_ref = data[needed[0]]
        folds = list(rolling_origin_folds(X_ref, Y_ref, horizon=horizon))

        base_names = list(TREE_MODELS) + ["lstm"]
        fold_preds, fold_index = [], []

        for fold in folds:
            started = time.perf_counter()
            tr_idx = fold["train"][0].index
            va_idx = fold["val"][0].index
            te_idx = fold["test"][0].index
            tr = np.searchsorted(index, tr_idx)
            va = np.searchsorted(index, va_idx)
            te = np.searchsorted(index, te_idx)
            Ytr = fold["train"][1].to_numpy(np.float32)
            Yva = fold["val"][1].to_numpy(np.float64)
            Yte = fold["test"][1].to_numpy(np.float64)

            n_train = len(tr)
            oof = np.full((n_train, horizon, len(base_names)), np.nan)
            splitter = TimeSeriesSplit(n_splits=INNER_SPLITS)

            # --- out-of-fold predictions on the training window ------------
            for inner_train, inner_test in splitter.split(np.arange(n_train)):
                for b, name in enumerate(base_names):
                    if name == "lstm":
                        preds = fit_predict_lstm(
                            lstm_units, horizon,
                            sequences[tr[inner_train]], Ytr[inner_train],
                            sequences[va], Yva.astype(np.float32),
                            [sequences[tr[inner_test]]])
                    else:
                        feats = config[(name, horizon)]["features"]
                        Xf = data[feats][0]
                        Xtr_all = Xf.to_numpy(np.float32)[tr]
                        preds = fit_predict_tree(
                            name, config[(name, horizon)]["params"], horizon,
                            Xtr_all[inner_train], Ytr[inner_train],
                            [Xtr_all[inner_test]])
                    oof[inner_test, :, b] = preds[0]

            # --- full-window fits -> validation and test -------------------
            val_stack = np.zeros((len(va), horizon, len(base_names)))
            test_stack = np.zeros((len(te), horizon, len(base_names)))
            for b, name in enumerate(base_names):
                if name == "lstm":
                    pv, pt = fit_predict_lstm(
                        lstm_units, horizon, sequences[tr], Ytr,
                        sequences[va], Yva.astype(np.float32),
                        [sequences[va], sequences[te]])
                else:
                    feats = config[(name, horizon)]["features"]
                    Xf = data[feats][0].to_numpy(np.float32)
                    pv, pt = fit_predict_tree(
                        name, config[(name, horizon)]["params"], horizon,
                        Xf[tr], Ytr, [Xf[va], Xf[te]])
                val_stack[:, :, b] = pv
                test_stack[:, :, b] = pt

                metric = scores(Yte, pt)
                base_rows.append({
                    "model": name, "horizon_steps": horizon,
                    "horizon_min": minutes, "fold": fold["fold"],
                    "features": (lstm_variant if name == "lstm"
                                 else config[(name, horizon)]["features"]),
                    **metric})
                # Distribution diagnostic, for the acceptance-failure path.
                covered = ~np.isnan(oof[:, 0, b])
                diagnostics.append({
                    "model": name, "horizon_steps": horizon,
                    "horizon_min": minutes, "fold": fold["fold"],
                    "oof_mean": float(np.nanmean(oof[covered, :, b])),
                    "oof_std": float(np.nanstd(oof[covered, :, b])),
                    "test_mean": float(pt.mean()), "test_std": float(pt.std()),
                    "oof_rmse": float(np.sqrt(np.nanmean(
                        (oof[covered, :, b] - Ytr[covered]) ** 2))),
                    "test_rmse": metric["rmse"],
                })

            # --- meta-learner ---------------------------------------------
            covered = ~np.isnan(oof[:, 0, 0])
            kind, meta, val_rmse, detail = select_meta(
                horizon, oof[covered], Ytr[covered].astype(np.float64),
                val_stack, Yva)
            pred = apply_meta(kind, meta, test_stack)

            seconds = time.perf_counter() - started
            records.append(fold_record(STACK_NAME, STACK_FEATURES, horizon,
                                       fold["fold"], "test", Yte, pred,
                                       run_time=seconds))
            step_records.extend(per_step_records(
                STACK_NAME, STACK_FEATURES, horizon, fold["fold"], "test",
                Yte, pred, run_time=seconds))
            meta_rows.append({
                "horizon_steps": horizon, "horizon_min": minutes,
                "fold": fold["fold"], "meta": kind,
                "val_rmse": val_rmse, "ridge_val_rmse": detail["ridge_rmse"],
                "mlp_val_rmse": detail["mlp_rmse"],
                "mlp_banned": horizon >= PER_STEP_FROM,
                "alphas": str(detail["alphas"]),
                "n_oof_rows": int(covered.sum()), "seconds": seconds,
            })
            fold_preds.append(pred.astype(np.float32))
            fold_index.append(te_idx.values)
            joblib.dump({"kind": kind, "meta": meta},
                        MODELS / f"stack_h{horizon}_fold{fold['fold']}.joblib",
                        compress=3)
            print(f"  h={minutes:>3} min fold {fold['fold']}  meta={kind:<15} "
                  f"val RMSE {val_rmse:.4f}  test RMSE "
                  f"{records[-1]['rmse']:.4f}  ({seconds:.0f}s)", flush=True)

        np.savez_compressed(
            PREDICTIONS / f"stack_h{horizon}.npz",
            **{f"fold{k}": p for k, p in enumerate(fold_preds)},
            **{f"index{k}": i for k, i in enumerate(fold_index)})
        del sequences

    return records, step_records, base_rows, meta_rows, diagnostics


# --------------------------------------------------------------------------
# acceptance criterion
# --------------------------------------------------------------------------
def check_acceptance(metrics: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Stack must be <= best single base model + 1 % at every horizon."""
    rows = metrics[(metrics["split"] == "test")
                   & (metrics["fold"].astype(str) == "mean")]
    stack = rows[rows["model"] == STACK_NAME]
    base_mean = (base.groupby(["model", "horizon_steps", "horizon_min"])["rmse"]
                 .mean().reset_index())

    out = []
    for horizon in HORIZON_STEPS:
        s = stack[stack["horizon_steps"] == horizon]
        b = base_mean[base_mean["horizon_steps"] == horizon]
        if s.empty or b.empty:
            continue
        best = b.loc[b["rmse"].idxmin()]
        stack_rmse = float(s.iloc[0]["rmse"])
        limit = float(best["rmse"]) * (1.0 + ACCEPTANCE_TOLERANCE)
        out.append({
            "horizon_steps": horizon, "horizon_min": horizon * STEP_MINUTES,
            "stack_rmse": stack_rmse, "best_base_model": best["model"],
            "best_base_rmse": float(best["rmse"]), "limit": limit,
            "delta_pct": 100.0 * (stack_rmse - float(best["rmse"]))
            / float(best["rmse"]),
            "passes": bool(stack_rmse <= limit),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def main() -> None:
    started = time.perf_counter()
    MODELS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Stacking ensemble: TimeSeriesSplit(5) OOF inside each rolling fold")
    print("=" * 78, flush=True)
    records, step_records, base_rows, meta_rows, diagnostics = run_stack()

    metrics, per_step = log_metrics(records, step_records)
    base = pd.DataFrame(base_rows)
    base.to_csv(BASE_METRICS_CSV, index=False)
    pd.DataFrame(meta_rows).to_csv(META_CHOICE_CSV, index=False)
    pd.DataFrame(diagnostics).to_csv(DIAGNOSTIC_CSV, index=False)

    acceptance = check_acceptance(metrics, base)
    acceptance.to_csv(ACCEPTANCE_CSV, index=False)
    print("\n" + "=" * 78)
    print("Acceptance criterion: stack <= best single base model + 1 %")
    print("=" * 78)
    print(acceptance.to_string(index=False))
    print(f"\ntotal wall clock: {(time.perf_counter() - started) / 60:.1f} min")

    build_outputs()


def report_only() -> None:
    build_outputs()


def build_outputs() -> None:
    from build_stack_report import build_all
    build_all()


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report_only()
    else:
        main()
