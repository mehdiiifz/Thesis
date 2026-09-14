"""Stacking ensemble with rolling-origin evaluation + final RQ1 deliverables.

Base models per horizon (from the completed leaderboard analysis):
- All horizons: tuned RF/ET/XGB/LGBM in BOTH feature-set variants (8 base
  models, params from results/selected_params.json) + lstm lags_only
  (fold-stable at every horizon).
- H=1 and H=2 ONLY: additionally lstm full (reduced-exog channels), the
  short-horizon champion; excluded at H>=4 (unstable at 12-24 h).

Within EACH of the 4 rolling-origin folds:
1. Out-of-fold base predictions on the fold's training window via
   TimeSeriesSplit(5) (never plain KFold). LSTM OOF fits early-stop on the
   temporally-last 15% of each inner training split (validation_split).
   Fold-level base fits reuse the saved fold-3 models from train_trees /
   train_lstm (they were trained on exactly fold 3's training window);
   earlier folds retrain since no saved model exists for them.
2. Meta-learner input = concatenated base predictions (B*H columns).
   Ridge vs small MLP (64 hidden units) compared on fold 0's validation
   block; the winner per horizon is reused for all folds (CLAUDE.md rule).
3. Test-block evaluation; per-fold + mean/std rows to the metrics files;
   predictions -> results/predictions_stack_{meta}_h{H}_fold{k}.csv.

Also writes the final RQ1 deliverables (figures + LaTeX table +
operational recommendation) from results/metrics*.csv.

Run from project root: python src/train_stack.py  (full run, hours on CPU)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_hourly  # noqa: E402
from make_dataset import HORIZONS, build_supervised  # noqa: E402
from metrics_utils import (METRICS_CSV, METRICS_PER_STEP_CSV,  # noqa: E402
                           leaderboard_rolling, log_fold_summaries,
                           log_metrics, log_metrics_per_step,
                           log_per_step_summaries, scores)
from rolling_eval import rolling_origin_folds  # noqa: E402
from train_lstm import (CANDIDATES, N_STEPS, WIND_SPEC, build_model,  # noqa: E402
                        build_sequences, to_3d)
from train_trees import fit_predict as tree_fit_predict  # noqa: E402

SEED = 42
MAX_EPOCHS = 100
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"
FIGDIR = ROOT / "figures"

TREE_NAMES = ["random_forest", "extra_trees", "xgboost", "lightgbm"]
TREE_PARAMS = json.loads((RESULTS / "selected_params.json").read_text())
LSTM_UNITS = json.loads((RESULTS / "lstm_units.json").read_text())
LSTM_SPECS = {"lags_only": WIND_SPEC, "full": CANDIDATES["a_reduced_exog"]}


def base_models_for(horizon: int) -> list[tuple[str, str]]:
    """(model, features) base models for one horizon (see module docstring)."""
    bases = [(n, f) for n in TREE_NAMES for f in ("lags_only", "full")]
    bases.append(("lstm", "lags_only"))
    if horizon <= 2:
        bases.append(("lstm", "full"))
    return bases


def lstm_spec(features: str, horizon: int) -> dict:
    spec = dict(LSTM_SPECS[features])
    if spec.get("tune_units"):
        spec["units"] = LSTM_UNITS[features][str(horizon)]
    return spec


def lstm_train(spec: dict, horizon: int, Xtr: pd.DataFrame, Ytr: pd.DataFrame,
               val=None):
    """Train an LSTM; early-stop on `val` (block) or the last 15% of train.

    Returns (model, prep) where prep scales+reshapes a flat frame using the
    scaler fit on Xtr only.
    """
    C = len(spec["channels"])
    scaler = StandardScaler().fit(to_3d(Xtr, C).reshape(-1, C))

    def prep(Xf):
        a = to_3d(Xf, C)
        return scaler.transform(a.reshape(-1, C)).reshape(a.shape)

    keras.utils.set_random_seed(SEED)
    model = build_model(horizon, C, spec)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)
    kw = (dict(validation_data=(prep(val[0]), val[1].values)) if val
          else dict(validation_split=0.15))
    model.fit(prep(Xtr), Ytr.values, epochs=MAX_EPOCHS, batch_size=64,
              verbose=0, callbacks=[es], **kw)
    return model, prep


class BaseRunner:
    """Fit/predict one base model on arbitrary slices of one horizon's data."""

    def __init__(self, name, features, horizon, X, Y):
        self.name, self.features, self.h = name, features, horizon
        self.X, self.Y = X, Y

    def _fit(self, Xtr, Ytr, val=None):
        if self.name == "lstm":
            model, prep = lstm_train(lstm_spec(self.features, self.h),
                                     self.h, Xtr, Ytr, val)
            return lambda Xf: model.predict(prep(Xf), verbose=0,
                                            batch_size=512)
        params = TREE_PARAMS[f"{self.name}_{self.features}_h{self.h}"]
        model, _ = tree_fit_predict(self.name, params, Xtr, Ytr,
                                    Xtr.iloc[:1], self.h)
        return lambda Xf: np.asarray(model.predict(Xf.values)).reshape(
            len(Xf), self.h)

    def _saved_fold3(self):
        """Predictor from the saved fold-3 model, or None."""
        if self.name == "lstm":
            p = MODELS_DIR / f"lstm_{self.features}_h{self.h}.keras"
            if not p.exists():
                return None
            model = keras.models.load_model(p)
            C = len(lstm_spec(self.features, self.h)["channels"])
            holder = {}

            def predict(Xf):
                a = to_3d(Xf, C)
                a = holder["scaler"].transform(
                    a.reshape(-1, C)).reshape(a.shape)
                return model.predict(a, verbose=0, batch_size=512)

            predict.needs_scaler = True
            predict.set_scaler = lambda Xtr: holder.__setitem__(
                "scaler", StandardScaler().fit(to_3d(Xtr, C).reshape(-1, C)))
            return predict
        p = MODELS_DIR / f"{self.name}_{self.features}_h{self.h}.joblib"
        if not p.exists():
            return None
        model = joblib.load(p)
        return lambda Xf: np.asarray(model.predict(Xf.values)).reshape(
            len(Xf), self.h)

    def oof_and_block_preds(self, fold, is_last_fold):
        """OOF predictions on the training window (TimeSeriesSplit(5)) plus
        val/test-block predictions from a full-training-window fit."""
        Xtr, Ytr = fold["train"]
        n = len(Xtr)
        oof = np.full((n, self.h), np.nan)
        for tr, te in TimeSeriesSplit(n_splits=5).split(np.arange(n)):
            predict = self._fit(Xtr.iloc[tr], Ytr.iloc[tr])
            oof[te] = predict(Xtr.iloc[te])
        predict = self._saved_fold3() if is_last_fold else None
        if predict is not None and getattr(predict, "needs_scaler", False):
            predict.set_scaler(Xtr)
        if predict is None:
            predict = self._fit(Xtr, Ytr, val=fold["val"])
        return oof, predict(fold["val"][0]), predict(fold["test"][0])


def make_meta(kind: str):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000,
                                      random_state=SEED))


def meta_fit_predict(kind, Ztr, ytr, blocks, horizon):
    m = make_meta(kind)
    y = ytr if horizon > 1 else ytr.ravel()
    m.fit(Ztr, y)
    return [np.asarray(m.predict(Z)).reshape(len(Z), horizon) for Z in blocks]


def run_stack_horizon(df: pd.DataFrame, horizon: int) -> str:
    """Full stacking procedure for one horizon. Returns winning meta name."""
    bases = base_models_for(horizon)
    data = {}
    for name, feats in bases:
        if name == "lstm":
            X, Y = build_sequences(df, horizon,
                                   lstm_spec(feats, horizon)["channels"])
        else:
            X, Y, _ = build_supervised(df, horizon, features=feats)
        data[(name, feats)] = (X, Y)
    canonical = next(iter(data.values()))
    folds = {b: list(rolling_origin_folds(*data[b], horizon=horizon))
             for b in data}
    for b in data:
        assert (data[b][0].index == canonical[0].index).all()

    winner = None
    test_scores, val_scores, steps, run_times = [], [], [], []
    for k in range(4):
        t0 = time.time()
        oofs, vals, tests = [], [], []
        for b in bases:
            runner = BaseRunner(*b, horizon, *data[b])
            oof, pv, pt = runner.oof_and_block_preds(folds[b][k], k == 3)
            oofs.append(oof)
            vals.append(pv)
            tests.append(pt)
        fold = folds[bases[0]][k]
        covered = ~np.isnan(oofs[0][:, 0])
        Ztr = np.hstack([o[covered] for o in oofs])
        ytr = fold["train"][1].values[covered]
        Zval, Zte = np.hstack(vals), np.hstack(tests)
        if k == 0:  # meta selection on fold 0's validation block only
            cand = {}
            for kind in ("ridge", "mlp"):
                pv, = meta_fit_predict(kind, Ztr, ytr, [Zval], horizon)
                cand[kind] = scores(fold["val"][1].values, pv)["rmse"]
            winner = min(cand, key=cand.get)
            print(f"  H={horizon}: meta ridge {cand['ridge']:.3f} vs "
                  f"mlp {cand['mlp']:.3f} (fold-0 val) -> {winner}",
                  flush=True)
        pred_val, pred = meta_fit_predict(winner, Ztr, ytr, [Zval, Zte],
                                          horizon)
        rt = time.time() - t0
        sc_val = scores(fold["val"][1].values, pred_val)
        sc_test = scores(fold["test"][1].values, pred)
        log_metrics("stack", winner, horizon, k, "val", sc_val, rt)
        log_metrics("stack", winner, horizon, k, "test", sc_test, rt)
        steps.append(log_metrics_per_step("stack", winner, horizon, k,
                                          "test", fold["test"][1].values,
                                          pred, rt))
        out = pd.DataFrame(index=fold["test"][1].index)
        for j, col in enumerate(fold["test"][1].columns):
            out[f"true_{col}"] = fold["test"][1][col].values
            out[f"pred_{col}"] = pred[:, j]
        out.to_csv(RESULTS /
                   f"predictions_stack_{winner}_h{horizon}_fold{k}.csv")
        val_scores.append(sc_val)
        test_scores.append(sc_test)
        run_times.append(rt)
    log_fold_summaries("stack", winner, horizon, "val", val_scores, run_times)
    s = log_fold_summaries("stack", winner, horizon, "test", test_scores,
                           run_times)
    log_per_step_summaries("stack", winner, horizon, "test", steps, run_times)
    print(f"  stack ({winner}) H={horizon:2d}: test RMSE {s['rmse_mean']:.3f} "
          f"+/- {s['rmse_std']:.3f}  R {s['r_mean']:.3f}  "
          f"({sum(run_times):.0f}s)", flush=True)
    return winner


# ------------------------ per-step repair (H>=12) ------------------------
# The flat (B*H)-feature meta stack is pathological at long horizons: its
# meta-features come from TSS inner models trained on small windows, whose
# prediction distribution does not match the full-window fits seen at
# val/test time, and the 216-dim meta amplifies the mismatch. Repair:
# per-step stacking - for each lead step j the meta input is ONLY the five
# base models' step-j predictions. Ridge (alpha grid, per step, chosen on
# the fold's validation block) vs a non-negative least-squares blend;
# whichever validates better per fold is used. MLP is banned at H >= 12.

REPAIR_HORIZONS = [12, 24]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
# better feature-set variant per tree family at H>=12 (from the leaderboard)
BEST_VARIANT = {"random_forest": "lags_only", "extra_trees": "full",
                "xgboost": "lags_only", "lightgbm": "lags_only"}


def base_models_repair(horizon: int) -> list[tuple[str, str]]:
    """5 base models for per-step stacking: best tree variants + LSTM."""
    return ([(n, BEST_VARIANT[n]) for n in TREE_NAMES]
            + [("lstm", "lags_only")])


def perstep_fit_predict(kind, Zoof, ytr, Zval, yval, Zte, horizon):
    """Fit one per-step meta per lead time; return (val, test) predictions.

    Zoof/Zval/Zte: (n, B, H) stacked base predictions. Ridge alpha is
    selected per step on the fold's validation block; NNLS has no knobs.
    """
    pv = np.empty((len(Zval), horizon))
    pt = np.empty((len(Zte), horizon))
    for j in range(horizon):
        A, b = Zoof[:, :, j], ytr[:, j]
        if kind == "ridge":
            best_m, best_r = None, np.inf
            for a in RIDGE_ALPHAS:
                m = Ridge(alpha=a).fit(A, b)
                r = float(np.sqrt(np.mean(
                    (m.predict(Zval[:, :, j]) - yval[:, j]) ** 2)))
                if r < best_r:
                    best_m, best_r = m, r
            pv[:, j] = best_m.predict(Zval[:, :, j])
            pt[:, j] = best_m.predict(Zte[:, :, j])
        else:  # non-negative least squares blend
            w, _ = nnls(A, b)
            pv[:, j] = Zval[:, :, j] @ w
            pt[:, j] = Zte[:, :, j] @ w
    return pv, pt


def run_stack_perstep(df: pd.DataFrame, horizon: int) -> None:
    """Repaired per-step stack for one long horizon, all 4 folds."""
    bases = base_models_repair(horizon)
    data = {}
    for name, feats in bases:
        if name == "lstm":
            X, Y = build_sequences(df, horizon,
                                   lstm_spec(feats, horizon)["channels"])
        else:
            X, Y, _ = build_supervised(df, horizon, features=feats)
        data[(name, feats)] = (X, Y)
    folds = {b: list(rolling_origin_folds(*data[b], horizon=horizon))
             for b in data}
    test_scores, val_scores, steps, run_times = [], [], [], []
    for k in range(4):
        t0 = time.time()
        oofs, vals, tests = [], [], []
        for b in bases:
            runner = BaseRunner(*b, horizon, *data[b])
            oof, pv, pt = runner.oof_and_block_preds(folds[b][k], k == 3)
            oofs.append(oof)
            vals.append(pv)
            tests.append(pt)
        fold = folds[bases[0]][k]
        covered = ~np.isnan(oofs[0][:, 0])
        Zoof = np.stack([o[covered] for o in oofs], axis=1)
        ytr = fold["train"][1].values[covered]
        Zval = np.stack(vals, axis=1)
        Zte = np.stack(tests, axis=1)
        yval, yte = fold["val"][1].values, fold["test"][1].values
        cand = {}
        for kind in ("ridge", "nnls"):
            pv, pt = perstep_fit_predict(kind, Zoof, ytr, Zval, yval, Zte,
                                         horizon)
            cand[kind] = (scores(yval, pv)["rmse"], pv, pt)
        winner = min(cand, key=lambda c: cand[c][0])
        _, pred_val, pred = cand[winner]
        rt = time.time() - t0
        print(f"  H={horizon} fold {k}: ridge {cand['ridge'][0]:.3f} vs "
              f"nnls {cand['nnls'][0]:.3f} (val) -> {winner}", flush=True)
        sc_val = scores(yval, pred_val)
        sc_test = scores(yte, pred)
        log_metrics("stack", "perstep", horizon, k, "val", sc_val, rt)
        log_metrics("stack", "perstep", horizon, k, "test", sc_test, rt)
        steps.append(log_metrics_per_step("stack", "perstep", horizon, k,
                                          "test", yte, pred, rt))
        out = pd.DataFrame(index=fold["test"][1].index)
        for j, col in enumerate(fold["test"][1].columns):
            out[f"true_{col}"] = fold["test"][1][col].values
            out[f"pred_{col}"] = pred[:, j]
        out.to_csv(RESULTS /
                   f"predictions_stack_perstep_h{horizon}_fold{k}.csv")
        val_scores.append(sc_val)
        test_scores.append(sc_test)
        run_times.append(rt)
    log_fold_summaries("stack", "perstep", horizon, "val", val_scores,
                       run_times)
    s = log_fold_summaries("stack", "perstep", horizon, "test", test_scores,
                           run_times)
    log_per_step_summaries("stack", "perstep", horizon, "test", steps,
                           run_times)
    print(f"  stack (perstep) H={horizon:2d}: test RMSE {s['rmse_mean']:.3f} "
          f"+/- {s['rmse_std']:.3f}  R {s['r_mean']:.3f}", flush=True)


def purge_stack_horizons(horizons: list[int]) -> None:
    """Remove stack rows/files for the given horizons only."""
    for path in (METRICS_CSV, METRICS_PER_STEP_CSV):
        if path.exists():
            d = pd.read_csv(path)
            drop = (d["model"] == "stack") & (d["horizon"].isin(horizons))
            d[~drop].to_csv(path, index=False)
    for H in horizons:
        for p in RESULTS.glob(f"predictions_stack_*_h{H}_fold*.csv"):
            p.unlink()


def acceptance_check() -> bool:
    """Repaired stack must be <= best single base model * 1.01 at H>=12."""
    df = pd.read_csv(METRICS_CSV)
    m = df[(df["split"] == "test") & (df["fold"] == "mean")]
    ok = True
    for H in REPAIR_HORIZONS:
        sub = m[m["horizon"] == H]
        stack = sub[sub["model"] == "stack"]["rmse"].min()
        best_single = sub[sub["model"] != "stack"]["rmse"].min()
        passed = stack <= best_single * 1.01
        ok &= passed
        print(f"  acceptance H={H}: stack {stack:.3f} vs best single "
              f"{best_single:.3f} (+1% = {best_single * 1.01:.3f}) -> "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
    return ok


def main_repair() -> None:
    """Diagnose-driven repair: rerun ONLY H=12/24 with per-step stacking."""
    t0 = time.time()
    purge_stack_horizons(REPAIR_HORIZONS)
    df = load_hourly()
    for H in REPAIR_HORIZONS:
        print(f"[{time.time() - t0:6.0f}s] per-step stacking H={H}",
              flush=True)
        run_stack_perstep(df, H)
    ok = acceptance_check()
    if ok:
        fig_rmse_vs_leadtime()
        fig_nrmse_bars()
        fig_stack_per_step_h24()
        fig_case_study(*best_h24_model())
        write_final_table()
        write_operational_recommendation()
        print("deliverables regenerated", flush=True)
    else:
        print("ACCEPTANCE FAILED - deliverables NOT regenerated; "
              "review diagnostics before further changes", flush=True)
    print(f"\nTotal {time.time() - t0:.0f}s. Leaderboard:", flush=True)
    print(leaderboard_rolling().to_string())


# ----------------------------- deliverables -----------------------------

FAMILY_COLORS = {"persistence": "0.45", "seasonal_persistence": "0.7",
                 "random_forest": "tab:blue", "extra_trees": "tab:orange",
                 "xgboost": "tab:green", "lightgbm": "tab:red",
                 "lstm": "tab:purple", "stack": "black"}


def _mean_std(df):
    return df[df["fold"] == "mean"], df[df["fold"] == "std"]


def fig_rmse_vs_leadtime() -> None:
    df = pd.read_csv(METRICS_PER_STEP_CSV)
    df = df[df["split"] == "test"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, H in zip(axes.ravel(), HORIZONS):
        sub = df[df["horizon"] == H]
        for (model, feats), g in sub.groupby(["model", "features"]):
            mean, std = _mean_std(g.sort_values("step"))
            ls = "-" if feats in ("full", "ridge", "mlp", "perstep") else "--"
            lw = 2.2 if model == "stack" else 1.2
            if model in ("persistence", "seasonal_persistence"):
                ls = ":" if model == "seasonal_persistence" else "--"
            ax.plot(mean["step"], mean["rmse"], ls, lw=lw,
                    color=FAMILY_COLORS[model],
                    marker="o" if H == 1 else None, ms=4,
                    label=f"{model} ({feats})")
            if H > 1 and model == "stack":
                ax.fill_between(mean["step"],
                                mean["rmse"] - std["rmse"].values,
                                mean["rmse"] + std["rmse"].values,
                                color="black", alpha=0.10)
        ax.set_title(f"H = {H} h")
        ax.set_xlabel("lead time (h)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("RMSE (m/s)")
    axes[1, 0].set_ylabel("RMSE (m/s)")
    handles, labels = axes.ravel()[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7)
    fig.suptitle("Cross-fold mean test RMSE vs lead time, all models "
                 "(band: stack +/- 1 std)")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(FIGDIR / "final_rmse_vs_leadtime.png", dpi=150)
    plt.close(fig)


def _best_variant_rows(mean: pd.DataFrame) -> pd.DataFrame:
    """Per model family and horizon, keep the variant with lowest rmse."""
    return mean.loc[mean.groupby(
        [mean["model"], mean["horizon"]])["rmse"].idxmin()]


def fig_nrmse_bars() -> None:
    df = pd.read_csv(METRICS_CSV)
    mean = df[(df["split"] == "test") & (df["fold"] == "mean")]
    best = _best_variant_rows(mean)
    families = ["persistence", "seasonal_persistence", "random_forest",
                "extra_trees", "xgboost", "lightgbm", "lstm", "stack"]
    width = 0.1
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(HORIZONS))
    for i, fam in enumerate(families):
        vals = [best[(best["model"] == fam) & (best["horizon"] == H)
                     ]["nrmse"].squeeze() for H in HORIZONS]
        ax.bar(xs + (i - len(families) / 2 + 0.5) * width, vals, width,
               color=FAMILY_COLORS[fam], label=fam)
    ax.axhline(0.40, color="red", ls="--", lw=1.2,
               label="Mito et al. 2023, 24 h (nrmse = 0.40)")
    ax.set_xticks(xs, [f"{H} h" for H in HORIZONS])
    ax.set_ylabel("NRMSE (rmse / mean observed)")
    ax.set_title("Cross-fold mean NRMSE by horizon "
                 "(best feature-set variant per model)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "final_nrmse_by_horizon.png", dpi=150)
    plt.close(fig)


def fig_stack_per_step_h24() -> None:
    df = pd.read_csv(METRICS_PER_STEP_CSV)
    df = df[(df["split"] == "test") & (df["horizon"] == 24)]
    picks = [("stack", None, "black", "stack"),
             ("lstm", "lags_only", "tab:purple", "lstm lags_only"),
             ("extra_trees", "full", "tab:orange", "extra_trees full")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, feats, color, label in picks:
        g = df[df["model"] == model]
        if feats:
            g = g[g["features"] == feats]
        mean, std = _mean_std(g.sort_values("step"))
        ax.plot(mean["step"], mean["rmse"], color=color, lw=2, label=label)
        ax.fill_between(mean["step"], mean["rmse"] - std["rmse"].values,
                        mean["rmse"] + std["rmse"].values, color=color,
                        alpha=0.12)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("H = 24 h: per-step test RMSE - stack vs its parents "
                 "(bands: +/- 1 std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "stack_per_step_h24.png", dpi=150)
    plt.close(fig)


def fig_case_study(best_model: str, best_feats: str) -> None:
    """Windy / calm / sharp-drop 24 h case studies from DIFFERENT folds."""
    preds = {}
    for k in range(4):
        p = RESULTS / f"predictions_{best_model}_{best_feats}_h24_fold{k}.csv"
        preds[k] = pd.read_csv(p, index_col=0, parse_dates=[0])
    true_cols = [f"true_t+{h}h" for h in range(1, 25)]
    pred_cols = [f"pred_t+{h}h" for h in range(1, 25)]

    def candidates(metric):
        return {k: (metric(d[true_cols].values), d) for k, d in preds.items()}

    scores_ = {
        "windy day": {k: v[0].max() for k, v in
                      candidates(lambda a: a.mean(axis=1)).items()},
        "calm day": {k: -v[0].min() for k, v in
                     candidates(lambda a: a.mean(axis=1)).items()},
        "sharp drop": {k: v[0].max() for k, v in candidates(
            lambda a: np.maximum.accumulate(a, axis=1).max(axis=1)
            - a[:, -1]).items()},
    }
    used = set()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (label, per_fold) in zip(axes, scores_.items()):
        k = max((f for f in per_fold if f not in used), key=per_fold.get)
        used.add(k)
        d = preds[k]
        a = d[true_cols].values
        if label == "windy day":
            i = int(a.mean(axis=1).argmax())
        elif label == "calm day":
            i = int(a.mean(axis=1).argmin())
        else:
            i = int((np.maximum.accumulate(a, axis=1).max(axis=1)
                     - a[:, -1]).argmax())
        t = d.index[i]
        leads = np.arange(1, 25)
        ax.plot(leads, d.iloc[i][true_cols].values, "k-", lw=2,
                label="observed")
        ax.plot(leads, d.iloc[i][pred_cols].values, "-", lw=2,
                color="tab:red", label="24 h forecast")
        ax.set_title(f"{label} (fold {k})\nissued {t:%Y-%m-%d %H:%M}")
        ax.set_xlabel("lead time (h)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ws100 (m/s)")
    axes[0].legend()
    fig.suptitle(f"24 h case studies - {best_model} ({best_feats})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "case_study_days.png", dpi=150)
    plt.close(fig)


def write_final_table() -> None:
    """LaTeX table: per horizon block, per model rmse+/-std, nrmse, r."""
    df = pd.read_csv(METRICS_CSV)
    test = df[df["split"] == "test"]
    mean, std = _mean_std(test)
    key = ["model", "features", "horizon"]
    m = mean.merge(std[key + ["rmse"]], on=key, suffixes=("", "_sd"))
    lines = ["% Auto-generated by src/train_stack.py",
             "\\begin{tabular}{llrrr}", "\\toprule",
             "Horizon & Model & RMSE (m/s) & NRMSE & R \\\\"]
    for H in HORIZONS:
        sub = m[m["horizon"] == H].sort_values("rmse")
        lines.append("\\midrule")
        best_rmse = sub["rmse"].min()
        best_nrmse = sub["nrmse"].min()
        best_r = sub["r"].max()
        for i, (_, row) in enumerate(sub.iterrows()):
            def fmt(v, best, digits=3, maximize=False):
                s = f"{v:.{digits}f}"
                is_best = np.isclose(v, best)
                return f"\\textbf{{{s}}}" if is_best else s
            rmse_s = (f"{row.rmse:.3f} $\\pm$ {row.rmse_sd:.3f}")
            if np.isclose(row.rmse, best_rmse):
                rmse_s = f"\\textbf{{{rmse_s}}}"
            name = f"{row.model} ({row.features})".replace("_", "\\_")
            hcol = f"{H} h" if i == 0 else ""
            lines.append(f"{hcol} & {name} & {rmse_s} & "
                         f"{fmt(row.nrmse, best_nrmse)} & "
                         f"{fmt(row.r, best_r, maximize=True)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "",
              "\\smallskip\\noindent{\\footnotesize RMSE is pooled over "
              "all test samples and all lead-time steps jointly "
              "($\\sqrt{\\mathrm{mean}(\\mathrm{RMSE}_\\mathrm{step}^2)}$), "
              "not the mean of per-step RMSEs; mean $\\pm$ std over the "
              "4 rolling-origin folds.}"]
    (RESULTS / "final_results_table.tex").write_text("\n".join(lines) + "\n")


def write_operational_recommendation() -> None:
    df = pd.read_csv(METRICS_CSV)
    test = df[df["split"] == "test"]
    mean, std = _mean_std(test)
    key = ["model", "features", "horizon"]
    m = mean.merge(std[key + ["rmse"]], on=key, suffixes=("", "_sd"))
    lines = ["# Operational model recommendation (RQ1)", "",
             "Lowest cross-fold mean test RMSE per horizon "
             "(rolling-origin, 4 folds):", "",
             "| Horizon | Recommended model | RMSE (m/s) | NRMSE | R |",
             "|---|---|---|---|---|"]
    for H in HORIZONS:
        row = m[m["horizon"] == H].sort_values("rmse").iloc[0]
        lines.append(f"| {H} h | {row.model} ({row.features}) | "
                     f"{row.rmse:.3f} +/- {row.rmse_sd:.3f} | "
                     f"{row.nrmse:.3f} | {row.r:.3f} |")
    lines += ["", "Persistence is the fallback if the pipeline is "
              "unavailable; seasonal-persistence is never competitive.",
              "",
              "RMSE is pooled over all test samples and all lead-time steps "
              "jointly (= sqrt(mean(per-step RMSE^2))), not the mean of "
              "per-step RMSEs; mean +/- std over the 4 rolling-origin folds."]
    (RESULTS / "operational_recommendation.md").write_text(
        "\n".join(lines) + "\n")


def best_h24_model() -> tuple[str, str]:
    df = pd.read_csv(METRICS_CSV)
    m = df[(df["split"] == "test") & (df["fold"] == "mean")
           & (df["horizon"] == 24)]
    row = m.sort_values("rmse").iloc[0]
    return row.model, row.features


def purge_previous() -> None:
    for path in (METRICS_CSV, METRICS_PER_STEP_CSV):
        if path.exists():
            d = pd.read_csv(path)
            d[d["model"] != "stack"].to_csv(path, index=False)
    for p in RESULTS.glob("predictions_stack_*.csv"):
        p.unlink()


def main() -> None:
    t0 = time.time()
    purge_previous()
    df = load_hourly()
    for H in HORIZONS:
        print(f"[{time.time() - t0:6.0f}s] stacking H={H}", flush=True)
        run_stack_horizon(df, H)
    fig_rmse_vs_leadtime()
    fig_nrmse_bars()
    fig_stack_per_step_h24()
    fig_case_study(*best_h24_model())
    write_final_table()
    write_operational_recommendation()
    print(f"\nTotal {time.time() - t0:.0f}s. Final leaderboard:", flush=True)
    print(leaderboard_rolling().to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "repair":
        main_repair()
    else:
        main()
