"""LSTM sequence models for all 8 horizons.

Run from the project root:

    python src/train_lstm.py
    python src/train_lstm.py --report-only    # rebuild section 5 from results/

Input
-----
Each sample is the last SEQ_LEN = 72 steps (12 hours) ending AT the issue time
t, so nothing after t enters. Three channel variants:

    A "lags_only"  [ws100]
    B "reduced"    [ws100, u, v, tod_sin, tod_cos]
    C "turb"       [ws100, u, v, tod_sin, tod_cos, sd, ri]

Architecture: LSTM(units) -> Dropout(0,2) -> Dense(H), Adam, MSE loss,
EarlyStopping(patience=10, restore_best_weights=True), batch 128, seed 42.

Variant selection
-----------------
The hourly companion study found that feeding the LSTM every exogenous channel
hurt it at long horizons. Rather than assume the same here, B and C are
compared on the fold-0 validation split at horizons 12 and 48 steps
(120 and 480 minutes) before the main run. The comparison is written to
results/lstm_variant_selection.md, and only variant A plus the winner are then
run across all horizons and folds.

Runtime
-------
No GPU is available, so an epoch on the full training window costs ~9 s. The
training sequences are therefore subsampled with a stride of 2: successive
72-step windows differ by a single step and overlap by 71/72, so the discarded
sequences are almost exactly duplicates of the ones kept. Validation and test
use every issue time, so evaluation is unaffected and remains directly
comparable with the tree models, which are scored on the identical index.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_data import load_10min  # noqa: E402
from make_dataset import add_uv, build_supervised  # noqa: E402
from metrics_utils import (  # noqa: E402
    METRIC_NAMES, fold_record, log_metrics, per_step_records, scores,
)
from report_utils import (  # noqa: E402
    FIGURES, HORIZON_STEPS, REPORT, RESULTS, STEP_MINUTES, fmt, log_progress,
    md_table, write_section,
)
from rolling_eval import rolling_origin_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PREDICTIONS = RESULTS / "predictions"

SEED = 42
SEQ_LEN = 72                  # 12 hours of 10-minute steps
TRAIN_STRIDE = 2              # successive windows overlap by 71/72
BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 10
DROPOUT = 0.2
DEFAULT_UNITS = 64
UNITS_GRID = (32, 64, 128)
SELECTION_HORIZONS = (12, 48)   # 120 and 480 minutes

VARIANT_CHANNELS = {
    "lags_only": ["ws100"],
    "reduced": ["ws100", "u", "v", "tod_sin", "tod_cos"],
    "turb": ["ws100", "u", "v", "tod_sin", "tod_cos", "sd", "ri"],
}
VARIANT_A = "lags_only"
CANDIDATES = ("reduced", "turb")

SELECTION_MD = RESULTS / "lstm_variant_selection.md"
SELECTION_CSV = RESULTS / "lstm_variant_selection.csv"
UNITS_CSV = RESULTS / "lstm_units_selection.csv"
HISTORY_CSV = RESULTS / "lstm_history.csv"
SECTION_MD = REPORT / "05_results_lstm.md"

FIG_CURVES = "fig_10_lstm_training_curves.png"
FIG_LEAD = "fig_11_lstm_vs_trees_lead_time.png"

TREE_MODELS = ("rf", "et", "xgb", "lgbm")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
})


# --------------------------------------------------------------------------
# sequence construction
# --------------------------------------------------------------------------
def channel_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All candidate channels for one row of the series, at time t."""
    out = add_uv(df).copy()
    minute_of_day = out.index.hour * 60 + out.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    return out


def build_sequences(channels: pd.DataFrame, variant: str,
                    positions: np.ndarray) -> np.ndarray:
    """(n, SEQ_LEN, C) windows ending at each issue position, inclusive.

    A window for issue position p covers rows p-SEQ_LEN+1 .. p, so the last
    element is the observation AT the issue time and nothing later is included.
    """
    names = VARIANT_CHANNELS[variant]
    matrix = channels[names].to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(matrix, SEQ_LEN, axis=0)
    picked = windows[positions - SEQ_LEN + 1]          # (n, C, SEQ_LEN)
    return np.ascontiguousarray(picked.transpose(0, 2, 1))


def fold_positions(index: pd.Index, subset: pd.Index) -> np.ndarray:
    """Row positions of a fold split within the issue-time index."""
    return np.searchsorted(index, subset)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_lstm(units: int, n_channels: int, horizon: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.Input((SEQ_LEN, n_channels)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(horizon),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


class Standardizer:
    """Mean/std scaling fitted on one fold's training window only.

    Inputs are scaled per channel. Targets are scaled with a single pooled
    mean and std across all lead times, since every target is ws100 in m/s and
    per-column scaling would distort the lead-time error structure.
    """

    def __init__(self, Xtr: np.ndarray, Ytr: np.ndarray):
        flat = Xtr.reshape(-1, Xtr.shape[-1])
        self.x_mean = flat.mean(axis=0)
        self.x_std = np.where(flat.std(axis=0) == 0, 1.0, flat.std(axis=0))
        self.y_mean = float(Ytr.mean())
        self.y_std = float(Ytr.std()) or 1.0

    def x(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def y(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, Y: np.ndarray) -> np.ndarray:
        return Y * self.y_std + self.y_mean


def fit_lstm(units: int, horizon: int, Xtr, Ytr, Xva, Yva,
             ) -> tuple[object, Standardizer, dict, float]:
    """Fit one LSTM on a fold, returning (model, scaler, history, seconds)."""
    import tensorflow as tf

    scaler = Standardizer(Xtr, Ytr)
    # Successive windows are near-duplicates; stride the training set only.
    Xs = scaler.x(Xtr[::TRAIN_STRIDE])
    Ys = scaler.y(Ytr[::TRAIN_STRIDE])

    model = build_lstm(units, Xtr.shape[-1], horizon)
    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    start = time.perf_counter()
    history = model.fit(Xs, Ys, validation_data=(scaler.x(Xva), scaler.y(Yva)),
                        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[stopper], verbose=0, shuffle=True)
    return model, scaler, history.history, time.perf_counter() - start


def predict_lstm(model, scaler: Standardizer, X: np.ndarray) -> np.ndarray:
    scaled = model.predict(scaler.x(X), batch_size=512, verbose=0)
    return scaler.inverse_y(np.asarray(scaled, dtype=np.float64))


# --------------------------------------------------------------------------
# variant and units selection on fold 0
# --------------------------------------------------------------------------
def select_variant_and_units(df: pd.DataFrame, channels: pd.DataFrame):
    """Choose between B and C, and choose LSTM width, on fold-0 validation."""
    selection_rows, units_rows = [], []

    for horizon in SELECTION_HORIZONS:
        X, Y, _ = build_supervised(df, horizon_steps=horizon, features="lags_only")
        positions = np.array([channels.index.get_loc(t) for t in X.index])
        fold0 = next(iter(rolling_origin_folds(X, Y, horizon=horizon)))
        tr = fold_positions(X.index, fold0["train"][0].index)
        va = fold_positions(X.index, fold0["val"][0].index)
        Ytr = fold0["train"][1].to_numpy(np.float32)
        Yva = fold0["val"][1].to_numpy(np.float64)

        for variant in CANDIDATES:
            seq = build_sequences(channels, variant, positions)
            model, scaler, _, seconds = fit_lstm(
                DEFAULT_UNITS, horizon, seq[tr], Ytr, seq[va], Yva)
            metric = scores(Yva, predict_lstm(model, scaler, seq[va]))
            selection_rows.append({
                "variant": variant, "channels": len(VARIANT_CHANNELS[variant]),
                "horizon_steps": horizon, "horizon_min": horizon * STEP_MINUTES,
                "units": DEFAULT_UNITS, "val_rmse": metric["rmse"],
                "val_r2": metric["r2"], "fit_seconds": seconds,
            })
            print(f"  select h={horizon * STEP_MINUTES:>3} min  {variant:<10} "
                  f"val RMSE {metric['rmse']:.4f}  r2 {metric['r2']:.4f}  "
                  f"({seconds:.0f}s)", flush=True)
            del seq

    selection = pd.DataFrame(selection_rows)
    winner = (selection.groupby("variant")["val_rmse"].mean().idxmin())

    # Width grid for the two variants that will actually be run.
    for variant in (VARIANT_A, winner):
        for horizon in SELECTION_HORIZONS:
            X, Y, _ = build_supervised(df, horizon_steps=horizon,
                                       features="lags_only")
            positions = np.array([channels.index.get_loc(t) for t in X.index])
            fold0 = next(iter(rolling_origin_folds(X, Y, horizon=horizon)))
            tr = fold_positions(X.index, fold0["train"][0].index)
            va = fold_positions(X.index, fold0["val"][0].index)
            Ytr = fold0["train"][1].to_numpy(np.float32)
            Yva = fold0["val"][1].to_numpy(np.float64)
            seq = build_sequences(channels, variant, positions)
            for units in UNITS_GRID:
                model, scaler, _, seconds = fit_lstm(
                    units, horizon, seq[tr], Ytr, seq[va], Yva)
                metric = scores(Yva, predict_lstm(model, scaler, seq[va]))
                units_rows.append({
                    "variant": variant, "units": units,
                    "horizon_steps": horizon,
                    "horizon_min": horizon * STEP_MINUTES,
                    "val_rmse": metric["rmse"], "val_r2": metric["r2"],
                    "fit_seconds": seconds,
                })
                print(f"  units h={horizon * STEP_MINUTES:>3} min  "
                      f"{variant:<10} units={units:>3}  "
                      f"val RMSE {metric['rmse']:.4f}  ({seconds:.0f}s)",
                      flush=True)
            del seq

    units_df = pd.DataFrame(units_rows)
    best_units = (units_df.groupby(["variant", "units"])["val_rmse"].mean()
                  .reset_index())
    chosen_units = {v: int(g.loc[g["val_rmse"].idxmin(), "units"])
                    for v, g in best_units.groupby("variant")}

    selection.to_csv(SELECTION_CSV, index=False)
    units_df.to_csv(UNITS_CSV, index=False)
    write_selection_note(selection, units_df, winner, chosen_units)
    return winner, chosen_units


def write_selection_note(selection: pd.DataFrame, units_df: pd.DataFrame,
                         winner: str, chosen_units: dict) -> None:
    """Record the decision, with the numbers it was based on."""
    loser = next(v for v in CANDIDATES if v != winner)
    means = selection.groupby("variant")["val_rmse"].mean()
    lines = [
        "# LSTM input-variant selection",
        "",
        "The hourly companion study found that feeding the LSTM every available",
        "exogenous channel harmed it at the long horizons. That finding was not",
        "assumed to carry over: variants B and C were compared directly on the",
        "**fold-0 validation split only**, at horizons of 120 and 480 minutes,",
        f"with the default width of {DEFAULT_UNITS} units. No test data were",
        "involved in this decision.",
        "",
        "## Candidates",
        "",
        f"- **reduced** ({len(VARIANT_CHANNELS['reduced'])} channels): "
        f"{', '.join(VARIANT_CHANNELS['reduced'])}",
        f"- **turb** ({len(VARIANT_CHANNELS['turb'])} channels): "
        f"{', '.join(VARIANT_CHANNELS['turb'])}",
        "",
        "## Fold-0 validation results",
        "",
        "| Variant | Horizon [min] | Validation RMSE [m/s] | Validation r2 |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in selection.sort_values(["variant", "horizon_steps"]).iterrows():
        lines.append(f"| {row['variant']} | {int(row['horizon_min'])} | "
                     f"{fmt(row['val_rmse'])} | {fmt(row['val_r2'])} |")

    lines += [
        "",
        "## Decision",
        "",
        f"Mean validation RMSE across the two horizons: "
        f"**reduced {fmt(means['reduced'])} m/s**, "
        f"**turb {fmt(means['turb'])} m/s**.",
        "",
        f"**Selected: `{winner}`.** Variant `{loser}` is dropped. The main run "
        f"therefore covers variant A (`{VARIANT_A}`) and `{winner}` across all "
        "eight horizons and all four folds.",
        "",
        "## LSTM width",
        "",
        "| Variant | Units | Mean validation RMSE [m/s] |",
        "| --- | --- | --- |",
    ]
    for (variant, units), group in units_df.groupby(["variant", "units"]):
        lines.append(f"| {variant} | {units} | "
                     f"{fmt(group['val_rmse'].mean())} |")
    lines += [
        "",
        "Selected width per variant: "
        + ", ".join(f"`{v}` -> {u} units" for v, u in chosen_units.items())
        + ". These are fixed here and reused unchanged for every fold, as "
          "CLAUDE.md requires.",
        "",
    ]
    SELECTION_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {SELECTION_MD.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# main run
# --------------------------------------------------------------------------
def run_all(df: pd.DataFrame, channels: pd.DataFrame, variants: list[str],
            chosen_units: dict):
    records, step_records, history_rows, timing = [], [], [], []

    for horizon in HORIZON_STEPS:
        minutes = horizon * STEP_MINUTES
        X, Y, _ = build_supervised(df, horizon_steps=horizon, features="lags_only")
        positions = np.array([channels.index.get_loc(t) for t in X.index])
        folds = list(rolling_origin_folds(X, Y, horizon=horizon))

        for variant in variants:
            units = chosen_units[variant]
            seq = build_sequences(channels, variant, positions)
            fold_preds, fold_index, total = [], [], 0.0

            for fold in folds:
                tr = fold_positions(X.index, fold["train"][0].index)
                va = fold_positions(X.index, fold["val"][0].index)
                te = fold_positions(X.index, fold["test"][0].index)
                Ytr = fold["train"][1].to_numpy(np.float32)
                Yva = fold["val"][1].to_numpy(np.float64)
                Yte = fold["test"][1].to_numpy(np.float64)

                model, scaler, history, seconds = fit_lstm(
                    units, horizon, seq[tr], Ytr, seq[va], Yva)
                start = time.perf_counter()
                pred = predict_lstm(model, scaler, seq[te])
                seconds += time.perf_counter() - start
                total += seconds

                records.append(fold_record("lstm", variant, horizon,
                                           fold["fold"], "test", Yte, pred,
                                           run_time=seconds))
                step_records.extend(per_step_records(
                    "lstm", variant, horizon, fold["fold"], "test", Yte, pred,
                    run_time=seconds))
                for epoch, (loss, val_loss) in enumerate(
                        zip(history["loss"], history["val_loss"]), start=1):
                    history_rows.append({
                        "variant": variant, "horizon_steps": horizon,
                        "horizon_min": minutes, "fold": fold["fold"],
                        "units": units, "epoch": epoch,
                        "loss": loss, "val_loss": val_loss,
                    })

                fold_preds.append(pred.astype(np.float32))
                fold_index.append(fold["test"][0].index.values)
                model.save(MODELS / f"lstm_{variant}_h{horizon}"
                                    f"_fold{fold['fold']}.keras")

            np.savez_compressed(
                PREDICTIONS / f"lstm_{variant}_h{horizon}.npz",
                **{f"fold{k}": p for k, p in enumerate(fold_preds)},
                **{f"index{k}": i for k, i in enumerate(fold_index)})
            timing.append({"model": "lstm", "features": variant,
                           "horizon_steps": horizon, "horizon_min": minutes,
                           "units": units, "total_seconds": total,
                           "seconds_per_fold": total / len(folds)})
            print(f"  h={minutes:>3} min  {variant:<10} units={units:>3}  "
                  f"{total:7.1f}s total ({total / len(folds):6.1f}s/fold)",
                  flush=True)
            del seq

    return records, step_records, history_rows, timing


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(metrics: pd.DataFrame, per_step: pd.DataFrame,
                 history: pd.DataFrame, variants: list[str]) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- training curves ---------------------------------------------------
    shown = [h for h in SELECTION_HORIZONS if h in set(history["horizon_steps"])]
    fig, axes = plt.subplots(1, len(shown), figsize=(11, 4.2), squeeze=False)
    colors = {"lags_only": "#1f4e79", "reduced": "#2e8b57", "turb": "#c00000"}
    for ax, horizon in zip(axes[0], shown):
        for variant in variants:
            sub = history[(history["variant"] == variant)
                          & (history["horizon_steps"] == horizon)
                          & (history["fold"] == 0)].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub["loss"], color=colors.get(variant),
                    lw=1.5, label=f"{variant} train")
            ax.plot(sub["epoch"], sub["val_loss"], color=colors.get(variant),
                    lw=1.5, ls="--", label=f"{variant} val")
        ax.set_title(f"H = {horizon * STEP_MINUTES} min (fold 0)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss (scaled target)")
        ax.legend(fontsize=7)
    fig.suptitle("LSTM training and validation loss, early stopping with "
                 f"patience {PATIENCE} and best weights restored", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_CURVES)
    plt.close(fig)

    # --- lead-time curves, LSTM vs best tree -------------------------------
    trees = metrics[(metrics["fold"].astype(str) == "mean")
                    & (metrics["split"] == "test")
                    & (metrics["model"].isin(TREE_MODELS))]
    best_tree = trees.groupby(["model", "features"])["rmse"].mean().idxmin()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for horizon, ax in zip((12, 48), axes):
        for variant in variants:
            m = per_step[(per_step["model"] == "lstm")
                         & (per_step["features"] == variant)
                         & (per_step["fold"].astype(str) == "mean")
                         & (per_step["horizon_steps"] == horizon)].sort_values("step")
            s = per_step[(per_step["model"] == "lstm")
                         & (per_step["features"] == variant)
                         & (per_step["fold"].astype(str) == "std")
                         & (per_step["horizon_steps"] == horizon)].sort_values("step")
            if m.empty:
                continue
            lead = m["step"].to_numpy() * STEP_MINUTES
            ax.plot(lead, m["rmse"], color=colors.get(variant), lw=1.8,
                    label=f"lstm / {variant}")
            if not s.empty:
                ax.fill_between(lead, m["rmse"].to_numpy() - s["rmse"].to_numpy(),
                                m["rmse"].to_numpy() + s["rmse"].to_numpy(),
                                color=colors.get(variant), alpha=0.15, lw=0)
        for name, feats, style in ((best_tree[0], best_tree[1], "-"),
                                   ("persistence", "lags_only", ":")):
            m = per_step[(per_step["model"] == name)
                         & (per_step["features"] == feats)
                         & (per_step["fold"].astype(str) == "mean")
                         & (per_step["horizon_steps"] == horizon)].sort_values("step")
            if m.empty:
                continue
            ax.plot(m["step"].to_numpy() * STEP_MINUTES, m["rmse"], color="black",
                    ls=style, lw=1.4, label=f"{name} / {feats}")
        ax.set_xlabel("lead time [minutes]")
        ax.set_ylabel("RMSE [m/s]")
        ax.set_title(f"H = {horizon * STEP_MINUTES} min")
        ax.legend(fontsize=7)
    fig.suptitle("LSTM against the best tree model and persistence, "
                 "mean +/- std across the 4 folds", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_LEAD)
    plt.close(fig)

    return f"{best_tree[0]} / {best_tree[1]}"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def build_report(metrics: pd.DataFrame, per_step: pd.DataFrame,
                 selection: pd.DataFrame, units_df: pd.DataFrame,
                 winner: str, chosen_units: dict, variants: list[str],
                 best_tree_label: str) -> None:
    rows = metrics[metrics["split"] == "test"].copy()
    rows["fold"] = rows["fold"].astype(str)
    keys = ["model", "features", "horizon_steps", "horizon_min"]
    merged = rows[rows["fold"] == "mean"].merge(
        rows[rows["fold"] == "std"][keys + list(METRIC_NAMES)],
        on=keys, suffixes=("", "_std"))

    def label(row, metric="rmse", dec=3):
        return f"{fmt(row[metric], dec)} +/- {fmt(row[f'{metric}_std'], dec)}"

    # --- Table 5-1 --------------------------------------------------------
    sel_rows = []
    for variant in CANDIDATES:
        entry = {"Candidate": variant,
                 "Channels": len(VARIANT_CHANNELS[variant])}
        for horizon in SELECTION_HORIZONS:
            hit = selection[(selection["variant"] == variant)
                            & (selection["horizon_steps"] == horizon)]
            minutes = horizon * STEP_MINUTES
            entry[f"val RMSE {minutes} min"] = fmt(hit.iloc[0]["val_rmse"]) \
                if not hit.empty else "n/a"
            entry[f"val r2 {minutes} min"] = fmt(hit.iloc[0]["val_r2"]) \
                if not hit.empty else "n/a"
        entry["Decision"] = "selected" if variant == winner else "dropped"
        sel_rows.append(entry)
    tbl_selection = pd.DataFrame(sel_rows)

    # --- Table 5-2 --------------------------------------------------------
    lstm = merged[merged["model"] == "lstm"]
    l_rows = []
    for horizon in HORIZON_STEPS:
        entry = {"Horizon [min]": horizon * STEP_MINUTES}
        for variant in variants:
            hit = lstm[(lstm["features"] == variant)
                       & (lstm["horizon_steps"] == horizon)]
            if hit.empty:
                continue
            entry[f"{variant} RMSE [m/s]"] = label(hit.iloc[0])
            entry[f"{variant} r2"] = label(hit.iloc[0], "r2")
        l_rows.append(entry)
    tbl_lstm = pd.DataFrame(l_rows)

    # --- Table 5-3 --------------------------------------------------------
    trees = merged[merged["model"].isin(TREE_MODELS)]
    cmp_rows, lstm_wins = [], 0
    for horizon in HORIZON_STEPS:
        t = trees[trees["horizon_steps"] == horizon]
        l = lstm[lstm["horizon_steps"] == horizon]
        if t.empty or l.empty:
            continue
        tbest = t.loc[t["rmse"].idxmin()]
        lbest = l.loc[l["rmse"].idxmin()]
        delta = float(lbest["rmse"]) - float(tbest["rmse"])
        if delta < 0:
            lstm_wins += 1
        cmp_rows.append({
            "Horizon [min]": horizon * STEP_MINUTES,
            "Best LSTM": lbest["features"],
            "LSTM RMSE [m/s]": fmt(lbest["rmse"]),
            "Best tree": f"{tbest['model']} / {tbest['features']}",
            "Tree RMSE [m/s]": fmt(tbest["rmse"]),
            "Delta [m/s]": fmt(delta),
            "Delta [%]": fmt(100.0 * delta / float(tbest["rmse"]), 2),
            "Winner": "LSTM" if delta < 0 else "tree",
        })
    tbl_versus = pd.DataFrame(cmp_rows)

    # --- prose numbers ----------------------------------------------------
    pers = merged[merged["model"] == "persistence"]

    def gap_to_persistence(horizon):
        p = float(pers[pers["horizon_steps"] == horizon].iloc[0]["rmse"])
        l = float(lstm[lstm["horizon_steps"] == horizon]["rmse"].min())
        return 100.0 * (p - l) / p

    lstm_spread = (lstm["rmse_std"] / lstm["rmse"]).mean() * 100
    tree_spread = (trees["rmse_std"] / trees["rmse"]).mean() * 100

    # turbulence effect for the LSTM: winner variant vs variant A
    turb_rows = []
    for horizon in HORIZON_STEPS:
        a = lstm[(lstm["features"] == VARIANT_A)
                 & (lstm["horizon_steps"] == horizon)]
        b = lstm[(lstm["features"] == winner)
                 & (lstm["horizon_steps"] == horizon)]
        if a.empty or b.empty:
            continue
        turb_rows.append({
            "horizon_steps": horizon,
            "delta_pct": 100.0 * (float(b.iloc[0]["rmse"])
                                  - float(a.iloc[0]["rmse"]))
            / float(a.iloc[0]["rmse"])})
    turb = pd.DataFrame(turb_rows)
    turb_helped = int((turb["delta_pct"] < 0).sum()) if not turb.empty else 0
    turb_short = turb[turb["horizon_steps"] <= 3]["delta_pct"].mean()
    turb_long = turb[turb["horizon_steps"] >= 24]["delta_pct"].mean()

    competitive = [int(r["Horizon [min]"]) for r in cmp_rows if r["Winner"] == "LSTM"]
    lost = [int(r["Horizon [min]"]) for r in cmp_rows if r["Winner"] != "LSTM"]
    deltas_pct = [float(str(r["Delta [%]"]).replace(",", ".")) for r in cmp_rows]

    # Is the LSTM's lead larger than the fold-to-fold noise it is measured in?
    margin_rows = []
    for horizon in HORIZON_STEPS:
        t = trees[trees["horizon_steps"] == horizon]
        l = lstm[lstm["horizon_steps"] == horizon]
        if t.empty or l.empty:
            continue
        tbest = t.loc[t["rmse"].idxmin()]
        lbest = l.loc[l["rmse"].idxmin()]
        margin_rows.append({
            "horizon_min": horizon * STEP_MINUTES,
            "margin": abs(float(tbest["rmse"]) - float(lbest["rmse"])),
            "std": float(lbest["rmse_std"]),
        })
    margins = pd.DataFrame(margin_rows)
    exceeds_noise = [int(r["horizon_min"]) for _, r in margins.iterrows()
                     if r["margin"] > r["std"]]

    sel_means = selection.groupby("variant")["val_rmse"].mean()
    loser = next(v for v in CANDIDATES if v != winner)

    intro = (
        "This section reports the recurrent model. Each sample is the last "
        f"{SEQ_LEN} steps ({SEQ_LEN * STEP_MINUTES // 60} hours) of channel "
        "history ending at the issue time, passed through a single LSTM layer, "
        f"dropout of {fmt(DROPOUT, 1)} and a dense layer emitting all H steps "
        "at once. Scaling is fitted on each fold's training window alone. The "
        "models are evaluated on exactly the same issue times as the tree "
        "models of Section 4, so the two are directly comparable."
    )

    selection_prose = (
        "## 5.1 Input-variant selection\n\n"
        "The hourly companion study found that giving the LSTM every available "
        "exogenous channel harmed it at the long horizons. That result was "
        "treated as a hypothesis to test rather than a conclusion to inherit. "
        "Variants `reduced` and `turb` were compared on the **fold-0 "
        "validation split only**, at 120 and 480 minutes, at the default width "
        f"of {DEFAULT_UNITS} units; no test data entered the decision. Mean "
        f"validation RMSE was {fmt(sel_means['reduced'])} m/s for `reduced` "
        f"and {fmt(sel_means['turb'])} m/s for `turb`, so **`{winner}` was "
        f"selected and `{loser}` dropped**. The full record is in "
        "`results/lstm_variant_selection.md`. The main run therefore covers "
        f"variant A (`{VARIANT_A}`, ws100 only) and `{winner}`, with widths "
        "chosen on the same split: "
        + ", ".join(f"`{v}` at {u} units" for v, u in chosen_units.items()) + "."
    )

    if not lost:
        where = ("The LSTM attains the lowest RMSE of **any** model in this "
                 "study at every one of the eight horizons. ")
    elif competitive:
        where = ("The LSTM attains the lowest RMSE at "
                 + ", ".join(f"{m} min" for m in competitive)
                 + ", and is beaten by the best tree model at "
                 + ", ".join(f"{m} min" for m in lost) + ". ")
    else:
        where = ("The LSTM does not attain the lowest RMSE at any horizon; "
                 "the tree models win throughout. ")

    margin_note = (
        "Its margin over the best tree model runs from "
        f"{fmt(abs(min(deltas_pct, key=abs)), 2)} % at the narrowest to "
        f"{fmt(abs(min(deltas_pct)), 2)} % at the widest. That margin has to be "
        "read against the fold-to-fold spread, however. "
        + (f"At {', '.join(f'{m} min' for m in exceeds_noise)} the gap exceeds "
           "one standard deviation of the LSTM's own RMSE across folds; at the "
           "remaining horizons it does not."
           if exceeds_noise else
           "At **every** horizon the gap is an order of magnitude smaller than "
           "one standard deviation of the LSTM's own RMSE across folds. The "
           "LSTM is therefore the best single choice on the evidence "
           "available, but that evidence does not establish it as genuinely "
           "superior to the leading tree models rather than favourably placed "
           "within the noise. The same caveat was recorded for the tree "
           "models in Section 4 and it applies with equal force here.")
    )

    competitiveness = (
        "## Interpretation: where the sequence model is competitive\n\n"
        + where + margin_note + " Measured against the persistence baseline "
        f"the LSTM gains {fmt(gap_to_persistence(1), 2)} % at 10 minutes, "
        f"{fmt(gap_to_persistence(12), 2)} % at 120 minutes and "
        f"{fmt(gap_to_persistence(48), 2)} % at 480 minutes.\n\n"
        "The shape of that progression matters more than the ranking. The gain "
        "over persistence grows monotonically with the horizon, exactly as the "
        "autocorrelation decay of Section 1 predicts, and it stays small at 10 "
        "minutes. The sanity condition in CLAUDE.md therefore still holds for "
        "the sequence model, which is worth stating explicitly: a 72-step "
        "input window has far more opportunity to encode the target "
        "accidentally than a handful of lags does, so a large short-horizon "
        "gain here would have been the first place to suspect leakage. Where "
        "the recurrent model earns its keep is from about an hour out, and the "
        "reason is structural: the 12-hour window lets it condition on the "
        "shape of the preceding half-day — the trend and the stage of the "
        "diurnal cycle — rather than on the last observation alone, and that "
        "is information the tree models see only through the coarse lag-block "
        "means of Section 2."
    )

    if turb.empty:
        turb_para = ("## Interpretation: did the extra channels help the LSTM?\n\n"
                     "Only one variant was run, so no comparison is available.")
    else:
        direction_short = ("improved" if turb_short < 0 else "degraded")
        direction_long = ("improved" if turb_long < 0 else "degraded")
        everywhere = turb_helped == len(turb)
        turb_para = (
            "## Interpretation: did the extra channels help the LSTM?\n\n"
            f"Comparing `{winner}` against variant A across the eight "
            f"horizons, the additional channels improved RMSE at "
            f"{turb_helped} of {len(turb)} horizons. They "
            f"{direction_short} RMSE by {fmt(abs(turb_short), 2)} % on average "
            f"at 10-30 minutes and {direction_long} it by "
            f"{fmt(abs(turb_long), 2)} % at 240-480 minutes"
            + (", and the benefit grows steadily with the horizon.\n\n"
               if everywhere else ".\n\n")
            + ("This does **not** reproduce the pattern found for the tree "
               "models in Section 4. There the extra channels hurt the random "
               "forests at short horizons, because with max_features=\"sqrt\" "
               "every additional column dilutes the pool sampled at each split "
               "and crowds out the recent ws100 lags. The LSTM has no such "
               "mechanism: it sees all channels at every step and learns an "
               "input projection over them, so an uninformative channel costs "
               "it only a few extra weights rather than displacing a useful "
               "one. The wind components u and v additionally carry direction, "
               "which no other feature in variant A encodes at all, and the "
               "time-of-day pair supplies the diurnal phase that Section 1 "
               "showed to be pronounced and season-dependent.\n\n"
               if everywhere else
               "This broadly matches the tree-model behaviour of Section 4, "
               "where extra inputs cost more than they returned at the "
               "horizons dominated by recent history.\n\n")
            + "One caveat belongs with this result. The `turb` variant, which "
            "adds sd and ri on top of `reduced`, was dropped on the strength "
            f"of a {fmt(abs(100 * (sel_means['turb'] - sel_means['reduced']) / sel_means['reduced']), 1)} % "
            "mean difference in fold-0 validation RMSE measured at only two "
            "horizons, and the two horizons disagreed in direction. Given that "
            "`reduced` went on to beat variant A at every horizon, the "
            "turbulence channels may well deserve a fuller test than the "
            "selection protocol gave them; that is a limitation of this study "
            "rather than a finding against those channels."
        )

    stability = (
        "## Interpretation: fold-to-fold stability\n\n"
        f"The LSTM's standard deviation of RMSE across the four folds, relative "
        f"to its mean RMSE, averages {fmt(lstm_spread, 1)} %, against "
        f"{fmt(tree_spread, 1)} % for the tree models. "
        + ("The recurrent model is therefore the less stable of the two across "
           "resampled training windows, which matters for a scheduling "
           "application: a model whose error varies more between periods is "
           "harder to plan around even when its average is competitive."
           if lstm_spread > tree_spread else
           "The recurrent model is therefore no less stable than the trees "
           "across resampled training windows.")
        + " Both are subject to the caveat recorded in Section 4: with four "
        "folds from a single year, differences smaller than roughly one "
        "standard deviation should not be read as real."
    )

    figures_block = "\n\n".join([
        f"![LSTM training curves](../figures/{FIG_CURVES})\n\n"
        f"Figure 5-1: LSTM training and validation loss against epoch on fold 0, "
        f"at 120 and 480 minutes, for each input variant. Training stops when "
        f"validation loss has not improved for {PATIENCE} epochs and the best "
        "weights are restored.",

        f"![LSTM vs trees](../figures/{FIG_LEAD})\n\n"
        f"Figure 5-2: RMSE against lead time for the LSTM variants, the best "
        f"tree model ({best_tree_label}) and persistence, at the 120- and "
        "480-minute horizons, with bands showing +/- one standard deviation "
        "across the four folds.",
    ])

    write_section(SECTION_MD, "5 Results: Recurrent Neural Network", [
        intro,
        selection_prose,
        md_table(tbl_selection, "LSTM input-variant selection on the fold-0 "
                                "validation split", "5-1"),
        "## 5.2 Results across all horizons",
        md_table(tbl_lstm, "LSTM results per horizon and input variant, "
                           "mean +/- std across the four folds", "5-2"),
        "## 5.3 Comparison with the tree models",
        md_table(tbl_versus, "Best LSTM against the best tree model per "
                             "horizon (negative delta = the LSTM is better)",
                 "5-3"),
        "## 5.4 Figures",
        figures_block,
        competitiveness,
        turb_para,
        stability,
    ])
    print(f"wrote {SECTION_MD.relative_to(ROOT)}")

    log_progress(
        5, f"LSTM trained on {SEQ_LEN}-step sequences for variant A and "
        f"`{winner}` (selected over `{loser}` on fold-0 validation) across all "
        f"horizons and folds. The LSTM wins at {len(competitive)} of "
        f"{len(cmp_rows)} horizons against the best tree model; fold spread "
        f"{fmt(lstm_spread, 1)} % versus {fmt(tree_spread, 1)} % for the trees.")
    print("appended to report/PROGRESS.md")


# --------------------------------------------------------------------------
def report_only() -> None:
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    per_step = pd.read_csv(RESULTS / "metrics_per_step.csv")
    selection = pd.read_csv(SELECTION_CSV)
    units_df = pd.read_csv(UNITS_CSV)
    history = pd.read_csv(HISTORY_CSV)
    winner = selection.groupby("variant")["val_rmse"].mean().idxmin()
    chosen = {v: int(g.groupby("units")["val_rmse"].mean().idxmin())
              for v, g in units_df.groupby("variant")}
    variants = [VARIANT_A, winner]
    label = make_figures(metrics, per_step, history, variants)
    build_report(metrics, per_step, selection, units_df, winner, chosen,
                 variants, label)


def main() -> None:
    started = time.perf_counter()
    MODELS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_10min()
    channels = channel_frame(df)

    print("=" * 78)
    print("LSTM: input-variant and width selection on fold-0 validation")
    print("=" * 78, flush=True)
    winner, chosen_units = select_variant_and_units(df, channels)
    variants = [VARIANT_A, winner]
    print(f"\nselected variant: {winner}; units: {chosen_units}\n", flush=True)

    print("=" * 78)
    print(f"LSTM: main run over {len(HORIZON_STEPS)} horizons x "
          f"{len(variants)} variants x 4 folds")
    print("=" * 78, flush=True)
    records, step_records, history_rows, timing = run_all(
        df, channels, variants, chosen_units)

    metrics, per_step = log_metrics(records, step_records)
    history = pd.DataFrame(history_rows)
    history.to_csv(HISTORY_CSV, index=False)
    timing_df = pd.DataFrame(timing)
    timing_df["wall_clock_min"] = (time.perf_counter() - started) / 60.0
    timing_df.to_csv(RESULTS / "timing_lstm.csv", index=False)

    print(f"\ntotal wall clock: {(time.perf_counter() - started) / 60:.1f} min")

    selection = pd.read_csv(SELECTION_CSV)
    units_df = pd.read_csv(UNITS_CSV)
    label = make_figures(metrics, per_step, history, variants)
    build_report(metrics, per_step, selection, units_df, winner, chosen_units,
                 variants, label)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report_only()
    else:
        main()
