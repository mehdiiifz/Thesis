"""Stage 13: print-quality result figures for Chapter 4, in the same style as
the descriptive figures of stage 12 (vector PDF at the 16 cm text width, serif
type, white paper).

Outputs: figures/thesis/fig_r*.pdf and .png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "thesis"
RES = ROOT / "results"
H = [1, 2, 3, 4, 6, 8, 12, 24]

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED, GRID, FAINT = "#111111", "#6b6a66", "#dcdbd5", "#c9d8ef"
WIDTH = 6.3

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# distinct colour AND marker per model, so the figure survives greyscale printing
SERIES = [
    ("DT",          "#eda100", "-",  "v", "Decision tree"),
    ("RF",          "#2a78d6", "-",  "o", "Random forest"),
    ("ET",          "#1baf7a", "-",  "s", "Extra trees"),
    ("XGB",         "#eb6834", "-",  "^", "XGBoost"),
    ("LSTM-v2",     "#4a3aa7", "-",  "D", "LSTM"),
    ("RF+XGB",      "#e34948", "--", "x", "RF + XGB"),
    ("RF+XGB+LSTM", "#111111", "-",  "o", "RF + XGB + LSTM"),
]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved figures/thesis/{name}.pdf and .png")


def logx(ax):
    ax.set_xscale("log")
    ax.set_xticks(H, [str(h) for h in H])


def fig_rmse_two_protocols():
    ss = pd.read_csv(RES / "final_comparison.csv")
    ss = ss[ss.headline].set_index(["model", "horizon"])["RMSE"]
    cv = pd.read_csv(RES / "seasonal_cv.csv")
    cvm = cv[cv.fold == "mean"].set_index(["model", "horizon"])["RMSE"]

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.1), sharex=True)
    for ax, data, title in [
            (axes[0], ss, "(a) Fixed chronological split"),
            (axes[1], cvm, "(b) Seasonal cross-validation, mean of four folds")]:
        avail = set(data.index.get_level_values(0))
        ax.plot(H, [data[("PERS24", h)] for h in H], color=MUTED, ls=(0, (5, 3)),
                lw=1.5, label="PERS24 reference", zorder=1)
        for name, colour, style, marker, label in SERIES:
            if name not in avail:
                continue
            headline = name == "RF+XGB+LSTM"
            ax.plot(H, [data[(name, h)] for h in H], color=colour, ls=style,
                    lw=2.4 if headline else 1.4, marker=marker,
                    ms=5.5 if headline else 4.4, mew=1.2,
                    markerfacecolor=colour if headline else "white",
                    label=label, zorder=4 if headline else 2)
        logx(ax)
        ax.set_title(title, color=INK, loc="left")
        ax.set_ylabel("Test RMSE (kW)")
        ax.set_ylim(45, 108)
    axes[1].set_xlabel("Forecast horizon (hours)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), labelcolor=INK, handlelength=2.6,
               columnspacing=1.6, fontsize=8)
    save(fig, "fig_r1_rmse_two_protocols")


def _rmse_panel(data, name):
    """One standalone RMSE-by-horizon figure for a single evaluation protocol.

    No in-figure title: the LaTeX caption identifies the protocol.
    """
    avail = set(data.index.get_level_values(0))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))
    ax.plot(H, [data[("PERS24", h)] for h in H], color=MUTED, ls=(0, (5, 3)),
            lw=1.5, label="PERS24 reference", zorder=1)
    for model, colour, style, marker, label in SERIES:
        if model not in avail:
            continue
        headline = model == "RF+XGB+LSTM"
        ax.plot(H, [data[(model, h)] for h in H], color=colour, ls=style,
                lw=2.4 if headline else 1.4, marker=marker,
                ms=5.5 if headline else 4.4, mew=1.2,
                markerfacecolor=colour if headline else "white",
                label=label, zorder=4 if headline else 2)
    logx(ax)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_ylim(45, 108)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              labelcolor=INK, handlelength=2.6, columnspacing=1.5, fontsize=8)
    save(fig, name)


def fig_rmse_separate():
    ss = pd.read_csv(RES / "final_comparison.csv")
    ss = ss[ss.headline].set_index(["model", "horizon"])["RMSE"]
    cv = pd.read_csv(RES / "seasonal_cv.csv")
    cvm = cv[cv.fold == "mean"].set_index(["model", "horizon"])["RMSE"]
    _rmse_panel(ss, "fig_r1a_rmse_fixed_split")
    _rmse_panel(cvm, "fig_r1b_rmse_seasonal_cv")


def fig_error_correlation():
    corr = pd.read_csv(RES / "error_correlations.csv")
    order = ["RF", "ET", "XGB", "LSTM"]
    cmap = LinearSegmentedColormap.from_list("b", ["#eef4fc", "#2a78d6", "#104281"])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    for ax, h in zip(axes, [1, 24]):
        c = corr[corr.horizon == h].set_index("model").loc[order, order]
        im = ax.imshow(c, cmap=cmap, vmin=0.7, vmax=1.0)
        ax.set_xticks(range(4), order)
        ax.set_yticks(range(4), order)
        for i in range(4):
            for j in range(4):
                v = c.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.92 else INK)
        ax.set_xlabel(f"({'a' if h == 1 else 'b'}) horizon {h} h", color=INK, labelpad=8)
        ax.grid(visible=False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_r2_error_correlation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/thesis/fig_r2_error_correlation.pdf and .png")


def fig_lstm_weight():
    hyb = pd.read_csv(RES / "hybrid.csv").set_index(["model", "horizon"])
    cvw = pd.read_csv(RES / "seasonal_cv_weights.csv")
    cvw = cvw[cvw.model == "RF+XGB+LSTM"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.plot(H, [hyb.loc[("RF+XGB+LSTM", h), "w_LSTM"] for h in H], color=BLUE, lw=1.8,
            marker="o", ms=4.5, label="fixed split")
    mu = cvw.groupby("horizon")["w_LSTM"].mean()
    sd = cvw.groupby("horizon")["w_LSTM"].std()
    ax.errorbar(H, mu[H], yerr=sd[H], color=ORANGE, lw=1.5, marker="s", ms=4,
                capsize=2.5, elinewidth=0.9, label="seasonal CV, mean $\\pm$ sd")
    logx(ax)
    ax.set_ylabel("Weight of the LSTM in the blend")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK, handlelength=1.6)
    save(fig, "fig_r3_lstm_weight")


def fig_paired():
    pc = pd.read_csv(RES / "paired_comparison.csv")
    pc = pc[pc.metric == "RMSE"]
    pairs = [("RF+XGB+LSTM vs RF+XGB", "(a) hybrid $-$ two-tree stack"),
             ("RF+XGB+LSTM vs RF", "(b) hybrid $-$ Random Forest"),
             ("RF+XGB vs RF", "(c) two-tree stack $-$ Random Forest")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), sharey=True)
    for ax, (pair, title) in zip(axes, pairs):
        sub = pc[pc.pair == pair].set_index("horizon")
        for f in range(4):
            ax.scatter(range(len(H)), sub.loc[H, f"delta_fold{f}"], s=13, color=MUTED,
                       alpha=0.75, zorder=2, label="individual fold" if f == 0 else None)
        ax.plot(range(len(H)), sub.loc[H, "mean_delta"], color=BLUE, lw=1.7, marker="o",
                ms=4, zorder=3, label="mean")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(H)), [str(h) for h in H])
        ax.set_title(title, color=INK, fontsize=8.5)
        ax.set_xlabel("Horizon (h)")
    axes[0].set_ylabel("$\\Delta$RMSE (kW)")
    axes[0].legend(frameon=False, loc="lower left", labelcolor=INK, handlelength=1.2)
    save(fig, "fig_r4_paired_differences")


def fig_importance():
    fi = pd.read_csv(RES / "feature_importance.csv")
    g = fi.groupby(["horizon", "family"])["importance"].sum().unstack()
    order = ["Recent power lags", "Daily power lags", "Rolling statistics",
             "Recent weather", "Sun geometry & calendar"]
    cols = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]
    hist = order[:-1]
    ren = g[hist].div(g[hist].sum(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for ax, data, cs, title in [(axes[0], g[order], order, "(a) all predictors"),
                                (axes[1], ren, hist, "(b) history-derived only")]:
        bottom = pd.Series(0.0, index=data.index)
        x = range(len(data.index))
        for fam, c in zip(cs, cols):
            ax.bar(x, data[fam], bottom=bottom, color=c, width=0.7,
                   label=fam.replace(" & ", " and "))
            bottom += data[fam]
        ax.set_xticks(list(x), [str(h) for h in data.index])
        ax.set_xlabel("Horizon (h)")
        ax.set_ylim(0, 1)
        ax.set_title(title, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of importance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.16), labelcolor=INK, handlelength=1.1,
               fontsize=7.5, columnspacing=1.4)
    save(fig, "fig_r5_feature_importance")


def _hybrid_pred(h):
    """Rebuild the RF+XGB+LSTM test forecasts for one horizon (stage-7 procedure)."""
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    tree = pd.read_pickle(ROOT / "data" / "tree_preds.pkl")
    lstm = pd.read_pickle(ROOT / "data" / "lstm_preds.pkl")
    A = np.column_stack([tree["RF"][h]["holdout"], tree["XGB"][h]["holdout"], lstm[h]["holdout"]])
    B = np.column_stack([tree["RF"][h]["test"], tree["XGB"][h]["test"], lstm[h]["test"]])
    meta = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(A, splits[h]["y_holdout"])
    p = np.clip(meta.predict(B), 0, 1000)
    p[(splits[h]["X_test"]["Hsun_target"] <= 0).to_numpy()] = 0
    return pd.Series(p, index=splits[h]["y_test"].index), splits[h]["y_test"]


def fig_scatter(h, name):
    """Standalone forecast-against-measurement scatter for one horizon."""
    pred, act = _hybrid_pred(h)
    fig, ax = plt.subplots(figsize=(4.3, 4.3))
    ax.scatter(act, pred, s=3.5, alpha=0.13, color=BLUE, edgecolors="none")
    ax.plot([0, 850], [0, 850], color=MUTED, lw=1.0, ls="--", label="ideal forecast")
    ax.set_xlabel("Measured power (kW)")
    ax.set_ylabel("Forecast power (kW)")
    ax.set_xlim(-20, 870)
    ax.set_ylim(-20, 870)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK, handlelength=2.0)
    save(fig, name)


def fig_two_weeks(name="fig_r6c_two_weeks"):
    """Two weeks of measurement against the one-hour forecast.

    The forecast lies almost on top of the measurement, so the measurement is
    drawn as a filled band and the forecast as a line over it.
    """
    pred, act = _hybrid_pred(1)
    w = slice("2013-09-01", "2013-09-14")
    a, f = act.loc[w], pred.loc[w]

    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    ax.fill_between(a.index, a, color=BLUE, alpha=0.20, linewidth=0, label="measured")
    ax.plot(a.index, a, color=BLUE, lw=0.9, alpha=0.65)
    ax.plot(f.index, f, color=INK, lw=1.1, label="forecast, 1 h ahead")
    ax.set_ylabel("Power (kW)")
    ax.set_ylim(0, 940)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK, handlelength=2.0, ncols=2)
    ax.grid(axis="x", visible=False)

    # mark the day on which the sky was obstructed, where the errors concentrate
    cloudy = pd.Timestamp((f - a).abs().idxmax()).normalize()
    ax.axvspan(cloudy, cloudy + pd.Timedelta(days=1), color=MUTED, alpha=0.10,
               linewidth=0, zorder=0)
    ax.annotate(f"cloudy day, {cloudy:%d %B}", xy=(cloudy + pd.Timedelta(hours=12), 830),
                ha="center", fontsize=7.5, color=INK)

    ticks = pd.date_range("2013-09-01", "2013-09-15", freq="2D", tz="UTC")
    ax.set_xticks(ticks)
    ax.set_xticklabels([t.strftime("%d %b") for t in ticks])
    ax.set_xlim(a.index[0], a.index[-1])
    save(fig, name)


def fig_diagnostics_separate():
    fig_scatter(1, "fig_r6a_scatter_1h")
    fig_scatter(24, "fig_r6b_scatter_24h")
    fig_two_weeks()


def fig_best_per_horizon(name="fig_r7_best_per_horizon"):
    """Lowest test RMSE at each horizon, labelled with the model that achieved it.

    No in-figure title: the LaTeX caption carries the description.
    """
    ss = pd.read_csv(RES / "final_comparison.csv")
    ss = ss[ss.headline & ~ss.model.isin(["PERS", "PERS24"])]
    best = ss.loc[ss.groupby("horizon")["RMSE"].idxmin()].set_index("horizon").loc[H]
    pretty = {"RF+XGB+LSTM": "RF + XGB + LSTM", "ET": "Extra trees", "RF": "Random forest",
              "XGB": "XGBoost", "RF+XGB": "RF + XGB", "ET+XGB": "ET + XGB",
              "ET+XGB+LSTM": "ET + XGB + LSTM", "LSTM-v2": "LSTM", "DT": "Decision tree"}
    colour = {"RF+XGB+LSTM": "#2a78d6", "ET": "#1baf7a"}

    x = np.arange(len(H))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    ax.bar(x, best["RMSE"], width=0.66,
           color=[colour.get(m, MUTED) for m in best["model"]])
    for xi, rmse in enumerate(best["RMSE"]):
        ax.annotate(f"{rmse:.1f}", xy=(xi, rmse), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[m])
               for m in best["model"].unique()]
    ax.legend(handles, [pretty.get(m, m) for m in best["model"].unique()],
              frameon=False, loc="upper left", labelcolor=INK, handlelength=1.2,
              ncols=2, fontsize=8)
    ax.set_xticks(x, [f"{h} h" for h in H])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_ylim(0, 82)
    ax.grid(axis="x", visible=False)
    save(fig, name)


def fig_diagnostics():
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    tree = pd.read_pickle(ROOT / "data" / "tree_preds.pkl")
    lstm = pd.read_pickle(ROOT / "data" / "lstm_preds.pkl")

    def hybrid(h):
        A = np.column_stack([tree["RF"][h]["holdout"], tree["XGB"][h]["holdout"], lstm[h]["holdout"]])
        B = np.column_stack([tree["RF"][h]["test"], tree["XGB"][h]["test"], lstm[h]["test"]])
        meta = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(A, splits[h]["y_holdout"])
        p = np.clip(meta.predict(B), 0, 1000)
        p[(splits[h]["X_test"]["Hsun_target"] <= 0).to_numpy()] = 0
        return pd.Series(p, index=splits[h]["y_test"].index), splits[h]["y_test"]

    fig = plt.figure(figsize=(WIDTH, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.82], hspace=0.45, wspace=0.28)
    for col, h in enumerate([1, 24]):
        pred, act = hybrid(h)
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(act, pred, s=2.5, alpha=0.12, color=BLUE, edgecolors="none")
        ax.plot([0, 850], [0, 850], color=MUTED, lw=0.9, ls="--")
        ax.set_title(f"({'a' if h == 1 else 'b'}) horizon {h} h", color=INK)
        ax.set_xlabel("Measured power (kW)")
        if col == 0:
            ax.set_ylabel("Forecast power (kW)")
        ax.set_aspect("equal")
        ax.set_xlim(-20, 870)
        ax.set_ylim(-20, 870)
    pred, act = hybrid(1)
    ax = fig.add_subplot(gs[1, :])
    w = slice("2013-09-01", "2013-09-14")
    ax.plot(act.loc[w].index, act.loc[w], color=INK, lw=1.1, label="measured")
    ax.plot(pred.loc[w].index, pred.loc[w], color=BLUE, lw=1.1, ls="--", label="forecast, 1 h ahead")
    ax.set_title("(c) two weeks of the test period, 1 to 14 September 2013", color=INK)
    ax.set_ylabel("Power (kW)")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK, handlelength=1.8)
    ax.grid(axis="x", visible=False)
    ticks = pd.date_range("2013-09-01", "2013-09-15", freq="2D", tz="UTC")
    ax.set_xticks(ticks)
    ax.set_xticklabels([t.strftime("%d %b") for t in ticks])
    ax.set_xlim(act.loc[w].index[0], act.loc[w].index[-1])
    save(fig, "fig_r6_diagnostics")


def main():
    fig_rmse_two_protocols()
    fig_rmse_separate()
    fig_error_correlation()
    fig_lstm_weight()
    fig_paired()
    fig_importance()
    fig_diagnostics()
    fig_diagnostics_separate()
    fig_best_per_horizon()


if __name__ == "__main__":
    main()
