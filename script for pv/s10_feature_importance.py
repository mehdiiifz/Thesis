"""Stage 10: which of the 22 features actually drive the forecast?

Fits the stage-4 Random Forest on each horizon's full training set and records
the impurity-based feature importances, so the report can state — with
evidence — which feature families matter at which horizon.

Outputs: results/feature_importance.csv and figures/fig10_feature_importance.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from s4_trees import make_models

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]

FAMILY = {  # feature -> family used for the grouped summary
    **{f"P_lag{l}": "Recent power lags" for l in [0, 1, 2, 3]},
    **{f"P_lag{l}": "Daily power lags" for l in [23, 24, 25, 48]},
    "P_samehour_yday_target": "Daily power lags",
    "P_roll3_mean": "Rolling statistics", "P_roll24_mean": "Rolling statistics",
    "P_roll24_max": "Rolling statistics",
    "Gi_lag0": "Recent weather", "Gi_lag1": "Recent weather", "Gi_lag24": "Recent weather",
    "T2m_lag0": "Recent weather", "WS10m_lag0": "Recent weather",
    "Hsun_target": "Sun geometry & calendar",
    **{c: "Sun geometry & calendar" for c in
       ["sin_hour_target", "cos_hour_target", "sin_doy_target", "cos_doy_target"]},
}

SURFACE, INK, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "figure.dpi": 150,
})


def main() -> None:
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    rows = []
    for h in HORIZONS:
        s = splits[h]
        X = pd.concat([s["X_train"], s["X_holdout"]])
        y = pd.concat([s["y_train"], s["y_holdout"]])
        rf = make_models()["RF"]
        rf.fit(X, y)
        for feat, imp in zip(X.columns, rf.feature_importances_):
            rows.append({"horizon": h, "feature": feat,
                         "family": FAMILY[feat], "importance": imp})
        print(f"h={h:>2} fitted; top: "
              f"{', '.join(pd.Series(rf.feature_importances_, index=X.columns).nlargest(3).index)}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "feature_importance.csv", index=False)

    grouped = res.groupby(["horizon", "family"])["importance"].sum().unstack()
    order = ["Recent power lags", "Daily power lags", "Rolling statistics",
             "Recent weather", "Sun geometry & calendar"]
    print("\nImportance by family (share):")
    print(grouped[order].round(3).to_string())

    # Sun geometry alone explains the day/night switch and the diurnal envelope,
    # so it dwarfs everything; panel (b) renormalises over the history-derived
    # features to show what actually distinguishes one day from another.
    hist = order[:-1]
    renorm = grouped[hist].div(grouped[hist].sum(axis=1), axis=0)
    print("\nAmong history-derived features only (renormalised):")
    print(renorm.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, data, cols, colors, title, ylab in [
        (axes[0], grouped[order], order, COLORS,
         "(a) All 22 features", "Share of Random-Forest importance"),
        (axes[1], renorm, hist, COLORS,
         "(b) History-derived features only (renormalised)", "Share within these families"),
    ]:
        bottom = pd.Series(0.0, index=data.index)
        x = range(len(data.index))
        for fam, c in zip(cols, colors):
            ax.bar(x, data[fam], bottom=bottom, color=c, width=0.72, label=fam)
            bottom += data[fam]
        ax.set_xticks(list(x), [f"{h} h" for h in data.index])
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")
    fig.suptitle("Where the forecast information comes from, by horizon", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig10_feature_importance.png")
    print("\nSaved results/feature_importance.csv and figures/fig10_feature_importance.png")


if __name__ == "__main__":
    main()
