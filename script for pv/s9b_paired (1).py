"""Stage 9b: paired per-fold comparison of the key model pairs.

The seasonal-CV mean±std table's std (~16-25 kW) measures how much seasons
differ, not the uncertainty of a model comparison — it swamps the ~1-2 kW
between-model differences. The statistically correct comparison is paired:
both models of a pair see the identical fold (same training data, same test
quarter), so their per-fold difference cancels the seasonal effect.

For each horizon and metric (RMSE, MAE) this script computes the per-fold
delta (first model minus second; negative = first model better), the mean
and std of the delta, and the win count out of 4 folds, for:
  RF+XGB+LSTM vs RF+XGB   (does the LSTM add value?)
  RF+XGB+LSTM vs RF       (does the full hybrid beat the best single?)
  RF+XGB      vs RF       (does 2-tree stacking add value?)

Outputs: results/paired_comparison.csv and figures/fig9_paired_deltas.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24]
PAIRS = [("RF+XGB+LSTM", "RF+XGB"), ("RF+XGB+LSTM", "RF"), ("RF+XGB", "RF")]

SURFACE, INK, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "figure.dpi": 150,
})


def main() -> None:
    cv = pd.read_csv(ROOT / "results" / "seasonal_cv.csv")
    cv = cv[~cv["fold"].isin(["mean", "std"])].copy()
    cv["fold"] = cv["fold"].astype(int)
    m = cv.set_index(["model", "horizon", "fold"])

    rows = []
    for a, b in PAIRS:
        for h in HORIZONS:
            for metric in ["RMSE", "MAE"]:
                deltas = {f: m.loc[(a, h, f), metric] - m.loc[(b, h, f), metric]
                          for f in range(4)}
                s = pd.Series(deltas)
                rows.append({
                    "pair": f"{a} vs {b}", "horizon": h, "metric": metric,
                    **{f"delta_fold{f}": round(v, 3) for f, v in deltas.items()},
                    "mean_delta": s.mean(), "std_delta": s.std(),
                    "wins_of_4": int((s < 0).sum()),
                })
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "paired_comparison.csv", index=False)
    print(res.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (a, b) in zip(axes, PAIRS):
        sub = res[(res["pair"] == f"{a} vs {b}") & (res["metric"] == "RMSE")]
        for f in range(4):
            ax.scatter(range(len(HORIZONS)), sub[f"delta_fold{f}"], s=22,
                       color=MUTED, alpha=0.7, label="per-fold delta" if f == 0 else None)
        ax.plot(range(len(HORIZONS)), sub["mean_delta"], color=BLUE, lw=2.2,
                marker="o", ms=6, label="mean delta")
        ax.axhline(0, color=BASELINE, lw=1)
        ax.set_xticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
        ax.set_title(f"{a}\nvs {b}", color=INK, fontsize=10)
        ax.set_xlabel("Horizon (h)")
    axes[0].set_ylabel("RMSE delta (kW), negative = first model better")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Paired per-fold RMSE differences (seasonal effect cancelled)", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig9_paired_deltas.png")
    print("\nSaved results/paired_comparison.csv and figures/fig9_paired_deltas.png")


if __name__ == "__main__":
    main()
