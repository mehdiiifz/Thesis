"""Stage 12: print-quality versions of the four descriptive figures for the thesis.

Differences from the exploratory figures of stage 2: vector PDF output sized to
the 16 cm text width of the thesis, serif type matching the document, white
paper background, and annotation of the values the text quotes. Each figure is
written both as .pdf (for \\includegraphics) and .png (for previewing).

Outputs: figures/thesis/*.pdf and figures/thesis/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "thesis"

# validated categorical palette, fixed slot order; white paper for print
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, FAINT = "#111111", "#6b6a66", "#dcdbd5", "#c9d8ef"
WIDTH = 6.3  # inches = 16 cm text width

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved figures/thesis/{name}.pdf and .png")


def fig_sample_week(df):
    week = df.loc["2012-06-04":"2012-06-10", "P_kW"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.fill_between(week.index, week, color=BLUE, alpha=0.13, linewidth=0)
    ax.plot(week.index, week, color=BLUE, linewidth=1.4)

    peak = week.idxmax()
    ax.annotate(f"peak {week.max():.0f} kW",
                xy=(peak, week.max()), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
    days = pd.date_range(week.index[0].normalize(), week.index[-1].normalize(), freq="D")
    for d in days[1:]:
        ax.axvline(d, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xticks(days + pd.Timedelta(hours=12))
    ax.set_xticklabels([d.strftime("%a\n%d %b") for d in days])
    ax.set_xlim(week.index[0], week.index[-1])
    ax.set_ylim(0, 880)
    ax.set_ylabel("Power output (kW)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_sample_week_june2012")


def fig_diurnal(df):
    season = df.index.month.map(
        lambda m: "Winter" if m in (12, 1, 2) else
        "Spring" if m in (3, 4, 5) else
        "Summer" if m in (6, 7, 8) else "Autumn")
    prof = df["P_kW"].groupby([season, df.index.hour]).mean().unstack(0)
    order = [("Winter", BLUE), ("Spring", ORANGE),
             ("Summer", AQUA), ("Autumn", YELLOW)]

    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    for name, colour in order:
        ax.plot(prof.index, prof[name], color=colour, linewidth=1.6, label=name)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 780)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Mean power output (kW)")

    top = ax.secondary_xaxis("top")  # local time is UTC+3 at the site
    top.set_xticks(range(0, 24, 3))
    top.set_xticklabels([f"{(h + 3) % 24:02d}" for h in range(0, 24, 3)])
    top.set_xlabel("Local time (UTC+3)", labelpad=6)
    top.tick_params(colors=MUTED)
    top.spines["top"].set_color(MUTED)

    ax.legend(frameon=False, loc="upper right", handlelength=1.6, labelcolor=INK)
    save(fig, "fig_diurnal_by_season")


def fig_monthly(df):
    monthly = df["P_kW"].resample("MS").sum() / 1000.0
    tbl = pd.DataFrame({"month": monthly.index.month, "year": monthly.index.year,
                        "mwh": monthly.values}).pivot(index="month", columns="year", values="mwh")
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    ax.bar(x - 0.2, tbl[2012], width=0.38, color=BLUE,
           label=f"2012  (annual {tbl[2012].sum():,.0f} MWh)")
    ax.bar(x + 0.2, tbl[2013], width=0.38, color=ORANGE,
           label=f"2013  (annual {tbl[2013].sum():,.0f} MWh)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Energy production (MWh)")
    ax.set_ylim(0, 250)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncols=2, labelcolor=INK,
              handlelength=1.2, columnspacing=1.4)
    save(fig, "fig_monthly_energy")


def fig_autocorrelation(df):
    p = df["P_kW"]
    lags = np.arange(0, 73)
    acf = np.array([1.0 if l == 0 else p.autocorr(l) for l in lags])
    used = {0, 1, 2, 3, 23, 24, 25, 48}  # lags that became predictors

    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    colours = [BLUE if l in used else FAINT for l in lags]
    ax.bar(lags, acf, color=colours, width=0.75, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)

    for lag, dx, ha in [(1, 6, "left"), (24, 0, "center"), (48, 0, "center")]:
        ax.annotate(f"{acf[lag]:.2f}", xy=(lag, acf[lag]), xytext=(dx, 7),
                    textcoords="offset points", ha=ha, fontsize=7.5, color=INK)
    trough = 6 + int(np.argmin(acf[6:18]))  # first trough, the half-day anti-phase
    ax.annotate(f"{acf[trough]:.2f}", xy=(trough, acf[trough]), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top", fontsize=7.5, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=FAINT)]
    ax.legend(handles, ["lags used as predictors", "lags not used"],
              frameon=False, loc="lower right", handlelength=1.1, labelcolor=INK)
    ax.set_xticks(range(0, 73, 12))
    ax.set_xlim(-1, 73)
    ax.set_ylim(-0.82, 1.14)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation of $P$")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_autocorrelation")


def main():
    df = pd.read_pickle(ROOT / "data" / "clean.pkl")
    fig_sample_week(df)
    fig_diurnal(df)
    fig_monthly(df)
    fig_autocorrelation(df)


if __name__ == "__main__":
    main()
