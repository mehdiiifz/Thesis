"""Stage 2: exploratory data analysis of the cleaned PVGIS timeseries.

Reads data/clean.pkl (regenerate with scripts/s1_load_clean.py) and saves five
figures to figures/ plus the key statistics to results/stage2_eda_stats.txt.
All timestamps are UTC (local time at the site is UTC+3).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
STATS_TXT = ROOT / "results" / "stage2_eda_stats.txt"

# Validated palette (dataviz reference instance, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, fixed order
BLUE = SERIES[0]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#104281", "#f0efec", "#d03b3b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "figure.dpi": 150,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"saved figures/{name}")


def fig1_sample_week(df: pd.DataFrame) -> None:
    week = df.loc["2012-06-04":"2012-06-10", "P_kW"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week.index, week, color=BLUE, linewidth=2)
    ax.set_title("One week of PV power output — 4–10 June 2012 (UTC)", color=INK)
    ax.set_ylabel("Power (kW)")
    ax.set_ylim(bottom=0)
    save(fig, "fig1_sample_week_june2012.png")


def fig2_diurnal_by_season(df: pd.DataFrame) -> None:
    season = df.index.month.map(
        lambda m: "Winter (DJF)" if m in (12, 1, 2)
        else "Spring (MAM)" if m in (3, 4, 5)
        else "Summer (JJA)" if m in (6, 7, 8)
        else "Autumn (SON)"
    )
    profile = df["P_kW"].groupby([season, df.index.hour]).mean().unstack(0)
    order = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in zip(order, SERIES):
        ax.plot(profile.index, profile[name], color=color, linewidth=2, label=name)
    ax.set_title("Mean diurnal profile of PV power by season (hours in UTC)", color=INK)
    ax.set_xlabel("Hour of day (UTC; local time is UTC+3)")
    ax.set_ylabel("Mean power (kW)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=INK)
    save(fig, "fig2_diurnal_by_season.png")


def fig3_monthly_energy(df: pd.DataFrame) -> pd.Series:
    monthly_mwh = df["P_kW"].resample("MS").sum() / 1000.0  # hourly kW -> kWh -> MWh
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(monthly_mwh)), monthly_mwh, color=BLUE, width=0.8)
    ax.set_xticks(range(len(monthly_mwh)))
    ax.set_xticklabels([ts.strftime("%b\n%Y") for ts in monthly_mwh.index], fontsize=7)
    ax.set_title("Monthly energy production, 2012–2013", color=INK)
    ax.set_ylabel("Energy (MWh)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_monthly_energy.png")
    return monthly_mwh


def fig4_corr_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["P_kW", "G(i)", "H_sun", "T2m", "WS10m"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    ax.set_title("Correlation between power and weather variables", color=INK)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "fig4_corr_heatmap.png")
    return corr


def fig5_autocorrelation(df: pd.DataFrame) -> pd.Series:
    p = df["P_kW"]
    lags = range(0, 73)
    acf = pd.Series([p.autocorr(lag) if lag else 1.0 for lag in lags], index=list(lags))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(acf.index, acf, color=BLUE, width=0.6)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_title("Autocorrelation of PV power, lags 0–72 hours", color=INK)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation with lagged self")
    ax.set_xticks(range(0, 73, 12))
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_autocorr_p_kw.png")
    return acf


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = pd.read_pickle(ROOT / "data" / "clean.pkl")

    fig1_sample_week(df)
    fig2_diurnal_by_season(df)
    monthly = fig3_monthly_energy(df)
    corr = fig4_corr_heatmap(df)
    acf = fig5_autocorrelation(df)

    lines = [
        "Correlation matrix:",
        corr.round(4).to_string(),
        f"\ncorr(P_kW, concurrent G(i)): {corr.loc['P_kW', 'G(i)']:.4f}",
        f"\nMonthly energy (MWh): min {monthly.min():.1f} ({monthly.idxmin():%b %Y}), "
        f"max {monthly.max():.1f} ({monthly.idxmax():%b %Y})",
        f"Annual energy: 2012 {monthly['2012'].sum():.0f} MWh, 2013 {monthly['2013'].sum():.0f} MWh",
        "\nAutocorrelation of P_kW at key lags:",
        acf.loc[[1, 2, 3, 6, 12, 24, 48, 72]].round(4).to_string(),
    ]
    report = "\n".join(lines)
    print("\n" + report)
    STATS_TXT.write_text(report + "\n")
    print(f"\nSaved {STATS_TXT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
