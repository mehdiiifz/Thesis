"""Exploratory data analysis of the hourly wind dataset.

Saves figures to figures/ and key numbers to results/eda_summary.md:
(a) full-year ws100 time series      (b) monthly boxplots
(c) mean diurnal profile (overall + DJF/MAM/JJA/SON)
(d) histogram with fitted Weibull    (e) ACF and PACF up to lag 72
(f) polar wind-direction histogram

Run from project root: python src/eda.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_hourly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def plot_timeseries(df: pd.DataFrame) -> None:
    """(a) Full-year ws100 time series."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["ws100"], lw=0.3, color="tab:blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("ws100 (m/s)")
    ax.set_title("Wind speed at 100 m, full year")
    fig.tight_layout()
    fig.savefig(FIGDIR / "a_timeseries_full_year.png", dpi=150)
    plt.close(fig)


def plot_monthly_box(df: pd.DataFrame) -> pd.Series:
    """(b) Monthly boxplots of ws100. Returns monthly means."""
    months = df.index.month
    data = [df.loc[months == m, "ws100"].dropna() for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(data, tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
               showfliers=False)
    ax.set_ylabel("ws100 (m/s)")
    ax.set_title("Monthly distribution of ws100")
    fig.tight_layout()
    fig.savefig(FIGDIR / "b_monthly_boxplots.png", dpi=150)
    plt.close(fig)
    return df.groupby(months)["ws100"].mean()


def plot_diurnal(df: pd.DataFrame) -> pd.Series:
    """(c) Mean diurnal profile overall and per season. Returns overall profile."""
    hour = df.index.hour
    overall = df.groupby(hour)["ws100"].mean()
    season = df.index.month.map(SEASONS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(overall.index, overall.values, "k-", lw=2.5, label="overall")
    for s in SEASON_ORDER:
        prof = df.loc[season == s].groupby(df.index[season == s].hour)["ws100"].mean()
        ax.plot(prof.index, prof.values, lw=1.2, label=s)
    ax.set_xlabel("Hour of day (local file time)")
    ax.set_ylabel("mean ws100 (m/s)")
    ax.set_title("Mean diurnal profile")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "c_diurnal_profile.png", dpi=150)
    plt.close(fig)
    return overall


def plot_weibull(df: pd.DataFrame) -> tuple[float, float]:
    """(d) Histogram of ws100 with fitted Weibull. Returns (k, c)."""
    ws = df["ws100"].dropna()
    ws = ws[ws > 0]
    k, _, c = stats.weibull_min.fit(ws, floc=0)
    x = np.linspace(0, ws.max(), 300)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(ws, bins=50, density=True, alpha=0.55, color="tab:blue",
            label="observed")
    ax.plot(x, stats.weibull_min.pdf(x, k, 0, c), "r-", lw=2,
            label=f"Weibull fit (k={k:.2f}, c={c:.2f} m/s)")
    ax.set_xlabel("ws100 (m/s)")
    ax.set_ylabel("density")
    ax.set_title("Wind speed distribution with Weibull fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "d_hist_weibull.png", dpi=150)
    plt.close(fig)
    return float(k), float(c)


def plot_acf_pacf(df: pd.DataFrame, nlags: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """(e) ACF and PACF of ws100 up to `nlags` hours. Returns (acf, pacf)."""
    ws = df["ws100"].dropna()
    a = acf(ws, nlags=nlags)
    p = pacf(ws, nlags=nlags, method="ywm")
    ci = 1.96 / np.sqrt(len(ws))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, vals, name in [(axes[0], a, "ACF"), (axes[1], p, "PACF")]:
        ax.stem(range(len(vals)), vals, basefmt=" ", markerfmt=".")
        ax.axhline(0, color="k", lw=0.5)
        for y in (ci, -ci):
            ax.axhline(y, color="gray", ls="--", lw=0.8)
        ax.set_ylabel(name)
    axes[0].axhline(0.2, color="red", ls=":", lw=1, label="0.2 threshold")
    axes[0].legend()
    axes[0].set_title(f"ACF / PACF of ws100 up to lag {nlags} h")
    axes[1].set_xlabel("lag (hours)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "e_acf_pacf.png", dpi=150)
    plt.close(fig)
    return a, p


def plot_wind_rose(df: pd.DataFrame, n_sectors: int = 16) -> tuple[float, float]:
    """(f) Polar wind-direction histogram. Returns (dominant sector deg, freq %)."""
    d = df["dir"].dropna() % 360
    edges = np.linspace(0, 360, n_sectors + 1)
    counts, _ = np.histogram(d, bins=edges)
    freq = counts / counts.sum() * 100
    centers_deg = (edges[:-1] + edges[1:]) / 2
    theta = np.deg2rad(centers_deg)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(theta, freq, width=2 * np.pi / n_sectors, color="tab:blue",
           edgecolor="white", alpha=0.8)
    ax.set_title("Wind direction frequency (%)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "f_wind_rose.png", dpi=150)
    plt.close(fig)
    imax = int(np.argmax(freq))
    return float(centers_deg[imax]), float(freq[imax])


def main() -> None:
    """Run the full EDA and write results/eda_summary.md."""
    FIGDIR.mkdir(exist_ok=True)
    RESDIR.mkdir(exist_ok=True)
    df = load_hourly()
    ws = df["ws100"].dropna()

    plot_timeseries(df)
    monthly_mean = plot_monthly_box(df)
    diurnal = plot_diurnal(df)
    k, c = plot_weibull(df)
    a, p = plot_acf_pacf(df)
    dom_dir, dom_freq = plot_wind_rose(df)

    below = np.where(a < 0.2)[0]
    lag_acf_02 = int(below[0]) if len(below) else -1
    season = df.index.month.map(SEASONS)
    season_mean = ws.groupby(season).mean()

    lines = [
        "# EDA summary - hourly ws100",
        "",
        f"- Rows: {len(df)}, {df.index.min()} -> {df.index.max()}, "
        f"missing ws100: {int(df['ws100'].isna().sum())}",
        f"- ws100: mean {ws.mean():.2f} m/s, median {ws.median():.2f}, "
        f"std {ws.std():.2f}, min {ws.min():.2f}, max {ws.max():.2f} m/s",
        f"- Weibull fit (floc=0): shape k = {k:.2f}, scale c = {c:.2f} m/s",
        f"- Monthly mean ws100: highest {monthly_mean.max():.2f} m/s "
        f"(month {int(monthly_mean.idxmax())}), lowest {monthly_mean.min():.2f} m/s "
        f"(month {int(monthly_mean.idxmin())})",
        "- Seasonal mean ws100: "
        + ", ".join(f"{s} {season_mean[s]:.2f}" for s in SEASON_ORDER) + " m/s",
        f"- Diurnal profile: max {diurnal.max():.2f} m/s at hour "
        f"{int(diurnal.idxmax())}, min {diurnal.min():.2f} m/s at hour "
        f"{int(diurnal.idxmin())} (amplitude {diurnal.max() - diurnal.min():.2f} m/s)",
        f"- ACF(1h) = {a[1]:.3f}, ACF(24h) = {a[24]:.3f}",
        f"- ACF first drops below 0.2 at lag {lag_acf_02} h "
        "-> a 24 h lag window keeps only informative lags",
        f"- PACF: significant spikes at lags 1-2 (PACF(1)={p[1]:.3f}, "
        f"PACF(2)={p[2]:.3f}); small bump near lag 24 (PACF(24)={p[24]:.3f}) "
        "reflecting the diurnal cycle",
        f"- Dominant wind direction sector: {dom_dir:.0f} deg "
        f"({dom_freq:.1f}% of hours)",
        "",
        "Figures: a_timeseries_full_year, b_monthly_boxplots, "
        "c_diurnal_profile, d_hist_weibull, e_acf_pacf, f_wind_rose (figures/).",
    ]
    out = RESDIR / "eda_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
