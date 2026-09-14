"""Exploratory analysis of the native 10-minute wind data set (Study 2, step 2).

Produces every figure in figures/ and every number in results/eda_stats.csv,
then writes report/01_data_and_eda.md from that CSV via report_utils.

The point of this step is to establish, before any model is trained, how much
structure the 10-minute series actually carries:

- how fast the autocorrelation decays, which bounds the accuracy any model can
  reach at each of the eight forecast horizons;
- how large a typical 10-minute change is, which explains why persistence is
  such a strong baseline at short lead times;
- whether the turbulence and stability channels (sd, ri) carry information
  about the size of the next change -- the physical justification for the
  "full_turb" feature set.

Run from the project root:

    python src/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as gamma_fn
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_data import load_10min, describe_dataset, TARGET, STEP_MINUTES  # noqa: E402
from report_utils import (  # noqa: E402
    HORIZON_MIN, HORIZON_STEPS, FIGURES, REPORT, RESULTS, fmt, fmt_int,
    log_progress, md_table, write_section,
)

SEED = 42
MAX_LAG = 300                 # steps = 3000 minutes = 50 hours
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES        # 144
ZOOM_DAYS = 3
MIN_WS_FOR_TI = 1.0           # TI = sd/ws is not meaningful at near-calm wind
N_BINS = 10                   # decile bins for the turbulence relations

EDA_STATS_CSV = RESULTS / "eda_stats.csv"
SECTION_MD = REPORT / "01_data_and_eda.md"

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

FIG_TIMESERIES = "fig_01_ws100_timeseries.png"
FIG_ACF = "fig_02_acf_pacf.png"
FIG_DIURNAL = "fig_03_diurnal_profile.png"
FIG_WEIBULL = "fig_04_weibull_fit.png"
FIG_CHANGE = "fig_05_step_change.png"
FIG_TURB = "fig_06_turbulence_vs_change.png"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# --------------------------------------------------------------------------
# results collector
# --------------------------------------------------------------------------
class Stats:
    """Collects every EDA number into one tidy long table."""

    COLUMNS = ["group", "key", "horizon_steps", "horizon_min",
               "bin_center", "n", "value", "text"]

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, group: str, key, value=None, *, text=None, n=None,
            horizon_steps=None, horizon_min=None, bin_center=None) -> None:
        self.rows.append({
            "group": group, "key": str(key), "horizon_steps": horizon_steps,
            "horizon_min": horizon_min, "bin_center": bin_center, "n": n,
            "value": None if value is None else float(value), "text": text,
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLUMNS)


def pick(df: pd.DataFrame, group: str, key: str, column: str = "value"):
    """Single value from the tidy stats table -- the report reads only via this."""
    hit = df[(df["group"] == group) & (df["key"] == str(key))]
    if hit.empty:
        raise KeyError(f"no row for group={group!r} key={key!r} in {EDA_STATS_CSV.name}")
    return hit.iloc[0][column]


# --------------------------------------------------------------------------
# (a) time series
# --------------------------------------------------------------------------
def plot_timeseries(df: pd.DataFrame, st: Stats) -> None:
    ws = df[TARGET]
    window = ZOOM_DAYS * STEPS_PER_DAY

    # Deterministic zoom choice: the most variable 3-day window in the year.
    roll_std = ws.rolling(window).std()
    # nanargmax: the first `window - 1` rolling values are NaN.
    end_pos = int(np.nanargmax(roll_std.to_numpy()))
    start_pos = end_pos - window + 1
    zoom = ws.iloc[start_pos:end_pos + 1]

    st.add("zoom_window", "start", text=str(zoom.index.min()))
    st.add("zoom_window", "end", text=str(zoom.index.max()))
    st.add("zoom_window", "days", ZOOM_DAYS)
    st.add("zoom_window", "std", float(zoom.std()))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    axes[0].plot(ws.index, ws.to_numpy(), lw=0.25, color="#1f4e79")
    axes[0].axhline(ws.mean(), color="#c00000", lw=1.0, ls="--",
                    label=f"mean = {ws.mean():.2f} m/s")
    axes[0].set_title(f"ws100 at 10-minute resolution, full record "
                      f"({ws.index.min():%d.%m.%Y} - {ws.index.max():%d.%m.%Y})")
    axes[0].set_ylabel("wind speed [m/s]")
    axes[0].legend(loc="upper right")

    axes[1].plot(zoom.index, zoom.to_numpy(), lw=0.9, color="#1f4e79",
                 marker=".", ms=2)
    axes[1].set_title(f"Zoom: most variable {ZOOM_DAYS}-day window "
                      f"({zoom.index.min():%d.%m.%Y %H:%M} - "
                      f"{zoom.index.max():%d.%m.%Y %H:%M}), 10-minute detail")
    axes[1].set_ylabel("wind speed [m/s]")
    axes[1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_TIMESERIES)
    plt.close(fig)


# --------------------------------------------------------------------------
# (b) ACF / PACF
# --------------------------------------------------------------------------
def plot_acf_pacf(df: pd.DataFrame, st: Stats) -> None:
    ws = df[TARGET].to_numpy()
    acf_vals = acf(ws, nlags=MAX_LAG, fft=True)
    pacf_vals = pacf(ws, nlags=MAX_LAG, method="ywm")

    for lag in range(MAX_LAG + 1):
        st.add("acf_curve", lag, acf_vals[lag], horizon_steps=lag,
               horizon_min=lag * STEP_MINUTES)
        st.add("pacf_curve", lag, pacf_vals[lag], horizon_steps=lag,
               horizon_min=lag * STEP_MINUTES)

    # ACF exactly at the eight forecast horizons.
    for steps, minutes in zip(HORIZON_STEPS, HORIZON_MIN):
        st.add("acf_horizon", f"h{steps}", acf_vals[steps],
               horizon_steps=steps, horizon_min=minutes)

    conf = 1.96 / np.sqrt(len(ws))
    st.add("acf_meta", "conf95", conf)
    below = np.where(acf_vals < conf)[0]
    st.add("acf_meta", "first_lag_below_conf",
           int(below[0]) if below.size else np.nan)

    lags = np.arange(MAX_LAG + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, vals, name in ((axes[0], acf_vals, "ACF"), (axes[1], pacf_vals, "PACF")):
        ax.bar(lags, vals, width=0.9, color="#1f4e79")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(conf, color="#c00000", lw=0.8, ls=":")
        ax.axhline(-conf, color="#c00000", lw=0.8, ls=":")
        ax.set_ylabel(name)
    for steps in HORIZON_STEPS:
        for ax in axes:
            ax.axvline(steps, color="#7f7f7f", lw=0.6, ls="--", alpha=0.8)
    axes[0].set_title(f"Autocorrelation of ws100 up to {MAX_LAG} steps "
                      f"({MAX_LAG * STEP_MINUTES} minutes); dashed lines mark "
                      "the eight forecast horizons")
    axes[1].set_xlabel("lag [steps of 10 min]")
    secax = axes[0].secondary_xaxis(
        "top", functions=(lambda x: x * STEP_MINUTES, lambda x: x / STEP_MINUTES))
    secax.set_xlabel("lag [minutes]")
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_ACF)
    plt.close(fig)


# --------------------------------------------------------------------------
# (c) diurnal profile
# --------------------------------------------------------------------------
def plot_diurnal(df: pd.DataFrame, st: Stats) -> None:
    minute_of_day = df.index.hour * 60 + df.index.minute
    season = pd.Series([SEASONS[m] for m in df.index.month], index=df.index)

    overall = df[TARGET].groupby(minute_of_day).mean()
    for minute, value in overall.items():
        st.add("diurnal_overall", int(minute), value)
    st.add("diurnal_meta", "n_points", len(overall))
    st.add("diurnal_meta", "amplitude", float(overall.max() - overall.min()))
    st.add("diurnal_meta", "peak_minute", float(overall.idxmax()))
    st.add("diurnal_meta", "trough_minute", float(overall.idxmin()))

    fig, ax = plt.subplots(figsize=(10, 5))
    hours = overall.index.to_numpy() / 60.0
    ax.plot(hours, overall.to_numpy(), lw=2.2, color="black", label="All seasons")

    colors = {"Winter": "#1f4e79", "Spring": "#2e8b57",
              "Summer": "#c00000", "Autumn": "#d78500"}
    for name in SEASON_ORDER:
        mask = (season == name).to_numpy()
        if not mask.any():
            continue
        prof = df.loc[mask, TARGET].groupby(minute_of_day[mask]).mean()
        for minute, value in prof.items():
            st.add(f"diurnal_{name.lower()}", int(minute), value)
        st.add("diurnal_season_meta", f"{name.lower()}_amplitude",
               float(prof.max() - prof.min()), n=int(mask.sum()))
        st.add("diurnal_season_meta", f"{name.lower()}_mean",
               float(df.loc[mask, TARGET].mean()), n=int(mask.sum()))
        ax.plot(prof.index.to_numpy() / 60.0, prof.to_numpy(), lw=1.3,
                color=colors[name], label=name)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("time of day [h]")
    ax.set_ylabel("mean wind speed [m/s]")
    ax.set_title(f"Mean diurnal profile of ws100 at 10-minute resolution "
                 f"({len(overall)} points per day)")
    ax.legend(ncol=5, loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_DIURNAL)
    plt.close(fig)


# --------------------------------------------------------------------------
# (d) Weibull fit
# --------------------------------------------------------------------------
def plot_weibull(df: pd.DataFrame, st: Stats) -> None:
    ws = df[TARGET].to_numpy()
    positive = ws[ws > 0]
    k, loc, c = stats.weibull_min.fit(positive, floc=0)

    st.add("weibull", "k", k)
    st.add("weibull", "c", c)
    st.add("weibull", "loc", loc, n=len(positive))
    # Mean implied by the fit, c * Gamma(1 + 1/k), for comparison with the
    # empirical mean. Note gamma_fn is the gamma FUNCTION, not the gamma
    # distribution -- confusing the two inflates the fitted mean badly.
    st.add("weibull", "mean_fitted", c * float(gamma_fn(1.0 + 1.0 / k)))
    st.add("weibull", "mean_empirical", float(ws.mean()))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ws, bins=60, density=True, color="#9dc3e6",
            edgecolor="white", label="observed")
    grid = np.linspace(0, ws.max(), 400)
    ax.plot(grid, stats.weibull_min.pdf(grid, k, loc, c), lw=2, color="#c00000",
            label=f"Weibull fit: k = {k:.3f}, c = {c:.3f} m/s")
    ax.set_xlabel("wind speed [m/s]")
    ax.set_ylabel("density")
    ax.set_title("Distribution of ws100 with fitted Weibull density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_WEIBULL)
    plt.close(fig)


# --------------------------------------------------------------------------
# (e) step-to-step change
# --------------------------------------------------------------------------
def plot_step_change(df: pd.DataFrame, st: Stats) -> pd.Series:
    change = df[TARGET].diff().abs().dropna()

    st.add("change", "median", float(change.median()), n=len(change))
    st.add("change", "mean", float(change.mean()), n=len(change))
    st.add("change", "p90", float(change.quantile(0.90)))
    st.add("change", "p95", float(change.quantile(0.95)))
    st.add("change", "p99", float(change.quantile(0.99)))
    st.add("change", "max", float(change.max()))
    st.add("change", "std", float(change.std()))
    st.add("change", "share_below_0p5",
           float((change < 0.5).mean()))
    st.add("change", "relative_to_mean_ws",
           float(change.median() / df[TARGET].mean()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(change, bins=80, color="#9dc3e6", edgecolor="white")
    for q, color, label in ((change.median(), "#c00000", "median"),
                            (change.quantile(0.95), "#d78500", "95th pct")):
        axes[0].axvline(q, color=color, lw=1.6, ls="--",
                        label=f"{label} = {q:.3f} m/s")
    axes[0].set_xlabel("|ws(t) - ws(t-1)| [m/s]")
    axes[0].set_ylabel("count")
    axes[0].set_title("Distribution of the 10-minute change")
    axes[0].legend()

    ordered = np.sort(change.to_numpy())
    cdf = np.arange(1, len(ordered) + 1) / len(ordered)
    axes[1].plot(ordered, cdf, lw=1.6, color="#1f4e79")
    axes[1].axhline(0.5, color="#c00000", lw=0.9, ls=":")
    axes[1].axhline(0.95, color="#d78500", lw=0.9, ls=":")
    axes[1].set_xlim(0, float(change.quantile(0.999)))
    axes[1].set_xlabel("|ws(t) - ws(t-1)| [m/s]")
    axes[1].set_ylabel("cumulative share")
    axes[1].set_title("Empirical CDF of the 10-minute change")
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_CHANGE)
    plt.close(fig)
    return change


# --------------------------------------------------------------------------
# (f) turbulence and stability vs change magnitude
# --------------------------------------------------------------------------
def plot_turbulence(df: pd.DataFrame, st: Stats) -> None:
    """Relate TI and ri at time t to the size of the NEXT 10-minute change.

    The forward-looking pairing is the one that matters for forecasting: if TI
    at the issue time says something about |ws(t+1) - ws(t)|, then feeding the
    turbulence channels to the model is physically justified.
    """
    work = pd.DataFrame({
        "ws": df[TARGET],
        "sd": df["sd"],
        "ri": df["ri"],
        "next_change": df[TARGET].shift(-1) - df[TARGET],
    }).dropna()
    work["abs_change"] = work["next_change"].abs()
    work = work[work["ws"] >= MIN_WS_FOR_TI]
    work["ti"] = work["sd"] / work["ws"]

    st.add("turb_meta", "n_pairs", value=len(work), n=len(work))
    st.add("turb_meta", "min_ws_for_ti", MIN_WS_FOR_TI)
    st.add("turb_meta", "ti_median", float(work["ti"].median()))
    st.add("turb_meta", "ri_median", float(work["ri"].median()))

    for name in ("ti", "ri"):
        rho, p = stats.spearmanr(work[name], work["abs_change"])
        st.add("turb_corr", f"{name}_spearman", float(rho), n=len(work))
        st.add("turb_corr", f"{name}_spearman_p", float(p))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    summary = {}
    for ax, name, label in ((axes[0], "ti", "turbulence intensity TI = sd / ws100"),
                            (axes[1], "ri", "Richardson number ri")):
        # Decile bins: robust to the heavy tails of TI and the clipping of ri.
        edges = np.unique(np.quantile(work[name], np.linspace(0, 1, N_BINS + 1)))
        bins = pd.cut(work[name], bins=edges, include_lowest=True)
        grouped = work.groupby(bins, observed=True)["abs_change"]
        means, counts = grouped.mean(), grouped.size()
        # Plot at the within-bin MEDIAN, not the interval midpoint: the open
        # top decile of TI reaches 1,5 and its midpoint would misplace the
        # point far to the right of where the data actually sit.
        centers = work.groupby(bins, observed=True)[name].median().to_numpy()

        for center, interval, value, count in zip(centers, means.index,
                                                  means.to_numpy(), counts.to_numpy()):
            st.add(f"{name}_bins", f"[{interval.left:.4f}, {interval.right:.4f}]",
                   float(value), bin_center=float(center), n=int(count))

        lo, hi = float(means.iloc[0]), float(means.iloc[-1])
        summary[name] = (lo, hi)
        st.add("turb_effect", f"{name}_bottom_bin_mean_change", lo)
        st.add("turb_effect", f"{name}_top_bin_mean_change", hi)
        st.add("turb_effect", f"{name}_ratio_top_over_bottom",
               hi / lo if lo else np.nan)

        ax.plot(centers, means.to_numpy(), marker="o", lw=1.8, color="#1f4e79")
        ax.set_xlabel(label)
        ax.set_ylabel("mean |ws(t+1) - ws(t)| [m/s]")
        ax.set_title(f"Next-step change vs {name.upper()} (decile bins)")

    fig.suptitle("Turbulence and stability at the issue time against the size "
                 "of the next 10-minute change", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / FIG_TURB)
    plt.close(fig)


# --------------------------------------------------------------------------
# report section
# --------------------------------------------------------------------------
def build_report(stats_df: pd.DataFrame) -> None:
    """Write report/01_data_and_eda.md purely from results/eda_stats.csv."""
    # --- Table 1-1: data set characteristics -----------------------------
    chars = pd.DataFrame({
        "Characteristic": [
            "Source file", "Period", "Resolution", "Records",
            "Missing values", "Mean wind speed ws100 [m/s]",
            "Standard deviation [m/s]", "Range [m/s]",
            "Weibull shape k [-]", "Weibull scale c [m/s]",
        ],
        "Value": [
            pick(stats_df, "dataset", "source", "text"),
            pick(stats_df, "dataset", "period", "text"),
            f"{fmt(pick(stats_df, 'dataset', 'resolution_min'), 0)} min",
            fmt_int(pick(stats_df, "dataset", "records")),
            fmt(pick(stats_df, "dataset", "missing_values"), 0),
            fmt(pick(stats_df, "dataset", "mean_ws100"), 2),
            fmt(pick(stats_df, "dataset", "std_ws100"), 2),
            f"{fmt(pick(stats_df, 'dataset', 'min_ws100'), 2)} - "
            f"{fmt(pick(stats_df, 'dataset', 'max_ws100'), 2)}",
            fmt(pick(stats_df, "weibull", "k"), 3),
            fmt(pick(stats_df, "weibull", "c"), 3),
        ],
    })

    # --- Table 1-2: ACF at the eight forecast horizons -------------------
    acf_rows = (stats_df[stats_df["group"] == "acf_horizon"]
                .sort_values("horizon_steps"))
    acf_tbl = pd.DataFrame({
        "Horizon [min]": acf_rows["horizon_min"].astype(int).to_numpy(),
        "ACF [-]": acf_rows["value"].to_numpy(),
        "Persistence share of variance r2 [-]": (acf_rows["value"] ** 2).to_numpy(),
    })

    # --- Table 1-3: step-to-step change statistics -----------------------
    change_tbl = pd.DataFrame({
        "Statistic": ["Median", "90th percentile", "95th percentile",
                      "Mean", "Maximum"],
        "|ws(t) - ws(t-1)| [m/s]": [
            pick(stats_df, "change", "median"),
            pick(stats_df, "change", "p90"),
            pick(stats_df, "change", "p95"),
            pick(stats_df, "change", "mean"),
            pick(stats_df, "change", "max"),
        ],
    })

    acf_1 = float(pick(stats_df, "acf_horizon", "h1"))
    acf_3 = float(pick(stats_df, "acf_horizon", "h3"))
    acf_12 = float(pick(stats_df, "acf_horizon", "h12"))
    acf_48 = float(pick(stats_df, "acf_horizon", "h48"))
    median_change = float(pick(stats_df, "change", "median"))
    p95_change = float(pick(stats_df, "change", "p95"))
    mean_ws = float(pick(stats_df, "dataset", "mean_ws100"))

    ti_rho = float(pick(stats_df, "turb_corr", "ti_spearman"))
    ri_rho = float(pick(stats_df, "turb_corr", "ri_spearman"))
    ti_lo = float(pick(stats_df, "turb_effect", "ti_bottom_bin_mean_change"))
    ti_hi = float(pick(stats_df, "turb_effect", "ti_top_bin_mean_change"))
    ti_ratio = float(pick(stats_df, "turb_effect", "ti_ratio_top_over_bottom"))
    ri_lo = float(pick(stats_df, "turb_effect", "ri_bottom_bin_mean_change"))
    ri_hi = float(pick(stats_df, "turb_effect", "ri_top_bin_mean_change"))
    ri_ratio = float(pick(stats_df, "turb_effect", "ri_ratio_top_over_bottom"))
    n_pairs = int(pick(stats_df, "turb_meta", "n_pairs"))

    def strength(rho: float) -> str:
        a = abs(rho)
        if a < 0.05:
            return "negligible"
        if a < 0.15:
            return "weak but systematic"
        if a < 0.30:
            return "moderate"
        return "strong"

    ri_direction = "falls" if ri_rho < 0 else "rises"
    justified = abs(ti_rho) >= 0.05 or abs(ri_rho) >= 0.05
    verdict = (
        "The turbulence and stability channels therefore carry genuine "
        "information about the size of the coming change, and the "
        "\"full_turb\" feature set is physically justified rather than merely "
        "opportunistic."
        if justified else
        "Neither channel shows a usable relation to the coming change, so the "
        "\"full_turb\" feature set cannot be justified on physical grounds from "
        "this evidence alone."
    )

    intro = (
        "This section establishes the empirical properties of the 10-minute "
        "record before any model is fitted: the distribution of the target, the "
        "speed at which its autocorrelation decays, the typical size of a "
        "10-minute change, and whether the turbulence and stability channels "
        "carry information about that change. All numbers below are read from "
        "`results/eda_stats.csv`; all figures are written to `figures/`."
    )

    acf_text = (
        "## Interpretation: what the ACF implies for achievable accuracy\n\n"
        f"The autocorrelation of ws100 at one step is {fmt(acf_1)}, so the series "
        f"is almost perfectly self-predictive over 10 minutes. Squaring the "
        f"coefficient gives the share of variance a pure persistence forecast can "
        f"explain: about {fmt(100 * acf_1 ** 2, 1)} % at 10 minutes, "
        f"{fmt(100 * acf_3 ** 2, 1)} % at 30 minutes, "
        f"{fmt(100 * acf_12 ** 2, 1)} % at 120 minutes and only "
        f"{fmt(100 * acf_48 ** 2, 1)} % at 480 minutes. The practical consequence "
        "for this study is asymmetric across the horizon range. At 10 to 30 "
        "minutes there is very little unexplained variance left for a learned "
        "model to capture, so any improvement over persistence must be small — a "
        "few percent of RMSE at most — and a large improvement at 10 minutes "
        "would indicate leakage rather than skill. From roughly 120 minutes "
        "onwards the persistence share of variance drops sharply, and it is "
        "there that exogenous information and non-linear models have room to pay "
        "off. The eight horizons therefore span two genuinely different "
        "regimes, and the evaluation must not average them into a single "
        "headline number."
    )

    turb_text = (
        "## Interpretation: do turbulence and stability relate to short-term change?\n\n"
        "Pairing each issue time with the size of the following 10-minute change "
        f"over {fmt_int(n_pairs)} usable records gives a Spearman correlation "
        f"of {fmt(ti_rho)} for the "
        f"turbulence intensity TI = sd / ws100 and {fmt(ri_rho)} for the "
        f"Richardson number ri — {strength(ti_rho)} and {strength(ri_rho)} "
        "respectively. The decile view is more informative than the correlation "
        f"alone: mean |ws(t+1) - ws(t)| rises from {fmt(ti_lo)} m/s in the "
        f"lowest TI decile to {fmt(ti_hi)} m/s in the highest, a factor of "
        f"{fmt(ti_ratio, 2)}. Across the ri deciles the mean change "
        f"{ri_direction} from {fmt(ri_lo)} m/s to {fmt(ri_hi)} m/s, a factor of "
        f"{fmt(ri_ratio, 2)}, consistent with the physical expectation that a "
        "stably stratified boundary layer suppresses vertical mixing and hence "
        "short-term variability. Both channels are measured at the issue time "
        "and are used lagged only, so they introduce no leakage. " + verdict
    )

    figures_block = "\n\n".join([
        f"![ws100 time series](../figures/{FIG_TIMESERIES})\n\n"
        "Figure 1-1: Wind speed ws100 over the full record at 10-minute "
        "resolution, with a zoom on the most variable three-day window showing "
        "the 10-minute detail.",

        f"![ACF and PACF](../figures/{FIG_ACF})\n\n"
        f"Figure 1-2: Autocorrelation and partial autocorrelation of ws100 up to "
        f"{MAX_LAG} steps ({MAX_LAG * STEP_MINUTES} minutes), with the eight "
        "forecast horizons marked.",

        f"![Diurnal profile](../figures/{FIG_DIURNAL})\n\n"
        f"Figure 1-3: Mean diurnal profile of ws100 at 10-minute resolution "
        f"({STEPS_PER_DAY} points per day), overall and by season.",

        f"![Weibull fit](../figures/{FIG_WEIBULL})\n\n"
        "Figure 1-4: Distribution of ws100 with the fitted Weibull density.",

        f"![Step change](../figures/{FIG_CHANGE})\n\n"
        "Figure 1-5: Distribution and empirical CDF of the 10-minute change "
        "|ws(t) - ws(t-1)|, with median and 95th percentile marked.",

        f"![Turbulence vs change](../figures/{FIG_TURB})\n\n"
        "Figure 1-6: Mean absolute next-step change against turbulence intensity "
        "and Richardson number, in decile bins.",
    ])

    change_text = (
        f"A median 10-minute change of {fmt(median_change)} m/s against a mean "
        f"wind speed of {fmt(mean_ws, 2)} m/s — about "
        f"{fmt(100 * median_change / mean_ws, 1)} % — is the direct reason "
        "persistence is such a demanding baseline at short lead times. Even the "
        f"95th percentile change of {fmt(p95_change)} m/s is modest relative to "
        "the level of the series."
    )

    write_section(SECTION_MD, "1 Data Basis and Exploratory Analysis", [
        intro,
        "## 1.1 Data set",
        md_table(chars, "Characteristics of the 10-minute data set", "1-1"),
        "## 1.2 Autocorrelation at the forecast horizons",
        md_table(acf_tbl, "Autocorrelation of ws100 at the eight forecast "
                          "horizons", "1-2"),
        "## 1.3 Step-to-step variability",
        md_table(change_tbl, "Statistics of the 10-minute change in ws100", "1-3"),
        change_text,
        "## 1.4 Figures",
        figures_block,
        acf_text,
        turb_text,
    ])


# --------------------------------------------------------------------------
def main() -> None:
    np.random.seed(SEED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_10min()
    info = describe_dataset(df)
    st = Stats()

    st.add("dataset", "source", text="data/clean_10min.csv")
    st.add("dataset", "period",
           text=f"{info['start']:%d.%m.%Y} - {info['end']:%d.%m.%Y}")
    st.add("dataset", "period_start", text=str(info["start"]))
    st.add("dataset", "period_end", text=str(info["end"]))
    st.add("dataset", "records", info["records"], n=info["records"])
    st.add("dataset", "span_days", info["span_days"])
    st.add("dataset", "resolution_min", info["resolution_min"])
    st.add("dataset", "missing_values", info["missing_values"])
    st.add("dataset", "mean_ws100", info["mean_ws100"])
    st.add("dataset", "std_ws100", info["std_ws100"])
    st.add("dataset", "min_ws100", info["min_ws100"])
    st.add("dataset", "max_ws100", info["max_ws100"])

    print("(a) time series ...")
    plot_timeseries(df, st)
    print("(b) ACF / PACF ...")
    plot_acf_pacf(df, st)
    print("(c) diurnal profile ...")
    plot_diurnal(df, st)
    print("(d) Weibull fit ...")
    plot_weibull(df, st)
    print("(e) step-to-step change ...")
    plot_step_change(df, st)
    print("(f) turbulence and stability ...")
    plot_turbulence(df, st)

    stats_df = st.frame()
    stats_df.to_csv(EDA_STATS_CSV, index=False)
    print(f"\nwrote {EDA_STATS_CSV.relative_to(EDA_STATS_CSV.parents[1])} "
          f"({len(stats_df)} rows)")

    # Re-read from disk: the report must be generated from the result file.
    build_report(pd.read_csv(EDA_STATS_CSV))
    print(f"wrote {SECTION_MD.relative_to(SECTION_MD.parents[1])}")

    acf_1 = float(pick(stats_df, "acf_horizon", "h1"))
    ti_ratio = float(pick(stats_df, "turb_effect", "ti_ratio_top_over_bottom"))
    log_progress(2, "EDA on the 10-minute record completed; ACF at 10 min is "
                    f"{fmt(acf_1)} (persistence near-unbeatable at short lead "
                    f"times) and the mean next-step change is {fmt(ti_ratio, 2)}x "
                    "larger in the top TI decile than in the bottom, supporting "
                    "the full_turb feature set.")
    print("appended to report/PROGRESS.md")


if __name__ == "__main__":
    main()
