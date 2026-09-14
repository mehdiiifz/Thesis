"""Build supervised direct multi-output datasets from the 10-minute series.

Everything here is counted in ROWS/STEPS, not wall-clock time: one step is one
row of the input series (10 minutes for data/clean_10min.csv).

Direct multi-output framing (see CLAUDE.md):
    issue time t  ->  X row built from information available at or before t
                      Y row = ws100 at t+1, t+2, ..., t+horizon_steps

Multi-resolution lag design
---------------------------
ws100 at 10-minute resolution is autocorrelated at 0,992 over one step and
still at 0,55 over 48 steps, so 144 consecutive raw lags would be almost
perfectly collinear: a great many features carrying very little independent
information. Instead the wind history is compressed with a multi-resolution
design that keeps full detail where the series changes fastest and averages
progressively harder further back:

    fine    ws100 at lags 0..11                     12 features (last 2 hours)
    coarse  block means over lags [12..17], [18..23],
            [24..35], [36..47], [48..71], [72..143]  6 features (to ~24 hours)

18 wind features replace 144 raw lags while retaining both recent detail and
long memory. The design is produced by multiresolution_lag_design() so it can
be varied later without touching the feature-building code.

Feature sets:
    "lags_only"  the 18 multi-resolution wind features
    "full"       + time-of-day and day-of-year sin/cos
                 + last 3 steps of temp_c, pres_hpa, u, v
    "full_turb"  + last 3 steps of sd, dsd, ri, vert_ws
                 + current turbulence intensity TI = sd / ws100

Hard rules enforced here:
- ws107, ws95 and gust3s are NEVER used as features (same wind field as the
  target -> leakage).
- Every feature is observed at or before the issue time t; every target is
  strictly after it.

Run from the project root to regenerate the methodology section:

    python src/make_dataset.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

TARGET = "ws100"
STEP_MINUTES = 10

# Same wind field as the target: using these would leak the answer.
BANNED = ("ws107", "ws95", "gust3s")

# --- multi-resolution lag design ------------------------------------------
FINE_LAGS = 12
COARSE_BLOCKS: tuple[tuple[int, int], ...] = (
    (12, 17), (18, 23), (24, 35), (36, 47), (48, 71), (72, 143),
)

# --- exogenous channels ----------------------------------------------------
EXOG_LAGS = 3
EXOG_FULL = ("temp_c", "pres_hpa", "u", "v")
EXOG_TURB = ("sd", "dsd", "ri", "vert_ws")

FEATURE_SETS = ("lags_only", "full", "full_turb")

# Feature group labels, used for the methodology tables.
G_FINE = "Fine wind lags"
G_COARSE = "Coarse wind lag blocks"
G_TIME = "Time encodings"
G_EXOG = "Lagged exogenous (temp_c, pres_hpa, u, v)"
G_TURB = "Lagged turbulence and stability (sd, dsd, ri, vert_ws)"
G_TI = "Turbulence intensity TI"

GROUP_ORDER = [G_FINE, G_COARSE, G_TIME, G_EXOG, G_TURB, G_TI]


@dataclass(frozen=True)
class LagBlock:
    """One block of the multi-resolution wind-lag design.

    kind      "fine" (one feature per lag) or "coarse" (one mean per block)
    lag_from  smallest lag in the block, in steps back from the issue time t
    lag_to    largest lag in the block, in steps back from t
    columns   names of the features this block contributes
    """

    kind: str
    lag_from: int
    lag_to: int
    columns: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.columns)

    @property
    def minutes_from(self) -> int:
        return self.lag_from * STEP_MINUTES

    @property
    def minutes_to(self) -> int:
        return self.lag_to * STEP_MINUTES

    @property
    def label(self) -> str:
        # The step range is carried by its own table column, so the label only
        # names the kind of block.
        if self.kind == "fine":
            return "Fine — one feature per step"
        return "Coarse — block mean"


def multiresolution_lag_design(
    fine_lags: int = FINE_LAGS,
    coarse_blocks: tuple[tuple[int, int], ...] = COARSE_BLOCKS,
) -> list[LagBlock]:
    """Return the multi-resolution wind-lag design as a list of LagBlocks.

    The default is the design documented in the module docstring: `fine_lags`
    individual lags (0 .. fine_lags-1) followed by block means over each
    (lag_from, lag_to) pair in `coarse_blocks`, both inclusive.

    Vary the design by passing different arguments, e.g.

        design = multiresolution_lag_design(fine_lags=6,
                                            coarse_blocks=((6, 11), (12, 23)))
        X, Y, names = build_supervised(df, horizon_steps=6, design=design)

    Blocks must be contiguous, non-overlapping and increasing, so that the
    design covers a clean lag range with no gaps or double counting.
    """
    if fine_lags < 1:
        raise ValueError(f"fine_lags must be >= 1, got {fine_lags}")

    blocks = [LagBlock(
        kind="fine",
        lag_from=0,
        lag_to=fine_lags - 1,
        columns=tuple(f"{TARGET}_lag{j}" for j in range(fine_lags)),
    )]

    expected_start = fine_lags
    for lag_from, lag_to in coarse_blocks:
        if lag_to < lag_from:
            raise ValueError(f"coarse block ({lag_from}, {lag_to}) is reversed")
        if lag_from != expected_start:
            raise ValueError(
                f"coarse block ({lag_from}, {lag_to}) must start at lag "
                f"{expected_start} to stay contiguous with the previous block")
        blocks.append(LagBlock(
            kind="coarse",
            lag_from=lag_from,
            lag_to=lag_to,
            columns=(f"{TARGET}_mean_lag{lag_from}_{lag_to}",),
        ))
        expected_start = lag_to + 1

    return blocks


def max_lag(design: list[LagBlock] | None = None) -> int:
    """Largest lag touched by the design, in steps."""
    design = design or multiresolution_lag_design()
    return max(block.lag_to for block in design)


def lag_design_table(design: list[LagBlock] | None = None) -> pd.DataFrame:
    """The lag design as a table, for the methodology section."""
    design = design or multiresolution_lag_design()
    rows = [{
        "block": block.label,
        "kind": block.kind,
        "lag_from": block.lag_from,
        "lag_to": block.lag_to,
        "minutes_from": block.minutes_from,
        "minutes_to": block.minutes_to,
        "n_features": block.n_features,
    } for block in design]
    return pd.DataFrame(rows)


def classify_feature(name: str) -> str:
    """Map a built feature name onto its methodology group."""
    if name.startswith(f"{TARGET}_mean_lag"):
        return G_COARSE
    if name.startswith(f"{TARGET}_lag"):
        return G_FINE
    if name.startswith("ti_lag"):
        return G_TI
    if name in ("tod_sin", "tod_cos", "doy_sin", "doy_cos"):
        return G_TIME
    root = name.rsplit("_lag", 1)[0]
    if root in EXOG_TURB:
        return G_TURB
    if root in EXOG_FULL:
        return G_EXOG
    raise ValueError(f"unclassified feature {name!r}")


def add_uv(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind vector components u, v from ws100 and dir (meteorological).

    dir is the direction the wind comes FROM, in degrees; u is the eastward
    component and v the northward component:

        u = -ws * sin(dir_rad)      v = -ws * cos(dir_rad)
    """
    out = df.copy()
    if "dir" in out.columns and TARGET in out.columns:
        rad = np.deg2rad(out["dir"].to_numpy(dtype=float))
        ws = out[TARGET].to_numpy(dtype=float)
        out["u"] = -ws * np.sin(rad)
        out["v"] = -ws * np.cos(rad)
    return out


def _time_encodings(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Cyclic time encodings -- known exactly at the issue time, so not lagged.

    Time of day uses minutes since midnight with period 1440; day of year uses
    period 365,25 to stay stable across leap years.
    """
    minute_of_day = index.hour * 60 + index.minute
    doy = index.dayofyear
    return {
        "tod_sin": pd.Series(np.sin(2 * np.pi * minute_of_day / 1440.0), index=index),
        "tod_cos": pd.Series(np.cos(2 * np.pi * minute_of_day / 1440.0), index=index),
        "doy_sin": pd.Series(np.sin(2 * np.pi * doy / 365.25), index=index),
        "doy_cos": pd.Series(np.cos(2 * np.pi * doy / 365.25), index=index),
    }


def build_wind_features(ws: pd.Series,
                        design: list[LagBlock] | None = None) -> dict[str, pd.Series]:
    """Multi-resolution wind features from the ws100 series.

    Fine lags are plain shifts. A coarse block [a, b] is the mean of ws over
    lags a..b, computed as ws.shift(a).rolling(b - a + 1).mean(): the shift
    places ws(t-a) at t and the rolling window then averages back to ws(t-b),
    so nothing after t-a can enter.
    """
    design = design or multiresolution_lag_design()
    cols: dict[str, pd.Series] = {}
    for block in design:
        if block.kind == "fine":
            for lag, name in zip(range(block.lag_from, block.lag_to + 1), block.columns):
                cols[name] = ws.shift(lag)
        else:
            width = block.lag_to - block.lag_from + 1
            cols[block.columns[0]] = ws.shift(block.lag_from).rolling(width).mean()
    return cols


def build_supervised(df: pd.DataFrame, horizon_steps: int,
                     features: str = "lags_only",
                     design: list[LagBlock] | None = None,
                     exog_lags: int = EXOG_LAGS,
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (X, Y, feature_names) for direct multi-output forecasting.

    Parameters
    ----------
    df : DataFrame with a DatetimeIndex and at least a ws100 column.
    horizon_steps : number of steps ahead to predict (H >= 1). Y has H columns
        h1..hH holding ws100 at t+1 .. t+H.
    features : one of FEATURE_SETS.
    design : multi-resolution lag design; defaults to the documented one.
    exog_lags : number of lags per exogenous channel (0 .. exog_lags-1).

    X and Y share a DatetimeIndex of forecast-issue times t. Rows with any
    missing value are dropped, so the first usable issue time is the one where
    the deepest lag exists and no target reaches past the end of the series.
    """
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}, got {features!r}")
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")
    if TARGET not in df.columns:
        raise ValueError(f"df must contain the target column {TARGET!r}")

    design = design or multiresolution_lag_design()
    df = add_uv(df)
    ws = df[TARGET]

    cols: dict[str, pd.Series] = dict(build_wind_features(ws, design))

    if features in ("full", "full_turb"):
        cols.update(_time_encodings(df.index))
        for name in EXOG_FULL:
            if name not in df.columns:
                raise ValueError(f"feature set {features!r} needs column {name!r}")
            for lag in range(exog_lags):
                cols[f"{name}_lag{lag}"] = df[name].shift(lag)

    if features == "full_turb":
        for name in EXOG_TURB:
            if name not in df.columns:
                raise ValueError(f"feature set {features!r} needs column {name!r}")
            for lag in range(exog_lags):
                cols[f"{name}_lag{lag}"] = df[name].shift(lag)
        # Current turbulence intensity. Undefined at calm, so guard the divide;
        # the resulting NaN rows drop out with everything else.
        safe_ws = ws.where(ws > 0)
        cols["ti_lag0"] = df["sd"] / safe_ws

    X = pd.DataFrame(cols, index=df.index)

    # Targets: strictly future steps t+1 .. t+horizon_steps.
    Y = pd.DataFrame(
        {f"h{step}": ws.shift(-step) for step in range(1, horizon_steps + 1)},
        index=df.index,
    )

    keep = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X.loc[keep], Y.loc[keep]

    leaked = [c for c in X.columns if any(c.startswith(b) for b in BANNED)]
    assert not leaked, f"banned same-field channels leaked into X: {leaked}"

    return X, Y, list(X.columns)


def feature_group_counts(df: pd.DataFrame,
                         design: list[LagBlock] | None = None,
                         horizon_steps: int = 1) -> pd.DataFrame:
    """Feature count per group and feature set, derived from built matrices.

    Counting the columns that build_supervised actually produces keeps the
    methodology table from drifting away from the code.
    """
    counts = {}
    for feature_set in FEATURE_SETS:
        _, _, names = build_supervised(df, horizon_steps=horizon_steps,
                                       features=feature_set, design=design)
        groups = pd.Series([classify_feature(n) for n in names])
        counts[feature_set] = groups.value_counts()

    table = pd.DataFrame(counts).reindex(GROUP_ORDER).fillna(0).astype(int)
    table.index.name = "Feature group"
    table.loc["Total"] = table.sum()
    return table.reset_index()


# --------------------------------------------------------------------------
# report section
# --------------------------------------------------------------------------
def write_methodology_section() -> None:
    """Write report/02_methodology_features.md from results/*.csv."""
    from load_data import load_10min
    from report_utils import (HORIZON_MIN, REPORT, RESULTS, fmt, log_progress,
                              md_table, write_section)

    RESULTS.mkdir(parents=True, exist_ok=True)
    design = multiresolution_lag_design()
    df = load_10min()

    lag_csv = RESULTS / "lag_design.csv"
    feat_csv = RESULTS / "feature_design.csv"
    excl_csv = RESULTS / "excluded_channels.csv"
    lag_design_table(design).to_csv(lag_csv, index=False)
    feature_group_counts(df, design).to_csv(feat_csv, index=False)

    # Quantify why the banned channels are banned, rather than asserting it.
    # A ratio with near-zero spread means the channel is an exact rescaling of
    # the target and carries no independent information whatsoever.
    ratios = {c: df[c] / df[TARGET] for c in BANNED}
    pd.DataFrame({
        "channel": list(BANNED),
        "pearson_r_with_ws100": [float(df[c].corr(df[TARGET])) for c in BANNED],
        "ratio_to_ws100_mean": [float(ratios[c].mean()) for c in BANNED],
        "ratio_to_ws100_std": [float(ratios[c].std()) for c in BANNED],
    }).to_csv(excl_csv, index=False)
    print(f"wrote {lag_csv.name}, {feat_csv.name} and {excl_csv.name}")

    # Re-read from disk: the section must be generated from the result files.
    lag_df = pd.read_csv(lag_csv)
    feat_df = pd.read_csv(feat_csv)
    excl_raw = pd.read_csv(excl_csv).set_index("channel")
    excl_df = excl_raw["pearson_r_with_ws100"]

    n_wind = int(lag_df["n_features"].sum())
    deepest = int(lag_df["lag_to"].max())
    deepest_min = int(lag_df["minutes_to"].max())
    n_raw = deepest + 1

    tbl_lag = pd.DataFrame({
        "Block": lag_df["block"],
        "Steps covered": [f"t-{a} .. t-{b}" for a, b in
                          zip(lag_df["lag_from"], lag_df["lag_to"])],
        "Minutes covered": [f"{a} - {b}" for a, b in
                            zip(lag_df["minutes_from"], lag_df["minutes_to"])],
        "Features": lag_df["n_features"].astype(int),
    })
    tbl_lag.loc[len(tbl_lag)] = ["Total", f"t-0 .. t-{deepest}",
                                 f"0 - {deepest_min}", n_wind]

    counts = {row["Feature group"]: row for _, row in feat_df.iterrows()}
    total = counts["Total"]

    formulation = (
        "## 2.1 Forecast formulation\n\n"
        "The task is formulated as **direct multi-output** regression. For each "
        "forecast horizon H a separate model is trained, and that model emits "
        "the whole trajectory of the next H steps at once: from the information "
        "available at issue time t it predicts ws100 at t+1, t+2, ..., t+H "
        "simultaneously. Eight models are therefore trained per configuration, "
        f"one for each of the horizons {', '.join(str(m) for m in HORIZON_MIN)} "
        "minutes.\n\n"
        "The alternative would be a recursive one-step model applied to its own "
        "output H times. Direct multi-output is preferred here for three "
        "reasons. First, recursive forecasting feeds predictions back as inputs, "
        "so the error of every intermediate step propagates and compounds into "
        "the next; at H = 48 steps a one-step model would be applied to its own "
        "output 47 times, and the accumulated bias typically dominates the "
        "result long before the final step. Second, a recursive model is fitted "
        "to a one-step loss, which is not the loss the study actually cares "
        "about — each direct model is instead trained on exactly the horizon it "
        "will be judged on. Third, the recursive scheme needs the exogenous "
        "channels at future times as well, which are not available at the issue "
        "time and would have to be forecast themselves, introducing a second "
        "source of error. The cost of the direct approach is that the eight "
        "models do not share parameters and must each be trained separately."
    )

    lag_prose = (
        "## 2.2 Multi-resolution lag design\n\n"
        f"Section 1 established that ws100 is autocorrelated at 0,992 over one "
        f"step and still around 0,55 over 48 steps. Feeding {n_raw} consecutive "
        "raw lags to a model would therefore supply a large block of nearly "
        "collinear inputs: many parameters to estimate, very little independent "
        "information, and a needlessly large search space for the tree and "
        "network models alike.\n\n"
        "The wind history is instead compressed with a multi-resolution design. "
        "The most recent two hours are kept at full 10-minute resolution, "
        "because that is where the series carries the detail a short-horizon "
        "forecast depends on. Beyond that, lags are aggregated into block means "
        "of increasing width, so that the slower components of the flow — the "
        f"passage of weather systems over the past day — are still represented, "
        f"but by {n_wind - 12} summary values rather than {n_raw - 12} "
        f"individual ones. The result is {n_wind} wind features instead of "
        f"{n_raw}, retaining recent detail and long memory at once.\n\n"
        "The design is produced by `multiresolution_lag_design()` in "
        "`src/make_dataset.py` and can be varied by passing a different number "
        "of fine lags or different coarse blocks; the blocks are validated to "
        "be contiguous and non-overlapping, so a modified design still covers a "
        "clean lag range without gaps or double counting."
    )

    sets_prose = (
        "## 2.3 Feature sets\n\n"
        "Three nested feature sets are compared, so that the contribution of "
        "each additional information source can be isolated.\n\n"
        f"- **lags_only** ({total['lags_only']} features): the multi-resolution "
        "wind history alone. This is the reference against which any exogenous "
        "information must justify itself.\n"
        f"- **full** ({total['full']} features): adds cyclic time encodings "
        "(sine and cosine of minutes since midnight with period 1440, and of "
        "the day of year) together with the last three steps of temperature, "
        "pressure and the wind vector components u and v, where "
        "u = -ws * sin(dir) and v = -ws * cos(dir).\n"
        f"- **full_turb** ({total['full_turb']} features): adds the last three "
        "steps of the turbulence and stability channels sd, dsd, ri and "
        "vert_ws, plus the turbulence intensity TI = sd / ws100 at the issue "
        "time. These channels exist only at 10-minute resolution and are the "
        "novel element of this study; Section 1 showed that both TI and the "
        "Richardson number relate monotonically to the size of the next "
        "10-minute change, which is what makes them worth including."
    )

    leakage_prose = (
        "## 2.4 Leakage control\n\n"
        "Every feature listed above is observed at or before the issue time t. "
        "The fine lags are plain shifts of ws100; each coarse block is a mean "
        "over a lag window that ends at t-12 or earlier; the exogenous channels "
        "enter at lags 0, 1 and 2, where lag 0 is the observation made at t "
        "itself; and the time encodings are calendar properties of t, known "
        "exactly in advance. No feature draws on any observation after t, and "
        "the targets are strictly the values at t+1 .. t+H.\n\n"
        "**ws107, ws95 and gust3s are excluded from every feature set.** They "
        "measure the same wind field as the target — ws107 and ws95 are the "
        "wind speed at 107 m and 95 m, within metres of the 100 m target "
        "height, and gust3s is the 3-second gust of that same flow. Their "
        "contemporaneous Pearson correlation with ws100 is "
        f"{fmt(excl_df['ws107'], 4)}, {fmt(excl_df['ws95'], 4)} and "
        f"{fmt(excl_df['gust3s'], 4)} respectively.\n\n"
        "For ws107 and ws95 the case is stronger than a high correlation. In "
        "this data set they are an exact constant multiple of the target: "
        f"ws107 = {fmt(excl_raw.loc['ws107', 'ratio_to_ws100_mean'], 6)} * "
        f"ws100 and ws95 = {fmt(excl_raw.loc['ws95', 'ratio_to_ws100_mean'], 6)} "
        "* ws100, with a ratio standard deviation of order 1e-16 — that is, "
        "floating-point noise. They are evidently not independent measurements "
        "but a shear extrapolation computed from the 100 m record itself, so "
        "they carry no information the target does not already contain. A model "
        "given ws107 could recover ws100 exactly by division, which would "
        "produce a spectacular and entirely meaningless score. gust3s is a "
        "genuine separate quantity but is the 3-second gust of the same flow at "
        "the same instant, and is therefore excluded on the same principle. "
        "`build_supervised()` asserts on every call that no column derived from "
        "these three has entered the feature matrix."
    )

    write_section(REPORT / "02_methodology_features.md",
                  "2 Forecast Formulation and Input Features", [
        "This section defines how the forecasting task is posed and which "
        "inputs the models receive. All counts below are read from "
        "`results/lag_design.csv` and `results/feature_design.csv`, which are "
        "written from the feature-building code itself.",
        formulation,
        lag_prose,
        md_table(tbl_lag, "Multi-resolution lag design for the wind history", "2-1"),
        sets_prose,
        md_table(feat_df, "Feature count per group and feature set", "2-2"),
        leakage_prose,
    ])
    print("wrote report/02_methodology_features.md")

    log_progress(
        "2b", "Feature engineering defined; multi-resolution lag design "
        f"compresses {n_raw} raw wind lags into {n_wind} features "
        f"(12 fine + {n_wind - 12} coarse block means, reaching back "
        f"{deepest_min} minutes), giving feature sets of "
        f"{total['lags_only']}, {total['full']} and {total['full_turb']} "
        "features; ws107, ws95 and gust3s excluded as leakage.")
    print("appended to report/PROGRESS.md")


if __name__ == "__main__":
    from load_data import load_10min

    frame = load_10min()
    blocks = multiresolution_lag_design()
    print(f"lag design: {len(blocks)} blocks, deepest lag {max_lag(blocks)} steps "
          f"({max_lag(blocks) * STEP_MINUTES} min)")
    for fs in FEATURE_SETS:
        Xs, Ys, names = build_supervised(frame, horizon_steps=48, features=fs)
        print(f"  {fs:<10} X={Xs.shape}  Y={Ys.shape}")
    print()
    write_methodology_section()
