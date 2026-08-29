"""Load the native 10-minute wind data set.

data/clean_10min.csv holds 52.703 records at 10-minute resolution covering
10.01.2012-10.01.2013 with no gaps. The timestamp column becomes the
DatetimeIndex; every downstream step relies on that index being sorted,
uniformly spaced and free of duplicates, so load_10min() checks all three.

Run from the project root:

    python src/load_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "clean_10min.csv"

TARGET = "ws100"
STEP_MINUTES = 10

# Columns expected in the raw file, in order.
EXPECTED_COLUMNS = [
    "ws100", "dir", "sd", "dsd", "gust3s", "temp_c", "pres_hpa",
    "ri", "vert_ws", "ws107", "ws95",
]


def load_10min(path: Path | str = DATA_CSV, validate: bool = True) -> pd.DataFrame:
    """Return the 10-minute data set with a parsed DatetimeIndex.

    The index is named "timestamp" and is sorted ascending. With validate=True
    (the default) the function asserts that the series is uniformly spaced at
    10 minutes, has no duplicate timestamps and contains no missing values --
    the properties every later step assumes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- expected the cleaned 10-minute data set")

    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name} is missing expected columns: {missing_cols}")

    if validate:
        if df.index.has_duplicates:
            dupes = df.index[df.index.duplicated()].unique()
            raise ValueError(f"duplicate timestamps in {path.name}: {list(dupes[:5])}")

        spacing = df.index.to_series().diff().dropna().unique()
        expected = pd.Timedelta(minutes=STEP_MINUTES)
        if len(spacing) != 1 or spacing[0] != expected:
            raise ValueError(
                f"{path.name} is not uniformly spaced at {STEP_MINUTES} min; "
                f"found spacings {list(spacing[:5])}")

        n_missing = int(df.isna().sum().sum())
        if n_missing:
            raise ValueError(f"{path.name} contains {n_missing} missing values")

    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """Key characteristics of the loaded set, for logging and the report."""
    span = df.index.max() - df.index.min()
    return {
        "records": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "span_days": span.total_seconds() / 86400.0,
        "resolution_min": STEP_MINUTES,
        "missing_values": int(df.isna().sum().sum()),
        "mean_ws100": float(df[TARGET].mean()),
        "std_ws100": float(df[TARGET].std()),
        "min_ws100": float(df[TARGET].min()),
        "max_ws100": float(df[TARGET].max()),
    }


if __name__ == "__main__":
    frame = load_10min()
    info = describe_dataset(frame)

    print("=" * 72)
    print(f"Loaded {DATA_CSV.relative_to(ROOT)}")
    print("=" * 72)
    print(f"shape            : {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(f"period           : {info['start']}  ->  {info['end']}")
    print(f"span             : {info['span_days']:.2f} days")
    print(f"resolution       : {info['resolution_min']} min (uniform, no gaps)")
    print(f"index            : {type(frame.index).__name__}, name={frame.index.name!r}")
    print()

    print("-" * 72)
    print("Missing values per column")
    print("-" * 72)
    na = frame.isna().sum()
    print(na.to_string())
    print(f"total missing    : {int(na.sum())}")
    print()

    print("-" * 72)
    print("describe()")
    print("-" * 72)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.describe().T.to_string())
