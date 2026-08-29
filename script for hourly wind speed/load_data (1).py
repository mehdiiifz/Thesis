"""Load the hourly and 10-min wind datasets with a DatetimeIndex.

Run from project root: python src/load_data.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load(path: Path) -> pd.DataFrame:
    """Read a CSV with a 'timestamp' column into a sorted DatetimeIndex frame."""
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    df.index.name = "timestamp"
    return df


def load_hourly() -> pd.DataFrame:
    """Return the hourly dataset (8,784 rows, Jan 2012 - Jan 2013).

    Target column: ws100. ws107/ws95/gust3s are the same wind field
    (never features); n_obs is a QC counter.
    """
    return _load(DATA_DIR / "hourly.csv")


def load_10min() -> pd.DataFrame:
    """Return the original 10-min resolution dataset."""
    return _load(DATA_DIR / "clean_10min.csv")


def _report(name: str, df: pd.DataFrame) -> None:
    """Print shape, date range, index regularity, missing counts, describe()."""
    print(f"=== {name} ===")
    print(f"shape: {df.shape}")
    print(f"range: {df.index.min()} -> {df.index.max()}")
    print(f"inferred freq: {pd.infer_freq(df.index)}")
    gaps = df.index.to_series().diff().value_counts()
    if len(gaps) > 1:
        print("index steps (irregular!):")
        print(gaps.to_string())
    print("missing values per column:")
    print(df.isna().sum().to_string())
    print(df.describe().T.round(2).to_string())
    print()


if __name__ == "__main__":
    _report("hourly", load_hourly())
    _report("10min", load_10min())
