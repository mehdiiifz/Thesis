"""Stage 1: load the raw PVGIS timeseries, clean it, audit quality, save data/clean.pkl."""

import re
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "Timeseries_27.517_49.050_SA3_1000kWp_crystSi_15_i0deg_2012_2013_1.csv"
CLEAN_PKL = ROOT / "data" / "clean.pkl"
AUDIT_TXT = ROOT / "results" / "stage1_audit.txt"

DATA_ROW = re.compile(r"^\d{8}:\d{4},")


def load_raw(path: Path) -> pd.DataFrame:
    """Read the PVGIS CSV, skipping the metadata header and the legend footer.

    PVGIS files start with ~10 lines of site metadata, then the column header
    line ("time,P,..."), then hourly rows, then a legend/footer. Only rows whose
    first field matches YYYYMMDD:HHMM are actual data.
    """
    lines = path.read_text().splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("time,"))
    columns = lines[header_idx].split(",")
    data_lines = [line for line in lines[header_idx + 1 :] if DATA_ROW.match(line)]
    return pd.read_csv(StringIO("\n".join(data_lines)), names=columns)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse UTC timestamps, floor to the hour, index by time, add P_kW."""
    ts = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True).dt.floor("h")
    df = df.drop(columns="time").set_index(ts.rename("time")).sort_index()
    df["P_kW"] = df["P"] / 1000.0  # P is reported in W; the plant is 1000 kWp
    return df


def audit(df: pd.DataFrame) -> str:
    out = []
    out.append(f"Rows: {len(df)}")
    out.append(f"Date range (UTC): {df.index.min()} .. {df.index.max()}")

    out.append("\nMissing values per column:")
    out.append(df.isna().sum().to_string())

    dup = int(df.index.duplicated().sum())
    out.append(f"\nDuplicate timestamps: {dup}")

    full_grid = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    gaps = full_grid.difference(df.index)
    out.append(f"Gaps vs perfect hourly grid ({len(full_grid)} expected hours): {len(gaps)}")
    if len(gaps):
        out.append(f"  first gaps: {list(gaps[:5])}")

    out.append(f"\nRows flagged Int=1 (reconstructed irradiance): {int((df['Int'] == 1).sum())}")

    out.append("\nP_kW describe:")
    out.append(df["P_kW"].describe().to_string())

    cf = df["P_kW"].mean() / 1000.0
    out.append(f"\nCapacity factor: {cf:.1%} (mean output / 1000 kWp nameplate)")
    out.append(f"Hours with P > 0: {(df['P_kW'] > 0).mean():.1%}")
    return "\n".join(out)


def main() -> None:
    df = clean(load_raw(RAW_CSV))
    report = audit(df)
    print(report)
    AUDIT_TXT.write_text(report + "\n")
    df.to_pickle(CLEAN_PKL)
    print(f"\nSaved {CLEAN_PKL.relative_to(ROOT)} ({len(df)} rows) and {AUDIT_TXT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
