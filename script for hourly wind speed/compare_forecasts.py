"""Score our model against an archived NWP forecast, using ERA5 as truth.

Inputs
------
--era5 : Copernicus CDS CSV with valid_time, u100, v100 (the reference series;
         it supplies BOTH the 24 h history our model consumes and the truth)
--nwp  : Open-Meteo archive CSV from src/fetch_openmeteo_archive.py
         (must contain a ws100_nwp column on the same hours)

Our model is evaluated in three forms, because the August-2026 transfer test
showed that applying the 2012-fitted equation directly to raw ERA5 fails
(it carries the training site's climatology):

  our_raw          the equation exactly as fitted on the 2012 WRF-LES data
  our_biascorr     the same, shifted onto this window's mean
  our_refit        the equation REFITTED on an earlier slice of this same
                   ERA5 series and evaluated out-of-sample on the rest

`our_refit` is the scientifically fair version: same architecture and same
24-lag inputs, but calibrated to the data it is applied to, so the comparison
with the NWP measures the method rather than a climatology mismatch. The
refit uses a strict chronological split - the test period is never seen.

Baselines (persistence, climatology) are included because on a short window
climatology can be surprisingly hard to beat.

Usage:
    python src/compare_forecasts.py --era5 data/era5_2026/era5.csv \\
        --nwp data/openmeteo_forecasts/archive_2024-06-01_2024-08-31.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import scores  # noqa: E402
from predict_era5_window import load_equation, load_era5  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"
N_LAGS, HORIZON = 24, 24
TRAIN_MEAN = 6.71
REFIT_FRAC = 0.6          # first 60 % of the ERA5 series trains the refit


def build_xy(v: np.ndarray):
    """Supervised matrices from a 1-D series: X = 24 lags, Y = next 24 hours."""
    rows_x, rows_y, idx = [], [], []
    for i in range(N_LAGS - 1, len(v) - HORIZON):
        rows_x.append(v[i - N_LAGS + 1:i + 1][::-1])
        rows_y.append(v[i + 1:i + 1 + HORIZON])
        idx.append(i)
    return np.array(rows_x), np.array(rows_y), np.array(idx)


def refit_equation(X, Y, alpha: float = 1.0):
    """Per-step ridge, same form as src/transparent_model.py."""
    W = np.zeros((HORIZON, X.shape[1]))
    b = np.zeros(HORIZON)
    for j in range(HORIZON):
        m = Ridge(alpha=alpha).fit(X, Y[:, j])
        W[j], b[j] = m.coef_, m.intercept_
    return b, W


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--era5", type=Path, required=True)
    ap.add_argument("--nwp", type=Path, required=True)
    a = ap.parse_args()

    s = load_era5(a.era5)
    v = s.values
    X, Y, idx = build_xy(v)
    if len(X) < 200:
        print(f"WARNING: only {len(X)} forecast origins - results will be "
              "noisy. Use several months for a meaningful comparison.")

    # chronological split for the refit variant
    cut = int(len(X) * REFIT_FRAC)
    b0, W0 = load_equation()                      # 2012-fitted
    b1, W1 = refit_equation(X[:cut], Y[:cut])     # refitted on this series
    shift = s.mean() - TRAIN_MEAN

    te = slice(cut, len(X))
    Xte, Yte, ite = X[te], Y[te], idx[te]
    preds = {
        "our_raw": Xte @ W0.T + b0,
        "our_biascorr": (Xte - shift) @ W0.T + b0 + shift,
        "our_refit": Xte @ W1.T + b1,
        "persistence": np.repeat(Xte[:, [0]], HORIZON, axis=1),
        "climatology": np.full(Yte.shape, v[:cut].mean()),
    }

    # NWP: align by valid time (archive stores by valid time, no lead axis)
    nwp = pd.read_csv(a.nwp, index_col=0, parse_dates=[0])
    if "ws100_nwp" not in nwp:
        sys.exit("--nwp file has no ws100_nwp column")
    nwp_s = nwp["ws100_nwp"]
    nwp_grid = np.full(Yte.shape, np.nan)
    for r, i in enumerate(ite):
        valid = s.index[i + 1:i + 1 + HORIZON]
        nwp_grid[r] = nwp_s.reindex(valid).values
    if np.isfinite(nwp_grid).mean() < 0.5:
        print(f"WARNING: only {100*np.isfinite(nwp_grid).mean():.0f}% of hours "
              "matched between the ERA5 and NWP files - check the periods "
              "and time zones overlap.")
    preds["nwp_openmeteo"] = nwp_grid

    ok = np.isfinite(nwp_grid).all(axis=1)        # score on common rows only
    rows = []
    for name, p in preds.items():
        sc = scores(Yte[ok], p[ok])
        sc["model"] = name
        sc["bias"] = float((p[ok] - Yte[ok]).mean())
        rows.append(sc)
    tab = pd.DataFrame(rows)[["model", "rmse", "mae", "r", "nrmse", "bias"]]

    print(f"\nERA5 series : {len(s)} hours, mean {s.mean():.2f} m/s")
    print(f"refit train : {cut} origins | test: {ok.sum()} origins "
          f"(common with NWP)")
    print("\nScored against ERA5 (pooled over origins and lead times):")
    print(tab.round(3).to_string(index=False))
    tab.to_csv(RESULTS / "forecast_comparison_metrics.csv", index=False)

    per_lead = pd.DataFrame(
        {n: [np.sqrt(np.nanmean((p[ok][:, j] - Yte[ok][:, j]) ** 2))
             for j in range(HORIZON)] for n, p in preds.items()},
        index=range(1, HORIZON + 1))
    per_lead.index.name = "lead_h"
    per_lead.to_csv(RESULTS / "forecast_comparison_per_lead.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    style = {"our_raw": ("tab:red", "--"), "our_biascorr": ("tab:orange", "-"),
             "our_refit": ("tab:green", "-"), "nwp_openmeteo": ("tab:blue", "-"),
             "persistence": ("0.45", ":"), "climatology": ("0.7", ":")}
    for n in preds:
        c, ls = style[n]
        ax.plot(per_lead.index, per_lead[n], ls, color=c,
                lw=2.2 if n in ("our_refit", "nwp_openmeteo") else 1.3, label=n)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Our model vs archived NWP forecast (ERA5 as truth)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "forecast_comparison.png", dpi=150)
    print(f"\nwrote results/forecast_comparison_{{metrics,per_lead}}.csv and "
          "figures/forecast_comparison.png")
    print("\nReport 'our_refit' as the fair comparison; 'our_raw' shows what "
          "happens without recalibration. Truth is ERA5, not measurements.")


if __name__ == "__main__":
    main()
