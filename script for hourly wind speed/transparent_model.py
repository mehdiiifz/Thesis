"""A transparent, closed-form linear forecast equation for ws100.

Why this exists: the study's best models (stacking ensemble, Extra Trees,
LSTM) have no closed form - a fitted Extra Trees model is 300 trees, an LSTM
is thousands of weights. They can only be *applied* by loading the saved
model files. That makes them hard to quote in a thesis and hard to port to
another site.

This module fits a per-step regularised linear (ridge) model on the same
24 lags, which CAN be written as an equation:

    ws100(t+h) = b_h + sum_{k=0..23} w_{h,k} * ws100(t-k)

It is deliberately weaker than the ensembles, but it is fully transparent,
instantly portable to any site with an hourly wind series, and it quantifies
how much accuracy the "black box" models actually buy.

Evaluated under the identical rolling-origin protocol so the comparison with
results/metrics.csv is apples-to-apples.

Run from project root: python src/transparent_model.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_hourly  # noqa: E402
from make_dataset import HORIZONS, build_supervised  # noqa: E402
from metrics_utils import scores  # noqa: E402
from rolling_eval import rolling_origin_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALPHA = 1.0
N_LAGS = 24


def fit_equation(X: pd.DataFrame, Y: pd.DataFrame, alpha: float = ALPHA):
    """Fit one ridge model per lead step. Returns (intercepts, weights).

    weights has shape (H, n_lags): row h = the coefficients for step h+1.
    """
    H = Y.shape[1]
    W = np.zeros((H, X.shape[1]))
    b = np.zeros(H)
    for j in range(H):
        m = Ridge(alpha=alpha).fit(X.values, Y.values[:, j])
        W[j], b[j] = m.coef_, m.intercept_
    return b, W


def predict_equation(X: pd.DataFrame, b: np.ndarray, W: np.ndarray):
    """Apply the fitted equation: yhat[:, j] = b_j + X @ W_j."""
    return X.values @ W.T + b


def format_equation(b: np.ndarray, W: np.ndarray, step: int,
                    n_terms: int = 5) -> str:
    """Human-readable equation for one lead step (largest terms first)."""
    j = step - 1
    order = np.argsort(-np.abs(W[j]))[:n_terms]
    terms = " ".join(
        f"{'+' if W[j, k] >= 0 else '-'} {abs(W[j, k]):.3f}*ws100(t-{k}h)"
        for k in sorted(order))
    return f"ws100(t+{step}h) = {b[j]:+.3f} {terms} + ..."


def evaluate(horizon: int, df: pd.DataFrame) -> dict:
    """Rolling-origin evaluation of the linear equation at one horizon."""
    X, Y, _ = build_supervised(df, horizon=horizon, n_lags=N_LAGS,
                               features="lags_only")
    per_fold = []
    last = None
    for fold in rolling_origin_folds(X, Y, horizon=horizon):
        b, W = fit_equation(*fold["train"])
        pred = predict_equation(fold["test"][0], b, W)
        per_fold.append(scores(fold["test"][1].values, pred))
        last = (b, W)
    arr = {k: np.array([f[k] for f in per_fold]) for k in per_fold[0]}
    return {"horizon": horizon,
            "rmse": arr["rmse"].mean(), "rmse_sd": arr["rmse"].std(ddof=1),
            "mae": arr["mae"].mean(), "nrmse": arr["nrmse"].mean(),
            "r": arr["r"].mean(), "coef": last}


def main() -> None:
    """Fit, evaluate and export the transparent equation for all horizons."""
    df = load_hourly()
    rows, coefs = [], {}
    for H in HORIZONS:
        res = evaluate(H, df)
        b, W = res.pop("coef")
        coefs[H] = (b, W)
        rows.append(res)
    out = pd.DataFrame(rows)

    # compare against the study's best model at each horizon
    m = pd.read_csv(RESULTS / "metrics.csv")
    best = (m[(m.split == "test") & (m.fold == "mean")]
            .sort_values("rmse").groupby("horizon").first())
    out["best_model"] = out.horizon.map(
        lambda h: f"{best.loc[h, 'model']} ({best.loc[h, 'features']})")
    out["best_rmse"] = out.horizon.map(lambda h: best.loc[h, "rmse"])
    out["gap_%"] = ((out.rmse - out.best_rmse) / out.best_rmse * 100).round(1)
    out.to_csv(RESULTS / "transparent_equation_metrics.csv", index=False)

    print("Transparent linear equation vs the study's best model "
          "(rolling-origin, 4 folds):\n")
    print(out[["horizon", "rmse", "rmse_sd", "nrmse", "r",
               "best_model", "best_rmse", "gap_%"]]
          .round(3).to_string(index=False))

    # export coefficients of the 24 h equation for use elsewhere
    b, W = coefs[24]
    eq = pd.DataFrame(W, columns=[f"ws100(t-{k}h)" for k in range(N_LAGS)])
    eq.insert(0, "intercept", b)
    eq.insert(0, "lead_step_h", np.arange(1, 25))
    eq.to_csv(RESULTS / "equation_coefficients_h24.csv", index=False)
    print(f"\nH=24 equation coefficients -> "
          f"results/equation_coefficients_h24.csv")
    print("\nExample rows of the H=24 equation (5 largest terms shown):")
    for step in (1, 6, 12, 24):
        print("  " + format_equation(b, W, step))


if __name__ == "__main__":
    main()
