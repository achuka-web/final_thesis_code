"""Reusable GARCH-family model helpers for the thesis pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox


def _to_clean_series(series: pd.Series) -> pd.Series:
    """Convert input to numeric pandas Series and drop NaNs safely."""
    cleaned = pd.to_numeric(series, errors="coerce")
    return cleaned.dropna()


def _prepare_garch_input(series: pd.Series) -> pd.Series:
    """Prepare returns for ARCH-family estimation."""
    cleaned = _to_clean_series(series)
    return cleaned * 100.0


def fit_garch(
    series: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
) -> Any:
    """Fit a GARCH(p, q) model using the arch package."""
    s = _prepare_garch_input(series)
    if len(s) == 0:
        raise ValueError("Series is empty after NaN cleaning.")

    model = arch_model(s, vol="GARCH", p=p, q=q, dist=dist)
    return model.fit(disp="off")


def fit_egarch(
    series: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
) -> Any:
    """Fit an EGARCH(p, q) model using the arch package."""
    s = _prepare_garch_input(series)
    if len(s) == 0:
        raise ValueError("Series is empty after NaN cleaning.")

    model = arch_model(s, vol="EGARCH", p=p, q=q, o=o, dist=dist)
    return model.fit(disp="off")


def fit_tgarch(
    series: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
) -> Any:
    """Fit a TGARCH/GJR-GARCH(p, q) model using the arch package."""
    s = _prepare_garch_input(series)
    if len(s) == 0:
        raise ValueError("Series is empty after NaN cleaning.")

    model = arch_model(s, vol="GARCH", p=p, q=q, o=o, dist=dist)
    return model.fit(disp="off")


def extract_conditional_volatility(result: Any) -> pd.Series:
    """Extract conditional volatility and scale back to return units."""
    if result is None or not hasattr(result, "conditional_volatility"):
        return pd.Series(dtype=float)

    vol = pd.Series(result.conditional_volatility, copy=True)
    vol = pd.to_numeric(vol, errors="coerce") / 100.0
    return vol.dropna()


def model_diagnostics(result: Any) -> pd.DataFrame:
    """Return thesis-ready model diagnostics."""
    if result is None:
        return pd.DataFrame(
            [
                {
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Log-likelihood": np.nan,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "AIC": float(getattr(result, "aic", np.nan)),
                "BIC": float(getattr(result, "bic", np.nan)),
                "Log-likelihood": float(getattr(result, "loglikelihood", np.nan)),
            }
        ]
    )


def print_model_summary(
    result: Any,
    asset_name: str,
    model_name: str,
) -> None:
    """Print a thesis-style fitted model summary."""
    print("=" * 60)
    print(f"{model_name} - {asset_name}")
    print("=" * 60)
    print(result.summary())


def garch_parameter_table(
    result: Any,
    asset_name: str,
    model_name: str,
) -> pd.DataFrame:
    """Return a thesis-ready parameter significance table."""
    if result is None:
        return pd.DataFrame(
            columns=[
                "Asset",
                "Model",
                "Parameter",
                "Coefficient",
                "p-value",
                "Significance at 5%",
            ]
        )

    params = getattr(result, "params", pd.Series(dtype=float))
    pvalues = getattr(result, "pvalues", pd.Series(dtype=float))

    rows = []
    for parameter in params.index:
        pvalue = pvalues.get(parameter, np.nan)
        rows.append(
            {
                "Asset": asset_name,
                "Model": model_name,
                "Parameter": parameter,
                "Coefficient": float(params[parameter]),
                "p-value": float(pvalue) if pd.notna(pvalue) else np.nan,
                "Significance at 5%": (
                    "Significant" if pd.notna(pvalue) and pvalue < 0.05 else "Not significant"
                ),
            }
        )

    return pd.DataFrame(rows)


def standardized_residual_tests(
    result: Any,
    lags: tuple[int, ...] = (10, 20, 30, 40),
) -> pd.DataFrame:
    """Run Ljung-Box tests on standardized residuals and their squares."""
    if result is None or not hasattr(result, "std_resid"):
        rows = []
        for lag in lags:
            rows.append(
                {
                    "Test": "Ljung-Box Std Residuals",
                    "Lag": int(lag),
                    "Statistic": np.nan,
                    "p-value": np.nan,
                }
            )
            rows.append(
                {
                    "Test": "Ljung-Box Squared Std Residuals",
                    "Lag": int(lag),
                    "Statistic": np.nan,
                    "p-value": np.nan,
                }
            )
        return pd.DataFrame(rows)

    std_resid = pd.to_numeric(pd.Series(result.std_resid), errors="coerce").dropna()
    lag_list = list(lags)

    if len(std_resid) <= max(lag_list):
        rows = []
        for lag in lag_list:
            rows.append(
                {
                    "Test": "Ljung-Box Std Residuals",
                    "Lag": int(lag),
                    "Statistic": np.nan,
                    "p-value": np.nan,
                }
            )
            rows.append(
                {
                    "Test": "Ljung-Box Squared Std Residuals",
                    "Lag": int(lag),
                    "Statistic": np.nan,
                    "p-value": np.nan,
                }
            )
        return pd.DataFrame(rows)

    lb_std = acorr_ljungbox(std_resid, lags=lag_list, return_df=True)
    lb_sq = acorr_ljungbox(std_resid**2, lags=lag_list, return_df=True)

    rows = []
    for lag in lag_list:
        rows.append(
            {
                "Test": "Ljung-Box Std Residuals",
                "Lag": int(lag),
                "Statistic": float(lb_std.loc[lag, "lb_stat"]),
                "p-value": float(lb_std.loc[lag, "lb_pvalue"]),
            }
        )
        rows.append(
            {
                "Test": "Ljung-Box Squared Std Residuals",
                "Lag": int(lag),
                "Statistic": float(lb_sq.loc[lag, "lb_stat"]),
                "p-value": float(lb_sq.loc[lag, "lb_pvalue"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_garch_models(results_dict: dict[str, Any]) -> pd.DataFrame:
    """Summarize fitted GARCH-family models in a thesis-ready table."""
    rows = []

    for model_name, result in results_dict.items():
        if result is None:
            rows.append(
                {
                    "Model": model_name,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Log-likelihood": np.nan,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "gamma": np.nan,
                    "persistence": np.nan,
                }
            )
            continue

        params = getattr(result, "params", pd.Series(dtype=float))
        alpha = params.get("alpha[1]", np.nan)
        beta = params.get("beta[1]", np.nan)
        gamma = params.get("gamma[1]", np.nan)
        persistence = alpha + beta if pd.notna(alpha) and pd.notna(beta) else np.nan

        rows.append(
            {
                "Model": model_name,
                "AIC": float(getattr(result, "aic", np.nan)),
                "BIC": float(getattr(result, "bic", np.nan)),
                "Log-likelihood": float(getattr(result, "loglikelihood", np.nan)),
                "alpha": float(alpha) if pd.notna(alpha) else np.nan,
                "beta": float(beta) if pd.notna(beta) else np.nan,
                "gamma": float(gamma) if pd.notna(gamma) else np.nan,
                "persistence": float(persistence) if pd.notna(persistence) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compare_garch_models(
    series: pd.Series,
    asset_name: str,
) -> pd.DataFrame:
    """Fit and compare thesis GARCH-family models."""
    results_dict = {
        "GARCH(1,1)": fit_garch(series, p=1, q=1, dist="t"),
        "EGARCH(1,1,1)": fit_egarch(series, p=1, o=1, q=1, dist="t"),
        "TGARCH(1,1,1)": fit_tgarch(series, p=1, o=1, q=1, dist="t"),
    }

    summary = summarize_garch_models(results_dict)
    summary.insert(0, "Asset", asset_name)
    return summary
