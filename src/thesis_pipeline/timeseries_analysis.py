"""Reusable time-series analysis functions for the thesis pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def _to_clean_series(series: pd.Series) -> pd.Series:
    """Convert input to numeric pandas Series and drop NaNs safely."""
    cleaned = pd.to_numeric(series, errors="coerce")
    return cleaned.dropna()


def compute_log_returns(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute log returns from a price column and return a new DataFrame."""
    data = df.copy()

    if price_col not in data.columns:
        raise KeyError(f"Missing price column: {price_col}")

    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    invalid_mask = data[price_col] <= 0
    data.loc[invalid_mask, price_col] = np.nan

    data["log_return"] = np.log(data[price_col] / data[price_col].shift(1))
    return data


def descriptive_statistics(series: pd.Series) -> pd.DataFrame:
    """Return thesis-ready descriptive statistics as a one-row DataFrame."""
    s = _to_clean_series(series)

    result = {
        "N": int(len(s)),
        "Mean": float(s.mean()) if len(s) > 0 else np.nan,
        "Std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "Skewness": float(skew(s, bias=False)) if len(s) > 2 else np.nan,
        "Kurtosis": float(kurtosis(s, fisher=False, bias=False)) if len(s) > 3 else np.nan,
        "Min": float(s.min()) if len(s) > 0 else np.nan,
        "Max": float(s.max()) if len(s) > 0 else np.nan,
    }

    return pd.DataFrame([result])


def jarque_bera_test(series: pd.Series) -> pd.DataFrame:
    """Run the Jarque-Bera normality test and return a one-row DataFrame."""
    s = _to_clean_series(series)

    if len(s) < 3:
        result = {
            "Test": "Jarque-Bera",
            "Statistic": np.nan,
            "p-value": np.nan,
        }
        return pd.DataFrame([result])

    stat, pvalue = jarque_bera(s)
    result = {
        "Test": "Jarque-Bera",
        "Statistic": float(stat),
        "p-value": float(pvalue),
    }
    return pd.DataFrame([result])


def adf_test(series: pd.Series) -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test and return a one-row DataFrame."""
    s = _to_clean_series(series)

    if len(s) < 10:
        result = {
            "Test": "ADF",
            "Statistic": np.nan,
            "p-value": np.nan,
            "Used Lag": np.nan,
            "N Obs": int(len(s)),
        }
        return pd.DataFrame([result])

    stat, pvalue, usedlag, nobs, _, _ = adfuller(s, autolag="AIC")
    result = {
        "Test": "ADF",
        "Statistic": float(stat),
        "p-value": float(pvalue),
        "Used Lag": int(usedlag),
        "N Obs": int(nobs),
    }
    return pd.DataFrame([result])


def ljung_box_test(
    series: pd.Series,
    lags: int = 20,
) -> pd.DataFrame:
    s = _to_clean_series(series)

    if len(s) <= lags:
        result = {
            "Test": "Ljung-Box",
            "Lag": int(lags),
            "Statistic": np.nan,
            "p-value": np.nan,
        }
        return pd.DataFrame([result])

    lb = acorr_ljungbox(s, lags=[lags], return_df=True)
    result = {
        "Test": "Ljung-Box",
        "Lag": int(lags),
        "Statistic": float(lb.iloc[0]["lb_stat"]),
        "p-value": float(lb.iloc[0]["lb_pvalue"]),
    }
    return pd.DataFrame([result])


def ljung_box_squared_test(
    series: pd.Series,
    lags: int = 20,
) -> pd.DataFrame:
    """Run the Ljung-Box test on squared series values."""
    s = _to_clean_series(series)
    squared = s**2

    if len(squared) <= lags:
        result = {
            "Test": "Ljung-Box Squared",
            "Lag": int(lags),
            "Statistic": np.nan,
            "p-value": np.nan,
        }
        return pd.DataFrame([result])

    lb = acorr_ljungbox(squared, lags=[lags], return_df=True)
    result = {
        "Test": "Ljung-Box Squared",
        "Lag": int(lags),
        "Statistic": float(lb.iloc[0]["lb_stat"]),
        "p-value": float(lb.iloc[0]["lb_pvalue"]),
    }
    return pd.DataFrame([result])


def arch_lm_test(
    series: pd.Series,
    lags: int = 20,
) -> pd.DataFrame:
    """Run the ARCH-LM test and return a one-row DataFrame."""
    s = _to_clean_series(series)

    if len(s) <= lags + 1:
        result = {
            "Test": "ARCH-LM",
            "Lag": int(lags),
            "LM Statistic": np.nan,
            "LM p-value": np.nan,
            "F Statistic": np.nan,
            "F p-value": np.nan,
        }
        return pd.DataFrame([result])

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(s, nlags=lags)
    result = {
        "Test": "ARCH-LM",
        "Lag": int(lags),
        "LM Statistic": float(lm_stat),
        "LM p-value": float(lm_pvalue),
        "F Statistic": float(f_stat),
        "F p-value": float(f_pvalue),
    }
    return pd.DataFrame([result])


def summarize_timeseries_tests(series: pd.Series) -> pd.DataFrame:
    """Return a thesis-ready combined summary table of core time-series tests."""
    parts = [
        jarque_bera_test(series),
        adf_test(series),
        ljung_box_test(series),
        ljung_box_squared_test(series),
        arch_lm_test(series),
    ]

    return pd.concat(parts, ignore_index=True, sort=False)
