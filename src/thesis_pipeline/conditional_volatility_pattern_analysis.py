"""Pattern analysis conditioned on GARCH volatility regimes."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, levene, mannwhitneyu, ttest_ind


MIN_GROUP_SIZE = 5


def _to_clean_series(series: pd.Series) -> pd.Series:
    """Convert input to numeric pandas Series and drop NaNs safely."""
    cleaned = pd.to_numeric(series, errors="coerce")
    return cleaned.dropna()


def _safe_summary(series: pd.Series) -> dict[str, float]:
    """Return group summary statistics with NaNs for empty inputs."""
    s = _to_clean_series(series)
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan}

    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
    }


def _safe_welch_t(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Welch t-test with a minimum sample-size rule."""
    x_clean = _to_clean_series(x)
    y_clean = _to_clean_series(y)
    if len(x_clean) < MIN_GROUP_SIZE or len(y_clean) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = ttest_ind(x_clean, y_clean, equal_var=False, nan_policy="omit")
    return float(stat), float(pvalue)


def _safe_mann_whitney(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test with a minimum sample-size rule."""
    x_clean = _to_clean_series(x)
    y_clean = _to_clean_series(y)
    if len(x_clean) < MIN_GROUP_SIZE or len(y_clean) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = mannwhitneyu(x_clean, y_clean, alternative="two-sided")
    return float(stat), float(pvalue)


def _safe_levene(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Levene variance test with a minimum sample-size rule."""
    x_clean = _to_clean_series(x)
    y_clean = _to_clean_series(y)
    if len(x_clean) < MIN_GROUP_SIZE or len(y_clean) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = levene(x_clean, y_clean, center="median")
    return float(stat), float(pvalue)


def _safe_ks(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Two-sample KS test with a minimum sample-size rule."""
    x_clean = _to_clean_series(x)
    y_clean = _to_clean_series(y)
    if len(x_clean) < MIN_GROUP_SIZE or len(y_clean) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = ks_2samp(x_clean, y_clean, alternative="two-sided", mode="auto")
    return float(stat), float(pvalue)


def add_garch_volatility_regime(
    df: pd.DataFrame,
    volatility_col: str = "garch_volatility",
) -> pd.DataFrame:
    """Add high/low volatility regime columns from a conditional volatility series."""
    data = df.copy()

    if volatility_col not in data.columns:
        raise KeyError(f"Missing volatility column: {volatility_col}")

    vol = pd.to_numeric(data[volatility_col], errors="coerce")
    median_vol = vol.dropna().median()

    data[volatility_col] = vol
    if pd.isna(median_vol):
        data["high_volatility"] = pd.Series(0, index=data.index, dtype=int)
        data["low_volatility"] = pd.Series(0, index=data.index, dtype=int)
        return data

    data["high_volatility"] = ((vol >= median_vol) & vol.notna()).astype(int)
    data["low_volatility"] = ((vol < median_vol) & vol.notna()).astype(int)
    return data


def build_volatility_conditioned_table(
    df: pd.DataFrame,
    asset_name: str,
    pattern_cols: Iterable[str],
    horizons: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Build a thesis-ready table comparing high- and low-volatility pattern returns."""
    rows: list[dict[str, object]] = []

    for pattern_col in pattern_cols:
        if pattern_col not in df.columns:
            continue

        for h in horizons:
            ret_col = f"fwd_ret_{h}"
            if ret_col not in df.columns:
                continue

            temp = df.loc[df[pattern_col] == 1].copy()
            temp[ret_col] = pd.to_numeric(temp[ret_col], errors="coerce")

            high = temp.loc[temp["high_volatility"] == 1, ret_col].dropna()
            low = temp.loc[temp["low_volatility"] == 1, ret_col].dropna()

            high_stats = _safe_summary(high)
            low_stats = _safe_summary(low)

            _, welch_p = _safe_welch_t(high, low)
            _, mwu_p = _safe_mann_whitney(high, low)
            _, levene_p = _safe_levene(high, low)
            _, ks_p = _safe_ks(high, low)

            rows.append(
                {
                    "Asset": asset_name,
                    "Pattern": pattern_col,
                    "h": int(h),
                    "High N": high_stats["n"],
                    "Low N": low_stats["n"],
                    "High Mean": high_stats["mean"],
                    "Low Mean": low_stats["mean"],
                    "High Std": high_stats["std"],
                    "Low Std": low_stats["std"],
                    "Welch t-test p-value": welch_p,
                    "Mann-Whitney p-value": mwu_p,
                    "Levene p-value": levene_p,
                    "KS p-value": ks_p,
                }
            )

    return pd.DataFrame(rows).sort_values(["Asset", "Pattern", "h"]).reset_index(drop=True)
