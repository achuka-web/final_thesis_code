"""Pattern-level statistical tests and thesis-ready result tables.

Runner example
--------------
```python
from thesis_pipeline.pattern_tests import (
    add_forward_returns,
    build_pattern_significance_table,
    build_volume_conditioned_table,
    export_table_csv,
    export_table_latex,
)

PATTERN_COLS = [
    "hammer",
    "hanging_man",
    "bullish_engulfing",
    "bearish_engulfing",
    "morning_star",
    "evening_star",
    "doji",
    "spinning_top",
]

df_xau = add_forward_returns(df_xau)

pattern_table_xau = build_pattern_significance_table(
    df=df_xau,
    asset_name="XAU/USD",
    pattern_cols=PATTERN_COLS,
    horizons=(1, 3, 5),
)

volume_table_xau = build_volume_conditioned_table(
    df=df_xau,
    asset_name="XAU/USD",
    pattern_cols=PATTERN_COLS,
    horizons=(1, 3, 5),
    rv_col="RV",
)

export_table_csv(pattern_table_xau, "outputs/tables/xau_pattern_significance.csv")
export_table_latex(pattern_table_xau, "outputs/tables/xau_pattern_significance.tex")

export_table_csv(volume_table_xau, "outputs/tables/xau_volume_conditioned.csv")
export_table_latex(volume_table_xau, "outputs/tables/xau_volume_conditioned.tex")
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, levene, mannwhitneyu, ttest_1samp, ttest_ind


MIN_GROUP_SIZE = 5


def add_forward_returns(
    df: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5),
    close_col: str = "Close",
) -> pd.DataFrame:
    """Add forward log returns for the requested horizons."""
    data = df.copy()

    for h in horizons:
        data[f"fwd_ret_{h}"] = np.log(data[close_col].shift(-h) / data[close_col])

    return data


def process_volume(
    df: pd.DataFrame,
    volume_col: str = "Volume",
    window: int = 24,
) -> pd.DataFrame:
    """Create relative volume and binary high/low volume regime columns."""
    data = df.copy()

    if volume_col not in data.columns:
        raise KeyError(f"Missing volume column: {volume_col}")

    volume = pd.to_numeric(data[volume_col], errors="coerce")
    rolling_mean = volume.rolling(window=window).mean()

    data["RV"] = volume / rolling_mean

    rv_median = data["RV"].dropna().median()
    if pd.isna(rv_median):
        data["high_volume"] = pd.Series(0, index=data.index, dtype=int)
        data["low_volume"] = pd.Series(0, index=data.index, dtype=int)
        return data

    data["high_volume"] = (
        (data["RV"] >= rv_median) & data["RV"].notna()
    ).astype(int)
    data["low_volume"] = (
        (data["RV"] < rv_median) & data["RV"].notna()
    ).astype(int)

    return data


def get_pattern_vs_nonpattern_returns(
    df: pd.DataFrame,
    pattern_col: str,
    h: int,
) -> tuple[pd.Series, pd.Series]:
    """Return forward returns for pattern and non-pattern periods."""
    ret_col = f"fwd_ret_{h}"

    pattern_returns = df.loc[df[pattern_col] == 1, ret_col].dropna()
    nonpattern_returns = df.loc[df[pattern_col] == 0, ret_col].dropna()

    return pattern_returns, nonpattern_returns


def get_volume_conditioned_returns(
    df: pd.DataFrame,
    pattern_col: str,
    h: int,
    rv_col: str = "RV",
) -> tuple[pd.Series, pd.Series]:
    """Return forward returns for high- and low-volume pattern periods."""
    ret_col = f"fwd_ret_{h}"

    high_returns = df.loc[
        (df[pattern_col] == 1) & (df[rv_col] > 1),
        ret_col,
    ].dropna()
    low_returns = df.loc[
        (df[pattern_col] == 1) & (df[rv_col] <= 1),
        ret_col,
    ].dropna()

    return high_returns, low_returns


def _safe_summary(series: pd.Series) -> dict[str, float]:
    """Return summary statistics with NaNs for empty inputs."""
    if len(series) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "positive_ratio": np.nan,
        }

    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
        "median": float(series.median()),
        "positive_ratio": float((series > 0).mean()),
    }


def _safe_one_sample_t(series: pd.Series) -> tuple[float, float]:
    """One-sample t-test against zero mean."""
    if len(series) < 2:
        return np.nan, np.nan

    stat, pvalue = ttest_1samp(series, popmean=0.0, nan_policy="omit")
    return float(stat), float(pvalue)


def _safe_welch_t(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Welch two-sample t-test."""
    if len(x) < MIN_GROUP_SIZE or len(y) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(stat), float(pvalue)


def _safe_mann_whitney(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test."""
    if len(x) < MIN_GROUP_SIZE or len(y) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = mannwhitneyu(x, y, alternative="two-sided")
    return float(stat), float(pvalue)


def _safe_levene(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Levene variance equality test using median center."""
    if len(x) < MIN_GROUP_SIZE or len(y) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = levene(x, y, center="median")
    return float(stat), float(pvalue)


def _safe_ks(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    
    """Two-sample Kolmogorov-Smirnov test."""
    if len(x) < MIN_GROUP_SIZE or len(y) < MIN_GROUP_SIZE:
        return np.nan, np.nan

    stat, pvalue = ks_2samp(x, y, alternative="two-sided", mode="auto")
    return float(stat), float(pvalue)


def build_pattern_significance_row(
    df: pd.DataFrame,
    asset_name: str,
    pattern_col: str,
    h: int,
) -> dict[str, object]:
    """Build one row for the thesis pattern significance table."""
    pattern_returns, nonpattern_returns = get_pattern_vs_nonpattern_returns(
        df=df,
        pattern_col=pattern_col,
        h=h,
    )

    pattern_stats = _safe_summary(pattern_returns)
    nonpattern_stats = _safe_summary(nonpattern_returns)

    _, t_p = _safe_one_sample_t(pattern_returns)
    _, mwu_p = _safe_mann_whitney(pattern_returns, nonpattern_returns)
    _, levene_p = _safe_levene(pattern_returns, nonpattern_returns)
    _, ks_p = _safe_ks(pattern_returns, nonpattern_returns)

    return {
        "Asset": asset_name,
        "Pattern": pattern_col,
        "h": h,
        "N_pattern": pattern_stats["n"],
        "N_nonpattern": nonpattern_stats["n"],
        "Mean_pattern": pattern_stats["mean"],
        "Std_pattern": pattern_stats["std"],
        "Median_pattern": pattern_stats["median"],
        "Positive_ratio": pattern_stats["positive_ratio"],
        "t_test_pvalue_vs_0": t_p,
        "Mann_Whitney_pvalue_pattern_vs_nonpattern": mwu_p,
        "Levene_pvalue_pattern_vs_nonpattern": levene_p,
        "KS_pvalue_pattern_vs_nonpattern": ks_p,
    }


def build_pattern_significance_table(
    df: pd.DataFrame,
    asset_name: str,
    pattern_cols: Iterable[str],
    horizons: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Create the final thesis pattern significance table."""
    rows: list[dict[str, object]] = []

    for pattern_col in pattern_cols:
        if pattern_col not in df.columns:
            continue

        for h in horizons:
            rows.append(
                build_pattern_significance_row(
                    df=df,
                    asset_name=asset_name,
                    pattern_col=pattern_col,
                    h=h,
                )
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(["Asset", "Pattern", "h"]).reset_index(drop=True)


def build_volume_conditioned_row(
    df: pd.DataFrame,
    asset_name: str,
    pattern_col: str,
    h: int,
    rv_col: str = "RV",
) -> dict[str, object]:
    """Build one row for the thesis volume-conditioned table."""
    high_returns, low_returns = get_volume_conditioned_returns(
        df=df,
        pattern_col=pattern_col,
        h=h,
        rv_col=rv_col,
    )

    high_stats = _safe_summary(high_returns)
    low_stats = _safe_summary(low_returns)

    _, welch_p = _safe_welch_t(high_returns, low_returns)
    _, mwu_p = _safe_mann_whitney(high_returns, low_returns)
    _, levene_p = _safe_levene(high_returns, low_returns)
    _, ks_p = _safe_ks(high_returns, low_returns)

    return {
        "Asset": asset_name,
        "Pattern": pattern_col,
        "h": h,
        "N_high": high_stats["n"],
        "Mean_high": high_stats["mean"],
        "Std_high": high_stats["std"],
        "Median_high": high_stats["median"],
        "Positive_ratio_high": high_stats["positive_ratio"],
        "N_low": low_stats["n"],
        "Mean_low": low_stats["mean"],
        "Std_low": low_stats["std"],
        "Median_low": low_stats["median"],
        "Positive_ratio_low": low_stats["positive_ratio"],
        "Welch_ttest_pvalue_high_vs_low": welch_p,
        "Mann_Whitney_pvalue_high_vs_low": mwu_p,
        "Levene_pvalue_high_vs_low": levene_p,
        "KS_pvalue_high_vs_low": ks_p,
    }


def build_volume_conditioned_table(
    df: pd.DataFrame,
    asset_name: str,
    pattern_cols: Iterable[str],
    horizons: Iterable[int] = (1, 3, 5),
    rv_col: str = "RV",
) -> pd.DataFrame:
    """Create the final thesis volume-conditioned table."""
    rows: list[dict[str, object]] = []

    for pattern_col in pattern_cols:
        if pattern_col not in df.columns:
            continue

        for h in horizons:
            rows.append(
                build_volume_conditioned_row(
                    df=df,
                    asset_name=asset_name,
                    pattern_col=pattern_col,
                    h=h,
                    rv_col=rv_col,
                )
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(["Asset", "Pattern", "h"]).reset_index(drop=True)


def format_result_table(
    table: pd.DataFrame,
    decimals: int = 6,
) -> pd.DataFrame:
    """Round numeric columns for thesis-ready export."""
    formatted = table.copy()
    numeric_cols = formatted.select_dtypes(include=[np.number]).columns
    formatted[numeric_cols] = formatted[numeric_cols].round(decimals)
    return formatted


def export_table_csv(
    table: pd.DataFrame,
    output_path: str | Path,
    decimals: int = 6,
) -> Path:
    """Export a formatted result table to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    formatted = format_result_table(table, decimals=decimals)
    formatted.to_csv(path, index=False, encoding="utf-8")
    return path


def export_table_latex(
    table: pd.DataFrame,
    output_path: str | Path,
    decimals: int = 6,
) -> Path:
    """Export a formatted result table to LaTeX."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    formatted = format_result_table(table, decimals=decimals)
    latex = formatted.to_latex(index=False, float_format=lambda x: f"{x:.{decimals}f}")
    path.write_text(latex, encoding="utf-8")
    return path
