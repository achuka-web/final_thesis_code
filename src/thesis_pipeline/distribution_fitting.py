"""Reusable distribution fitting helpers for the thesis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    jarque_bera,
    kstest,
    laplace,
    logistic,
    norm,
    shapiro,
    t as student_t,
)


DISTRIBUTIONS: dict[str, Any] = {
    "Normal": norm,
    "Student-t": student_t,
    "Laplace": laplace,
    "Logistic": logistic,
}


def _to_clean_series(series: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Convert input to numeric pandas Series and drop NaNs safely."""
    cleaned = pd.to_numeric(pd.Series(series), errors="coerce")
    return cleaned.dropna().reset_index(drop=True)


def normality_tests(
    series: pd.Series | np.ndarray | list[float],
    asset_name: str,
) -> pd.DataFrame:
    """Return thesis-ready normality test results."""
    s = _to_clean_series(series)
    rows: list[dict[str, object]] = []

    if len(s) == 0:
        for test_name in ["Jarque-Bera", "Shapiro-Wilk", "Kolmogorov-Smirnov"]:
            rows.append(
                {
                    "Asset": asset_name,
                    "Test": test_name,
                    "Statistic": np.nan,
                    "p-value": np.nan,
                    "Conclusion": "Insufficient data",
                }
            )
        return pd.DataFrame(rows)

    jb_stat, jb_p = jarque_bera(s)
    rows.append(
        {
            "Asset": asset_name,
            "Test": "Jarque-Bera",
            "Statistic": float(jb_stat),
            "p-value": float(jb_p),
            "Conclusion": "Non-normal" if jb_p < 0.05 else "Normal",
        }
    )

    if len(s) >= 3:
        sample = s.sample(min(5000, len(s)), random_state=42)
        sw_stat, sw_p = shapiro(sample)
    else:
        sw_stat, sw_p = np.nan, np.nan
    rows.append(
        {
            "Asset": asset_name,
            "Test": "Shapiro-Wilk",
            "Statistic": float(sw_stat) if pd.notna(sw_stat) else np.nan,
            "p-value": float(sw_p) if pd.notna(sw_p) else np.nan,
            "Conclusion": (
                "Non-normal"
                if pd.notna(sw_p) and sw_p < 0.05
                else ("Normal" if pd.notna(sw_p) else "Insufficient data")
            ),
        }
    )

    mean = s.mean()
    std = s.std(ddof=1)
    if pd.isna(std) or std == 0:
        ks_stat, ks_p = np.nan, np.nan
    else:
        ks_stat, ks_p = kstest(s, "norm", args=(mean, std))
    rows.append(
        {
            "Asset": asset_name,
            "Test": "Kolmogorov-Smirnov",
            "Statistic": float(ks_stat) if pd.notna(ks_stat) else np.nan,
            "p-value": float(ks_p) if pd.notna(ks_p) else np.nan,
            "Conclusion": (
                "Non-normal"
                if pd.notna(ks_p) and ks_p < 0.05
                else ("Normal" if pd.notna(ks_p) else "Insufficient data")
            ),
        }
    )

    return pd.DataFrame(rows)


def fit_distributions(
    series: pd.Series | np.ndarray | list[float],
) -> dict[str, dict[str, Any]]:
    """Fit candidate distributions and return fit diagnostics."""
    s = _to_clean_series(series)
    x = s.to_numpy()
    results: dict[str, dict[str, Any]] = {}

    if len(x) == 0:
        return results

    n = len(x)
    for name, dist in DISTRIBUTIONS.items():
        params = dist.fit(x)
        log_likelihood = float(np.sum(dist.logpdf(x, *params)))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        ks_stat, ks_p = kstest(x, dist.cdf, args=params)

        results[name] = {
            "parameters": params,
            "log_likelihood": log_likelihood,
            "k": int(k),
            "AIC": float(aic),
            "BIC": float(bic),
            "KS statistic": float(ks_stat),
            "KS p-value": float(ks_p),
        }

    return results


def fitting_table(
    fit_results: dict[str, dict[str, Any]],
    asset_name: str,
) -> pd.DataFrame:
    """Return a thesis-ready distribution comparison table."""
    rows: list[dict[str, object]] = []

    if not fit_results:
        return pd.DataFrame(
            columns=[
                "Asset",
                "Distribution",
                "Parameters",
                "Log-likelihood",
                "k",
                "AIC",
                "BIC",
                "KS statistic",
                "KS p-value",
                "Best by AIC",
            ]
        )

    best_aic = min(result["AIC"] for result in fit_results.values())

    for name, result in fit_results.items():
        rows.append(
            {
                "Asset": asset_name,
                "Distribution": name,
                "Parameters": str(tuple(np.round(result["parameters"], 6))),
                "Log-likelihood": result["log_likelihood"],
                "k": result["k"],
                "AIC": result["AIC"],
                "BIC": result["BIC"],
                "KS statistic": result["KS statistic"],
                "KS p-value": result["KS p-value"],
                "Best by AIC": "Yes" if result["AIC"] == best_aic else "",
            }
        )

    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def compare_distribution_fits(
    series: pd.Series | np.ndarray | list[float],
    asset_name: str,
) -> pd.DataFrame:
    """Fit all candidate distributions and return the comparison table."""
    return fitting_table(fit_distributions(series), asset_name=asset_name)


def plot_hist_fit(
    series: pd.Series | np.ndarray | list[float],
    fit_results: dict[str, dict[str, Any]],
    asset_name: str,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Create and save a histogram with fitted distribution curves."""
    s = _to_clean_series(series)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s, bins=60, density=True, alpha=0.4, color="lightgray", edgecolor="black")

    if len(s) > 0 and fit_results:
        x_grid = np.linspace(float(s.min()), float(s.max()), 500)
        for name, result in fit_results.items():
            dist = DISTRIBUTIONS[name]
            params = result["parameters"]
            ax.plot(x_grid, dist.pdf(x_grid, *params), label=name, linewidth=2)

    ax.set_title(f"{asset_name} - Histogram with Fitted Distributions")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = output_path / f"{asset_name.replace('/', '')}_hist_fit.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_qq(
    series: pd.Series | np.ndarray | list[float],
    fit_results: dict[str, dict[str, Any]],
    asset_name: str,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Create and save QQ plots for Normal, Student-t, Laplace, and Logistic."""
    s = _to_clean_series(series)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    if len(s) > 0:
        probs = (np.arange(1, len(s) + 1) - 0.5) / len(s)
        sample_quantiles = np.sort(s.to_numpy())

        for ax, name in zip(axes, DISTRIBUTIONS.keys()):
            if name not in fit_results:
                ax.set_visible(False)
                continue

            dist = DISTRIBUTIONS[name]
            params = fit_results[name]["parameters"]
            theoretical_quantiles = dist.ppf(probs, *params)

            ax.scatter(theoretical_quantiles, sample_quantiles, s=10, alpha=0.6)
            min_val = min(np.nanmin(theoretical_quantiles), np.nanmin(sample_quantiles))
            max_val = max(np.nanmax(theoretical_quantiles), np.nanmax(sample_quantiles))
            ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
            ax.set_title(f"{asset_name} - QQ Plot ({name})")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.grid(True, alpha=0.3)
    else:
        for ax in axes:
            ax.set_visible(False)

    save_path = output_path / f"{asset_name.replace('/', '')}_qq_plot.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_aic_bic(
    fit_table: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Create and save an AIC/BIC comparison plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if not fit_table.empty:
        labels = fit_table["Distribution"].astype(str)
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, fit_table["AIC"], width=width, label="AIC")
        ax.bar(x + width / 2, fit_table["BIC"], width=width, label="BIC")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)

    ax.set_title("Distribution Fit Comparison: AIC vs BIC")
    ax.set_ylabel("Information Criterion")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    save_path = output_path / "aic_bic_comparison.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
