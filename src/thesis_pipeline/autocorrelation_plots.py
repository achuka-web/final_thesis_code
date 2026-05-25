"""Reusable autocorrelation plotting helpers for the thesis pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def _to_clean_series(series) -> pd.Series:
    """Convert input to numeric pandas Series and drop NaNs safely."""
    cleaned = pd.to_numeric(pd.Series(series), errors="coerce")
    return cleaned.dropna().reset_index(drop=True)


def plot_return_acf_pacf(
    series,
    asset_name,
    lags: int = 40,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Create and save ACF/PACF plots for log returns."""
    s = _to_clean_series(series)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(s, lags=lags, ax=axes[0], alpha=0.05)
    plot_pacf(s, lags=lags, ax=axes[1], alpha=0.05, method="ywm")

    axes[0].set_title(f"{asset_name} - ACF of Log Returns")
    axes[1].set_title(f"{asset_name} - PACF of Log Returns")

    fig.tight_layout()
    save_path = output_path / f"{asset_name.replace('/', '')}_return_acf_pacf.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_squared_return_acf_pacf(
    series,
    asset_name,
    lags: int = 40,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Create and save ACF/PACF plots for squared log returns."""
    s = _to_clean_series(series)
    squared = s**2
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(squared, lags=lags, ax=axes[0], alpha=0.05)
    plot_pacf(squared, lags=lags, ax=axes[1], alpha=0.05, method="ywm")

    axes[0].set_title(f"{asset_name} - ACF of Squared Log Returns")
    axes[1].set_title(f"{asset_name} - PACF of Squared Log Returns")

    fig.tight_layout()
    save_path = output_path / f"{asset_name.replace('/', '')}_squared_return_acf_pacf.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_standardized_residual_acf(
    fitted_model,
    asset_name,
    lags: int = 40,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Create and save ACF plots for standardized residuals and their squares."""
    std_resid = pd.to_numeric(pd.Series(fitted_model.std_resid), errors="coerce").dropna().reset_index(drop=True)
    squared_std_resid = std_resid**2
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(std_resid, lags=lags, ax=axes[0], alpha=0.05)
    plot_acf(squared_std_resid, lags=lags, ax=axes[1], alpha=0.05)

    axes[0].set_title(f"{asset_name} - ACF of Standardized Residuals")
    axes[1].set_title(f"{asset_name} - ACF of Squared Standardized Residuals")

    fig.tight_layout()
    save_path = output_path / f"{asset_name.replace('/', '')}_standardized_residual_acf.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_all_autocorrelation_figures(
    returns_series,
    fitted_model,
    asset_name,
    lags: int = 40,
    output_dir: str | Path = "outputs/figures",
) -> dict[str, Path]:
    """Generate all autocorrelation figures for a thesis asset workflow."""
    return {
        "return_acf_pacf": plot_return_acf_pacf(
            series=returns_series,
            asset_name=asset_name,
            lags=lags,
            output_dir=output_dir,
        ),
        "squared_return_acf_pacf": plot_squared_return_acf_pacf(
            series=returns_series,
            asset_name=asset_name,
            lags=lags,
            output_dir=output_dir,
        ),
        "standardized_residual_acf": plot_standardized_residual_acf(
            fitted_model=fitted_model,
            asset_name=asset_name,
            lags=lags,
            output_dir=output_dir,
        ),
    }
