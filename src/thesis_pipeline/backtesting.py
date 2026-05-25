"""Reusable backtesting helpers for the thesis pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _to_clean_series(series: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Convert input to numeric pandas Series and drop NaNs safely."""
    cleaned = pd.to_numeric(pd.Series(series), errors="coerce")
    return cleaned.dropna().reset_index(drop=True)


def backtest_pattern_strategy(
    df: pd.DataFrame,
    pattern_col: str,
    h: int,
    direction: str,
) -> pd.Series:
    """Backtest a simple pattern strategy using precomputed forward returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the binary pattern column and `fwd_ret_{h}`.
    pattern_col : str
        Name of the candlestick pattern indicator column.
    h : int
        Forward return horizon, matched to `fwd_ret_{h}`.
    direction : str
        Either `"long"` or `"short"`.
    """
    if pattern_col not in df.columns:
        raise KeyError(f"Missing pattern column: {pattern_col}")

    ret_col = f"fwd_ret_{h}"
    if ret_col not in df.columns:
        raise KeyError(f"Missing forward return column: {ret_col}")

    if direction not in {"long", "short"}:
        raise ValueError("direction must be either 'long' or 'short'")

    data = df.copy()
    raw_returns = pd.to_numeric(data[ret_col], errors="coerce")
    signal_mask = pd.to_numeric(data[pattern_col], errors="coerce").fillna(0) == 1

    strategy_returns = raw_returns.loc[signal_mask].dropna().copy()
    if direction == "short":
        strategy_returns = -strategy_returns

    strategy_returns.name = "strategy_return"
    return strategy_returns.reset_index(drop=True)


def performance_metrics(returns: pd.Series | np.ndarray | list[float]) -> pd.DataFrame:
    """Compute thesis-ready performance metrics for a return series."""
    r = _to_clean_series(returns)

    if len(r) == 0:
        return pd.DataFrame(
            [
                {
                    "total_return": np.nan,
                    "mean_return": np.nan,
                    "std_return": np.nan,
                    "sharpe_ratio": np.nan,
                    "max_drawdown": np.nan,
                    "profit_factor": np.nan,
                    "win_rate": np.nan,
                    "n_trades": 0,
                }
            ]
        )

    mean_return = float(r.mean())
    std_return = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    sharpe_ratio = mean_return / std_return if pd.notna(std_return) and std_return != 0 else np.nan

    equity_curve = (1.0 + r).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else np.nan

    gross_profit = float(r[r > 0].sum())
    gross_loss = float(abs(r[r < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    result = {
        "total_return": float(equity_curve.iloc[-1] - 1.0),
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate": float((r > 0).mean()),
        "n_trades": int(len(r)),
    }
    return pd.DataFrame([result])


def apply_transaction_cost(
    returns: pd.Series | np.ndarray | list[float],
    cost: float = 0.0001,
) -> pd.Series:
    """Apply a fixed per-trade transaction cost to returns."""
    r = _to_clean_series(returns)
    if len(r) == 0:
        return pd.Series(dtype=float, name="net_return")

    net_returns = r - cost
    net_returns.name = "net_return"
    return net_returns


def bootstrap_metrics(
    returns: pd.Series | np.ndarray | list[float],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate bootstrap confidence intervals for key strategy metrics."""
    r = _to_clean_series(returns)

    if len(r) == 0:
        return pd.DataFrame(
            [
                {
                    "bootstrap_mean_return": np.nan,
                    "mean_return_ci_lower": np.nan,
                    "mean_return_ci_upper": np.nan,
                    "bootstrap_sharpe_ratio": np.nan,
                    "sharpe_ratio_ci_lower": np.nan,
                    "sharpe_ratio_ci_upper": np.nan,
                    "bootstrap_profit_factor": np.nan,
                    "profit_factor_ci_lower": np.nan,
                    "profit_factor_ci_upper": np.nan,
                }
            ]
        )

    rng = np.random.default_rng(random_state)
    values = r.to_numpy()
    n = len(values)

    boot_mean = []
    boot_sharpe = []
    boot_profit_factor = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        sample_mean = float(np.mean(sample))
        sample_std = float(np.std(sample, ddof=1)) if n > 1 else np.nan
        sample_sharpe = sample_mean / sample_std if pd.notna(sample_std) and sample_std != 0 else np.nan

        gross_profit = float(sample[sample > 0].sum())
        gross_loss = float(abs(sample[sample < 0].sum()))
        sample_profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

        boot_mean.append(sample_mean)
        boot_sharpe.append(sample_sharpe)
        boot_profit_factor.append(sample_profit_factor)

    result = {
        "bootstrap_mean_return": float(np.nanmean(boot_mean)),
        "mean_return_ci_lower": float(np.nanpercentile(boot_mean, 2.5)),
        "mean_return_ci_upper": float(np.nanpercentile(boot_mean, 97.5)),
        "bootstrap_sharpe_ratio": float(np.nanmean(boot_sharpe)),
        "sharpe_ratio_ci_lower": float(np.nanpercentile(boot_sharpe, 2.5)),
        "sharpe_ratio_ci_upper": float(np.nanpercentile(boot_sharpe, 97.5)),
        "bootstrap_profit_factor": float(np.nanmean(boot_profit_factor)),
        "profit_factor_ci_lower": float(np.nanpercentile(boot_profit_factor, 2.5)),
        "profit_factor_ci_upper": float(np.nanpercentile(boot_profit_factor, 97.5)),
    }
    return pd.DataFrame([result])


def build_backtest_table(
    df: pd.DataFrame,
    asset_name: str,
    pattern_directions: dict[str, list[str] | tuple[str, ...]],
    horizons: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Build a thesis-ready backtest summary table across patterns and horizons."""
    rows: list[dict[str, object]] = []

    for pattern_col, directions in pattern_directions.items():
        if pattern_col not in df.columns:
            continue

        for direction in directions:
            for h in horizons:
                strategy_returns = backtest_pattern_strategy(
                    df=df,
                    pattern_col=pattern_col,
                    h=h,
                    direction=direction,
                )

                metrics = performance_metrics(strategy_returns).iloc[0].to_dict()
                rows.append(
                    {
                        "Asset": asset_name,
                        "Pattern": pattern_col,
                        "Direction": direction,
                        "Holding Period": int(h),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def build_backtest_robustness_table(
    df: pd.DataFrame,
    asset_name: str,
    pattern_directions: dict[str, list[str] | tuple[str, ...]],
    horizons: Iterable[int] = (1, 3, 5),
    transaction_cost: float = 0.0001,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a thesis-ready table with gross, net, and bootstrap robustness metrics."""
    rows: list[dict[str, object]] = []

    for pattern_col, directions in pattern_directions.items():
        if pattern_col not in df.columns:
            continue

        for direction in directions:
            for h in horizons:
                gross_returns = backtest_pattern_strategy(
                    df=df,
                    pattern_col=pattern_col,
                    h=h,
                    direction=direction,
                )
                gross_metrics = performance_metrics(gross_returns).iloc[0].to_dict()

                net_returns = apply_transaction_cost(gross_returns, cost=transaction_cost)
                net_metrics = performance_metrics(net_returns).iloc[0].to_dict()

                boot_metrics = bootstrap_metrics(
                    net_returns,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                ).iloc[0].to_dict()

                rows.append(
                    {
                        "Asset": asset_name,
                        "Pattern": pattern_col,
                        "Direction": direction,
                        "Holding Period": int(h),
                        "Gross total_return": gross_metrics.get("total_return", np.nan),
                        "Gross sharpe_ratio": gross_metrics.get("sharpe_ratio", np.nan),
                        "Gross max_drawdown": gross_metrics.get("max_drawdown", np.nan),
                        "Gross profit_factor": gross_metrics.get("profit_factor", np.nan),
                        "Gross win_rate": gross_metrics.get("win_rate", np.nan),
                        "Gross n_trades": gross_metrics.get("n_trades", np.nan),
                        "Net total_return": net_metrics.get("total_return", np.nan),
                        "Net sharpe_ratio": net_metrics.get("sharpe_ratio", np.nan),
                        "Net max_drawdown": net_metrics.get("max_drawdown", np.nan),
                        "Net profit_factor": net_metrics.get("profit_factor", np.nan),
                        "Net win_rate": net_metrics.get("win_rate", np.nan),
                        "Bootstrap mean_return CI lower": boot_metrics.get("mean_return_ci_lower", np.nan),
                        "Bootstrap mean_return CI upper": boot_metrics.get("mean_return_ci_upper", np.nan),
                        "Bootstrap sharpe_ratio CI lower": boot_metrics.get("sharpe_ratio_ci_lower", np.nan),
                        "Bootstrap sharpe_ratio CI upper": boot_metrics.get("sharpe_ratio_ci_upper", np.nan),
                        "Bootstrap profit_factor CI lower": boot_metrics.get("profit_factor_ci_lower", np.nan),
                        "Bootstrap profit_factor CI upper": boot_metrics.get("profit_factor_ci_upper", np.nan),
                    }
                )

    return pd.DataFrame(rows)
