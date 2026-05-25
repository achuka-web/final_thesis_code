"""Reusable candlestick pattern detection helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PATTERN_COLS = [
    "hammer",
    "hanging_man",
    "bullish_engulfing",
    "bearish_engulfing",
    "morning_star",
    "evening_star",
    "doji",
    "spinning_top",
]


def add_candle_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable candle component columns based on OHLC data.

    Required columns:
    - Open
    - High
    - Low
    - Close
    """
    data = df.copy()

    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required OHLC columns: {missing}")

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required).copy()

    data["body"] = (data["Close"] - data["Open"]).abs()
    data["range"] = (data["High"] - data["Low"]).replace(0, np.nan)
    data["upper_shadow"] = data["High"] - data[["Open", "Close"]].max(axis=1)
    data["lower_shadow"] = data[["Open", "Close"]].min(axis=1) - data["Low"]
    data["body_to_range"] = data["body"] / data["range"]

    data["prev_open"] = data["Open"].shift(1)
    data["prev_close"] = data["Close"].shift(1)
    data["prev_high"] = data["High"].shift(1)
    data["prev_low"] = data["Low"].shift(1)
    data["prev_body"] = (data["prev_close"] - data["prev_open"]).abs()
    data["prev_candle_range"] = (data["prev_high"] - data["prev_low"]).replace(0, np.nan)
    data["prev_body_to_range"] = data["prev_body"] / data["prev_candle_range"]

    data["prev2_open"] = data["Open"].shift(2)
    data["prev2_close"] = data["Close"].shift(2)

    return data


def add_candlestick_patterns(
    df: pd.DataFrame,
    alpha: float = 0.3,
    beta: float = 0.5,
    gamma: float = 0.1,
    delta: float = 0.5,
    lambda_: float = 0.1,
) -> pd.DataFrame:
    """Add binary candlestick pattern columns using notebook-based rules.

    Parameters map to the thesis notation:
    - alpha: maximum body/range ratio for hammer-shape candles
    - beta: maximum upper-shadow/body ratio for hammer-shape candles
    - gamma: maximum body/range ratio for doji
    - delta: maximum body/range ratio for spinning top
    - lambda_: minimum body/range ratio for spinning top
    """
    data = add_candle_components(df)

    downtrend = data["Close"] < data["Close"].rolling(24).mean()
    uptrend = data["Close"] > data["Close"].rolling(24).mean()

    body = data["body"]
    candle_range = data["range"]
    body_to_range = data["body_to_range"]
    upper_shadow = data["upper_shadow"]
    lower_shadow = data["lower_shadow"]

    prev_open = data["prev_open"]
    prev_close = data["prev_close"]
    prev2_open = data["prev2_open"]
    prev2_close = data["prev2_close"]
    prev_body_to_range = data["prev_body_to_range"]

    data["doji"] = (
        (body_to_range <= gamma)
    ).fillna(False).astype(int)

    data["spinning_top"] = (
        (body_to_range > lambda_)
        & (body_to_range <= delta)
        & (upper_shadow > body)
        & (lower_shadow > body)
    ).fillna(False).astype(int)

    hammer_shape = (
        (lower_shadow >= 2 * body)
        & (upper_shadow <= beta * body)
        & (body_to_range <= alpha)
    )

    data["hammer"] = (
        hammer_shape & downtrend
    ).fillna(False).astype(int)

    data["hanging_man"] = (
        hammer_shape & uptrend
    ).fillna(False).astype(int)

    data["bullish_engulfing"] = (
        (prev_close < prev_open)
        & (data["Close"] > data["Open"])
        & (data["Open"] <= prev_close)
        & (data["Close"] >= prev_open)
    ).fillna(False).astype(int)

    data["bearish_engulfing"] = (
        (prev_close > prev_open)
        & (data["Close"] < data["Open"])
        & (data["Open"] >= prev_close)
        & (data["Close"] <= prev_open)
    ).fillna(False).astype(int)

    data["morning_star"] = (
        (prev2_close < prev2_open)
        & (prev_body_to_range <= alpha)
        & (data["Close"] > data["Open"])
        & (data["Close"] > ((prev2_open + prev2_close) / 2))
    ).fillna(False).astype(int)

    data["evening_star"] = (
        (prev2_close > prev2_open)
        & (prev_body_to_range <= alpha)
        & (data["Close"] < data["Open"])
        & (data["Close"] < ((prev2_open + prev2_close) / 2))
    ).fillna(False).astype(int)

    return data


def pattern_summary(
    df: pd.DataFrame,
    pattern_cols: Iterable[str],
) -> pd.DataFrame:
    """Summarize pattern counts and proportions."""
    rows: list[dict[str, float]] = []
    n_obs = len(df)

    for pattern_col in pattern_cols:
        if pattern_col not in df.columns:
            continue

        count = int(pd.to_numeric(df[pattern_col], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "Pattern": pattern_col,
                "Count": count,
                "Ratio": (count / n_obs) if n_obs > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("Pattern").reset_index(drop=True)
