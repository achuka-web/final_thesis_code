# File Plan

This document defines the target structure for the clean thesis pipeline. It is based on `3.1.ipynb` as reference only; the old notebook is not modified.

## Top-level layout

`data/`
- `raw/`
  - Intended for original input CSV files such as `XAUUSDH1.csv` and `EURUSDH1.csv`
- `interim/`
  - Optional cleaned intermediate datasets
- `processed/`
  - Final analysis-ready outputs if we choose to persist them

`outputs/`
- `figures/`
  - Saved plots for thesis figures
- `tables/`
  - Exported result tables in CSV or LaTeX-ready formats
- `reports/`
  - Optional text or markdown summaries

`src/thesis_pipeline/`
- `__init__.py`
  - Package marker
- `config.py`
  - Central configuration: file paths, horizons, transaction cost, volume window, selected distributions, selected GARCH models
- `io.py`
  - Raw data loading, column validation, datetime parsing, numeric coercion
- `returns.py`
  - Log return computation, forward return computation, reusable return helpers
- `descriptive.py`
  - Descriptive statistics, skewness, kurtosis, summary tables
- `distribution.py`
  - Normality tests and parametric distribution fitting with AIC/BIC/KS summaries
- `stationarity.py`
  - ADF test wrappers and result formatting
- `dependence.py`
  - Ljung-Box on returns and squared returns, ARCH-LM
- `volatility.py`
  - GARCH/EGARCH fitting, model comparison, conditional volatility extraction, residual diagnostics if retained
- `patterns.py`
  - Candlestick pattern detection rules for hammer, hanging man, bullish/bearish engulfing, morning/evening star, doji, spinning top
- `volume.py`
  - Relative tick volume calculation and high/low volume grouping
- `pattern_tests.py`
  - Pattern-level statistical tests for `h=1,3,5`: one-sample `t`-test, Mann-Whitney, Levene, two-sample KS
- `volume_conditioned_tests.py`
  - Pattern tests stratified by relative tick volume groups
- `backtest.py`
  - Simple signal-based backtesting and performance metrics
- `costs.py`
  - Transaction cost adjustments and net performance metrics
- `bootstrap.py`
  - Bootstrap robustness for return, Sharpe, and profit factor summaries
- `pipeline.py`
  - Main orchestration layer that runs the full workflow end-to-end in the correct order

`notebooks/`
- Reserved for a future clean demonstration notebook if needed
- The main pipeline should still run from Python modules, not depend on notebooks

`tests/`
- `test_smoke.md`
  - Placeholder checklist for manual validation during development

## Planned execution flow

1. Load and validate OHLC tick volume CSV files.
2. Compute log returns and core descriptive statistics.
3. Run normality tests and fit candidate return distributions.
4. Run ADF, Ljung-Box, and ARCH-LM diagnostics.
5. Fit GARCH/EGARCH volatility models and select retained outputs.
6. Detect candlestick patterns.
7. Compute relative tick volume and assign high/low volume groups.
8. Compute forward returns for `h=1,3,5`.
9. Run pattern-level hypothesis tests.
10. Run volume-conditioned pattern tests.
11. Run simple backtests by pattern, direction, and horizon.
12. Apply transaction costs.
13. Run bootstrap robustness summaries.
14. Save tables and figures to `outputs/`.

## Design choices

- The new pipeline will be reproducible from a clean Python run.
- Shared logic will live in modules instead of repeated notebook cells.
- Column names will be standardized consistently, with `log_return` used everywhere.
- ML, forecasting, and train/test split code will not be recreated.
- `3.1.ipynb` is reference material only and remains unchanged.
