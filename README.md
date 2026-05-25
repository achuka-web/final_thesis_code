# Final Thesis Code

This folder contains a clean, reproducible Python pipeline for the thesis analysis.

Scope kept in the new pipeline:
- Load OHLC tick volume data
- Compute log returns
- Descriptive statistics
- Normality tests and distribution fitting
- ADF
- Ljung-Box on returns and squared returns
- ARCH-LM
- GARCH/EGARCH volatility estimation
- Candlestick pattern detection
- Relative tick volume grouping
- Forward returns for `h=1,3,5`
- Pattern-level `t`-test, Mann-Whitney, Levene, and two-sample KS tests
- Volume-conditioned pattern tests
- Simple backtesting
- Transaction cost analysis
- Bootstrap robustness

Explicitly removed from this pipeline:
- Machine learning
- Logistic regression
- ROC/AUC
- Volatility forecasting
- In-sample / out-of-sample split logic

See [FILE_PLAN.md](/C:/Users/user/Documents/New%20project/final_thesis_code/FILE_PLAN.md) for the intended module layout and responsibilities.
