"""Self-financing call hedges and P&L statistics."""

import numpy as np


def self_financing_portfolio(
    prices: np.ndarray,
    holdings: np.ndarray,
    premium: float,
    payoff: np.ndarray,
) -> dict:
    """Rebalance through N-1 and liquidate at N, with zero rates and costs.

    prices: (n_paths, N+1, n_assets); holdings: (n_paths, N, n_assets).
    Holdings at i use only information available at i. Cash is recorded after
    each rebalance and liquidation. Terminal P&L is premium + gains - payoff.
    """
    prices = np.asarray(prices, dtype=float)
    holdings = np.asarray(holdings, dtype=float)
    payoff = np.asarray(payoff, dtype=float)
    if prices.ndim != 3 or prices.shape[1] < 1:
        raise ValueError("prices must have shape (n_paths, N+1, n_assets).")
    n_paths, n_dates, n_assets = prices.shape
    if n_paths < 1 or n_assets < 1:
        raise ValueError("At least one path and one asset are required.")
    if holdings.shape != (n_paths, n_dates - 1, n_assets):
        raise ValueError("holdings must have shape (n_paths, N, n_assets).")
    if payoff.shape != (n_paths,):
        raise ValueError("payoff must have shape (n_paths,).")
    premium = np.broadcast_to(np.asarray(premium, dtype=float), (n_paths,))
    if not all(np.all(np.isfinite(x)) for x in (prices, holdings, premium, payoff)):
        raise ValueError("Prices, holdings, premium and payoff must be finite.")

    # Hold at i, earn the move to i+1.
    increments = np.sum(holdings * np.diff(prices, axis=1), axis=2)
    gains = np.zeros((n_paths, n_dates))
    gains[:, 1:] = np.cumsum(increments, axis=1)
    wealth = premium[:, None] + gains

    cash = np.empty_like(wealth)
    cash[:, :-1] = wealth[:, :-1] - np.sum(holdings * prices[:, :-1], axis=2)
    cash[:, -1] = wealth[:, -1]
    liquidation = np.zeros(n_paths)
    if n_dates > 1:
        liquidation = np.sum(holdings[:, -1] * prices[:, -1], axis=1)

    return dict(
        cash=cash,
        wealth=wealth,
        gains=gains,
        liquidation=liquidation,
        pnl=wealth[:, -1] - payoff,
    )


def _mean_and_se(values: np.ndarray) -> tuple:
    mean = float(np.mean(values))
    se = (
        float(np.std(values, ddof=1) / np.sqrt(values.size))
        if values.size > 1 else np.nan
    )
    return mean, se


def _lower_tail_mean(values: np.ndarray, probability: float) -> float:
    """Average the lowest probability mass, weighting the boundary sample."""
    ordered = np.sort(values)
    mass = probability * ordered.size
    count = int(np.floor(mass))
    total = np.sum(ordered[:count])
    if count < ordered.size:
        total += (mass - count) * ordered[count]
    return float(total / mass)


def pnl_summary(pnl: np.ndarray, premium_standard_error: float = 0.0) -> dict:
    """Summarise short-option P&L; negative P&L is a hedging loss... bad.

    Standard errors and normal 95% intervals assume independent paths and
    independent inner noise per path. mean_se and mse_se condition on the
    common premium; with_premium fields add independent premium uncertainty
    (first-order delta method for MSE). Shared inner noise requires separate
    seed runs. Tail estimates are empirical, without confidence intervals.
    """
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1 or pnl.size < 1 or not np.all(np.isfinite(pnl)):
        raise ValueError("pnl must be a nonempty, finite one-dimensional array.")
    if not np.isfinite(premium_standard_error) or premium_standard_error < 0:
        raise ValueError("premium_standard_error must be finite and nonnegative.")
    mean, mean_se = _mean_and_se(pnl)
    mse, mse_se = _mean_and_se(pnl**2)
    mean_se_with_premium = float(np.hypot(mean_se, premium_standard_error))
    mse_se_with_premium = float(np.hypot(mse_se, 2 * mean * premium_standard_error))
    rmse = float(np.sqrt(mse))
    quantiles = np.quantile(pnl, [0.01, 0.05, 0.50, 0.95, 0.99])
    lower_tail = _lower_tail_mean(pnl, 0.05)
    return dict(
        n_paths=pnl.size,
        mean=mean,
        std=float(np.std(pnl, ddof=1)) if pnl.size > 1 else np.nan,
        mean_se=mean_se,
        mean_ci_low=mean - 1.96 * mean_se,
        mean_ci_high=mean + 1.96 * mean_se,
        mean_se_with_premium=mean_se_with_premium,
        mean_ci95_with_premium=(
            mean - 1.96 * mean_se_with_premium,
            mean + 1.96 * mean_se_with_premium,
        ),
        mse=mse,
        mse_se=mse_se,
        mse_ci_low=mse - 1.96 * mse_se,
        mse_ci_high=mse + 1.96 * mse_se,
        mse_se_with_premium_first_order=mse_se_with_premium,
        mse_ci95_with_premium_first_order=(
            mse - 1.96 * mse_se_with_premium,
            mse + 1.96 * mse_se_with_premium,
        ),
        rmse=rmse,
        rmse_se=mse_se / (2 * rmse) if rmse > 0 else mse_se,
        q01=float(quantiles[0]),
        q05=float(quantiles[1]),
        median=float(quantiles[2]),
        q95=float(quantiles[3]),
        q99=float(quantiles[4]),
        lower_tail_mean_05=lower_tail,
        loss_var_95=float(-quantiles[1]),
        loss_expected_shortfall_95=-lower_tail,
        loss_probability=float(np.mean(pnl < 0)),
    )
