"""Simulate rBergomi stock paths and get price of european call"""

import numpy as np

from simulated_data import (
    fractional_BM_paths,
    rough_vol_process,
    stock_price_process,
)


def simulate_price_path(
    S0: float = 100.0,
    T: float = 30 / 365,
    N: int = 30,
    H: float = 0.10,
    V0: float = 0.235**2,
    nu: float = 1.9,
    rho: float = -0.7,
    n_paths: int = None,
) -> np.ndarray:
    """Return shape (N+1,), or (n_paths, N+1) when n_paths is specified."""
    # The paper reports results for several Hurst parameters; H=0.10 is its
    # roughest example. The existing simulator uses alpha = H - 1/2 instead.
    alpha = H - 0.5
    if not 0 < H <= 0.5:
        raise ValueError("H must be in (0, 0.5] for the existing simulator.")

    # Generate the rough Brownian path and its underlying Brownian increments.
    # Reusing these increments below preserves the dependence between the
    # volatility and stock-price processes.
    rough_path, brownian_shocks = fractional_BM_paths(alpha, T, N, n_paths=n_paths)

    # Convert the rough path into instantaneous variance V at all N+1 dates.
    variance_path = rough_vol_process(
        alpha, T, N, V0, nu, rough_path, brownian_shocks
    )

    # Use that variance and the same shocks to simulate S at all N+1 dates.
    return stock_price_process(S0, T, N, rho, variance_path, brownian_shocks)


def p0(
    M: int = 10_000,
    K: float = 100.0,
    batch_size: int = 1000,
    **process_parameters,
) -> float:
    """Estimate the time-zero price of an undiscounted European call by Monte Carlo."""
    # Each simulated path contributes max(S_T - K, 0). Under the paper's
    # zero-drift stock model, their average estimates E[(S_T - K)^+].
    if M < 1:
        raise ValueError("M must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    # Only keep one batch in memory. The final batch may contain fewer paths.
    # Sum payoffs before dividing by M so that every path has equal weight.
    payoff_sum = 0.0
    for start in range(0, M, batch_size):
        count = min(batch_size, M - start)
        prices = simulate_price_path(n_paths=count, **process_parameters)
        payoff_sum += np.maximum(prices[:, -1] - K, 0.0).sum()
    return float(payoff_sum / M)


if __name__ == "__main__":
    print("Estimated price in 3.5M paths : ", p0(M = 3500000, K = 100.0, S0= 100.0,
        T= 30 / 365,
        N = 30,
        H= 0.10,
        V0= 0.235**2,
        nu= 1.9,
        rho= -0.7))
