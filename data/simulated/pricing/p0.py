"""Rough Bergomi stock simulation and time-zero call pricing."""

import numpy as np

if __package__:
    from .mc_pricer import pricer
    from ..simulated_data import (
        fractional_BM_paths,
        rough_vol_process,
        stock_price_process,
    )
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.simulated.pricing.mc_pricer import pricer
    from data.simulated.simulated_data import (
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
    *,
    rng=None,
) -> np.ndarray:
    """Simulate one stock path, or a batch with shape (n_paths, N+1)."""
    alpha = H - 0.5
    if not 0 < H <= 0.5:
        raise ValueError("H must be in (0, 0.5] for the existing simulator.")
    if rng is not None:
        rng = np.random.default_rng(rng)

    rough_path, brownian_shocks = fractional_BM_paths(
        alpha, T, N, n_paths=n_paths, rng=rng
    )

    variance_path = rough_vol_process(
        alpha, T, N, V0, nu, rough_path, brownian_shocks
    )

    # Reuse the variance driver to preserve the stock-variance correlation.
    return stock_price_process(
        S0, T, N, rho, variance_path, brownian_shocks, rng=rng
    )


def p0(
    M: int,
    K: float = 100.0,
    batch_size: int = 1000,
    *,
    S0: float = 100.0,
    rng=None,
    **process_parameters,
) -> float:
    """Estimate the time-zero European call price with spot S0 and strike K.

    S0 defaults to 100 independently of the strike.
    """
    return pricer(
        M=M, K=K, S_t=S0, t_index=0, batch_size=batch_size,
        rng=rng, **process_parameters,
    )
