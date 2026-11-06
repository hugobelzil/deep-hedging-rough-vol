import numpy as np
from scipy.stats import norm, multivariate_normal

# RIEMANN-LIOUVILLE REPRESENTATION OF FRACTIONAL BROWNIAN MOTION

def fractional_BM_paths(alpha: float, T: float, N: int, n_paths: int = None,
                        *, rng=None):
    """Return one path, or a batch shaped (n_paths, N+1), using an optional RNG."""
    if not(-1/2 < alpha <= 0):
        raise ValueError("alpha must be in (-1/2, 0].")
    batch = 1 if n_paths is None else n_paths
    if N < 1 or batch < 1:
        raise ValueError("N and n_paths must be positive.")
    delta = T/N
    # Helper function for coefficients of the fBM generation
    def b(k: int):
        weight = delta**alpha
        weight *= (k**(alpha + 1) - (k - 1)**(alpha + 1)) * (1 / (alpha + 1))
        return weight

    weights = np.array([b(k) for k in range(2, N + 1)])

    # Sample all paths and dates together. Each pair (A_i, Delta W_i) has
    # this covariance; different dates and different paths are independent.
    covariance = [
        [delta**(2*alpha + 1) / (2*alpha + 1), delta**(alpha + 1) / (alpha + 1)],
        [delta**(alpha + 1) / (alpha + 1), delta],
    ]

    pairs = multivariate_normal.rvs(
        mean=[0, 0], cov=covariance, size=(batch, N), random_state=rng
    ).reshape(batch, N, 2)
    
    singular_integrals = pairs[:, :, 0]
    brownian_shocks = pairs[:, :, 1]

    path = np.zeros((batch, N + 1))
    scale = np.sqrt(2*alpha + 1)
    for i in range(N):
        # Add the weighted past increments to the current singular integral.
        # At i=0 the dot product is zero because both slices are empty.
        # Each row uses only its own past shocks. Compute all rows at once.
        history = brownian_shocks[:, :i][:, ::-1] @ weights[:i]
        path[:, i + 1] = scale * (singular_integrals[:, i] + history)

    # Preserve the original single-path return format for existing callers.
    if n_paths is None:
        return path[0], brownian_shocks[0].tolist()
    return path, brownian_shocks

# ROUGH VOLATILITY PROCESS

def rough_vol_process(alpha: float, T: float, N: int, V0: float, 
                      nu: float, path: np.array, brownian_shocks: list):
    
    """Convert rough paths to variance; the last axis contains the dates."""

    y = np.asarray(path)
    t = np.linspace(0.0, T, N + 1)
    return V0*np.exp(nu * y - 0.5 * nu**2 * t**(2 * alpha + 1))


# STOCK PRICE PROCESS (EULER-MARUYAMA SCHEME)

def stock_price_process(S0: float, T: float, N: int, rho: float, 
                          path: np.array, brownian_shocks: list,
                          *, independent_shocks=None, rng=None):
    """Simulate stock prices, optionally reusing independent Brownian increments."""
    delta = T/N
    BM = np.asarray(brownian_shocks)
    variance = np.asarray(path)
    # One independent Brownian increment for every path and time interval.
    if independent_shocks is None:
        indep_BM = norm.rvs(scale=np.sqrt(delta), size=BM.shape, random_state=rng)
    else:
        indep_BM = np.asarray(independent_shocks, dtype=float)
        if indep_BM.shape != BM.shape:
            raise ValueError("independent_shocks must have the same shape as brownian_shocks.")
    stock_price = np.zeros_like(variance, dtype=float)
    stock_price[..., 0] = S0
    # Keep the Euler recurrence in time, but update all paths simultaneously.
    independent_scale = np.sqrt(1 - rho**2)
    for i in range(N):
        stock_price[..., i+1] = stock_price[..., i] * (
            1 + np.sqrt(variance[..., i])
            * (independent_scale * indep_BM[..., i] + rho * BM[..., i])
        )
    return stock_price

# FORWARD VARIANCE PATHS

def forward_variance_paths(alpha, T, N, V0, nu, T_fwd, brownian_shocks):
    if T_fwd <= T:
        raise ValueError("T_fwd must exceed T.")

    dW = np.asarray(brownian_shocks, dtype=float)
    if dW.ndim not in (1, 2) or dW.shape[-1] != N:
        raise ValueError("Expected shape (N,) or (n_paths, N).")

    t = np.arange(N) * (T / N)  # Left endpoints

    factors = 1 + (
        np.sqrt(2 * alpha + 1)
        * nu
        * (T_fwd - t)**alpha
        * dW
    )

    initial = np.ones(dW.shape[:-1] + (1,))
    return V0*np.concatenate((initial, np.cumprod(factors, axis=-1)), axis=-1)


# JOINT MARKET SIMULATION FOR HEDGING

def simulate_market(
    S0: float = 100.0,
    T: float = 30 / 365,
    N: int = 30,
    H: float = 0.10,
    V0: float = 0.235**2,
    nu: float = 1.9,
    rho: float = -0.7,
    T_fwd: float = 45 / 365,
    n_paths: int = None,
    *,
    rng=None,
) -> dict:
    """Return coupled market paths and their Brownian increments.

    Paths have shape (N+1,), or (n_paths, N+1) when n_paths is specified.
    Brownian increments have one fewer date. Pass a Generator or seed as rng;
    rng=None preserves the existing np.random.seed behaviour.
    """
    alpha = H - 0.5
    if not 0 < H <= 0.5:
        raise ValueError("H must be in (0, 0.5] for the existing simulator.")
    if T_fwd <= T:
        raise ValueError("T_fwd must exceed T.")
    # Normalize a supplied seed once so both Brownian sources share one RNG.
    if rng is not None:
        rng = np.random.default_rng(rng)

    rough_path, brownian_shocks = fractional_BM_paths(
        alpha, T, N, n_paths=n_paths, rng=rng
    )
    dW = np.asarray(brownian_shocks)
    dB = norm.rvs(scale=np.sqrt(T/N), size=dW.shape, random_state=rng)

    variance_path = rough_vol_process(
        alpha, T, N, V0, nu, rough_path, dW
    )
    stock_path = stock_price_process(
        S0, T, N, rho, variance_path, dW, independent_shocks=dB
    )
    forward_path = forward_variance_paths(
        alpha, T, N, V0, nu, T_fwd, dW
    )

    return dict(
        times=np.linspace(0.0, T, N + 1),
        Y=rough_path, V=variance_path, S=stock_path, F=forward_path,
        dW=dW, dB=dB,
        parameters=dict(
            S0=S0, T=T, N=N, H=H, alpha=alpha, V0=V0,
            nu=nu, rho=rho, T_fwd=T_fwd, n_paths=n_paths,
        ),
    )
