import numpy as np
from scipy.stats import norm, multivariate_normal

# RIEMANN-LIOUVILLE REPRESENTATION OF FRACTIONAL BROWNIAN MOTION

def fractional_BM_paths(alpha: float, T: float, N: int, n_paths: int = None):
    """Return one path, or a batch shaped (n_paths, N+1) when specified."""
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
        mean=[0, 0], cov=covariance, size=(batch, N)
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
                          path: np.array, brownian_shocks: list):
    delta = T/N
    BM = np.asarray(brownian_shocks)
    variance = np.asarray(path)
    # One independent Brownian increment for every path and time interval.
    indep_BM = norm.rvs(scale=np.sqrt(delta), size=BM.shape)
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
