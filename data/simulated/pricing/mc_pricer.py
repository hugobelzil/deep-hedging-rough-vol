"""Conditional Monte Carlo pricing of a European call under rough Bergomi."""

import numpy as np
from scipy.stats import norm

if __package__:
    from ..simulated_data import fractional_BM_paths
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.simulated.simulated_data import fractional_BM_paths


def conditional_theta(
    t_index: int, V_t: float, brownian_history: np.array,
    T: float, N: int, H: float, V0: float, nu: float,
):
    """Gaussian contribution of the observed history at dates t_index, ..., N."""
    if T <= 0 or N < 1:
        raise ValueError("T and N must be positive.")
    if not 0 <= t_index <= N:
        raise ValueError("t_index must be between 0 and N.")
    if not 0 < H <= 0.5:
        raise ValueError("H must be in (0, 0.5] for the existing simulator.")
    if V0 <= 0 or V_t <= 0:
        raise ValueError("V0 and V_t must be positive.")

    dW = np.asarray(brownian_history, dtype=float)
    if dW.shape != (t_index,):
        raise ValueError("brownian_history must contain exactly t_index increments.")
    if t_index == 0 and not np.isclose(V_t, V0, rtol=1e-12, atol=0.0):
        raise ValueError("V_t must equal V0 at time zero.")

    alpha = H - 0.5
    delta = T / N
    t = np.linspace(0.0, T, N + 1)
    theta = np.zeros(N - t_index + 1)

    # V_t includes the most recent singular integral of the hybrid scheme.
    theta[0] = np.log(V_t / V0) + 0.5 * nu**2 * t[t_index] ** (2 * H)
    for j in range(1, N - t_index + 1):
        lag = t_index + j - np.arange(t_index)
        weights = delta**alpha * (
            lag**(alpha + 1) - (lag - 1)**(alpha + 1)
        ) / (alpha + 1)
        theta[j] = nu * np.sqrt(2 * H) * np.dot(weights, dW)

    return theta


def sample_future_noise(
    M: int, t_index: int, T: float, N: int, H: float, *, rng=None,
) -> dict:
    """Sample rough paths and Brownian increments on the remaining grid."""
    if M < 1:
        raise ValueError("M must be at least 1.")
    if T <= 0 or N < 1:
        raise ValueError("T and N must be positive.")
    if not 0 <= t_index <= N:
        raise ValueError("t_index must be between 0 and N.")
    if not 0 < H <= 0.5:
        raise ValueError("H must be in (0, 0.5] for the existing simulator.")
    if rng is not None:
        rng = np.random.default_rng(rng)

    remaining_steps = N - t_index
    if remaining_steps == 0:
        return dict(Y=np.zeros((M, 1)), dW=np.empty((M, 0)), dB=np.empty((M, 0)))

    # Retain the original time step after conditioning.
    remaining_time = T if t_index == 0 else remaining_steps * (T / N)
    rough_path, dW = fractional_BM_paths(
        H - 0.5, remaining_time, remaining_steps, n_paths=M, rng=rng
    )
    dB = norm.rvs(scale=np.sqrt(T / N), size=dW.shape, random_state=rng)
    return dict(Y=rough_path, dW=dW, dB=dB)


def payoffs_from_noise(
    S_t, theta, noise: dict, times: np.array,
    H: float, V0: float, nu: float, rho: float, K: float, *, xp=np,
):
    """Evaluate call payoffs using fixed Monte Carlo draws.

    theta and times include the current date and expiry. Noise arrays have a
    leading path axis. Set xp=jax.numpy to differentiate the pricing graph.
    """
    rough_path = xp.asarray(noise["Y"])
    dW = xp.asarray(noise["dW"])
    dB = xp.asarray(noise["dB"])
    times = xp.asarray(times)

    variance = V0 * xp.exp(
        xp.asarray(theta) + nu * rough_path - 0.5 * nu**2 * times**(2 * H)
    )
    stock_price = S_t * xp.ones(dW.shape[0])
    independent_scale = xp.sqrt(1 - rho**2)
    for i in range(dW.shape[-1]):
        stock_price = stock_price * (
            1 + xp.sqrt(variance[:, i])
            * (independent_scale * dB[:, i] + rho * dW[:, i])
        )

    return xp.maximum(stock_price - K, 0.0)


def price_from_noise(
    S_t, theta, noise: dict, times: np.array,
    H: float, V0: float, nu: float, rho: float, K: float, *, xp=np,
):
    """Mean call payoff for fixed Monte Carlo draws."""
    return xp.mean(payoffs_from_noise(
        S_t, theta, noise, times, H, V0, nu, rho, K, xp=xp
    ))


def pricer(
    M: int,
    K: float = 100.0,
    S_t: float = 100.0,
    T: float = 30 / 365,
    N: int = 30,
    H: float = 0.10,
    V0: float = 0.235**2,
    nu: float = 1.9,
    rho: float = -0.7,
    t_index: int = 0,
    V_t: float = None,
    brownian_history=None,
    batch_size: int = 1000,
    *,
    rng=None,
    return_stats: bool = False,
):
    """Estimate E[(S_T-K)^+ | F_t] on the original simulation grid.

    T and N describe the full option life. At intermediate dates, supply V_t
    and the observed increments dW[:t_index]. At time zero they default to V0
    and an empty history; at expiry the price is intrinsic value. rng=None
    uses NumPy's global random state. return_stats adds Monte Carlo errors.
    """
    if M < 1:
        raise ValueError("M must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if T <= 0 or N < 1:
        raise ValueError("T and N must be positive.")
    if not 0 <= t_index <= N:
        raise ValueError("t_index must be between 0 and N.")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1.")
    if t_index == N:
        price = float(np.maximum(S_t - K, 0.0))
        if return_stats:
            return dict(price=price, standard_error=0.0, sample_std=0.0, n_paths=M)
        return price

    if V_t is None:
        if t_index != 0:
            raise ValueError("V_t is required after time zero.")
        V_t = V0
    if brownian_history is None:
        if t_index != 0:
            raise ValueError("brownian_history is required after time zero.")
        brownian_history = np.empty(0)

    theta = conditional_theta(t_index, V_t, brownian_history, T, N, H, V0, nu)
    times = np.linspace(0.0, T, N + 1)[t_index:]
    if rng is not None:
        rng = np.random.default_rng(rng)

    # Reuse the RNG across batches (very fucking important)
    price_sum = 0.0
    running_mean, squared_deviations, total = 0.0, 0.0, 0
    for start in range(0, M, batch_size):
        count = min(batch_size, M - start)
        noise = sample_future_noise(count, t_index, T, N, H, rng=rng)
        payoffs = payoffs_from_noise(
            S_t, theta, noise, times, H, V0, nu, rho, K
        )
        batch_mean = float(np.mean(payoffs))
        price_sum += count * batch_mean
        if return_stats:
            # Combine central moments to avoid cancellation.
            difference = batch_mean - running_mean
            new_total = total + count
            squared_deviations += float(np.sum((payoffs - batch_mean)**2))
            squared_deviations += difference**2 * total * count / new_total
            running_mean += difference * count / new_total
            total = new_total

    price = float(price_sum / M)
    if return_stats:
        sample_std = np.sqrt(squared_deviations / (M - 1)) if M > 1 else np.nan
        return dict(
            price=price, standard_error=float(sample_std / np.sqrt(M)),
            sample_std=float(sample_std), n_paths=M,
        )
    return price
