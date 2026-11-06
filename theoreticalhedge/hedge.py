"""rBergomi hedge weights computed with JAX."""

import numpy as np
import jax

# JAX > FDM supremacy
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from data.simulated.pricing import mc_pricer as pricing


def kernel_direction(times: np.ndarray, H: float, T_fwd: float):
    """Bucket-averaged kernel direction for the forward-variance hedge."""
    times = np.asarray(times, dtype=float)
    direction = np.zeros_like(times)
    if len(times) == 1:
        return direction

    alpha = H - 0.5
    offsets = times - times[0]
    direction[:-1] = np.diff(offsets ** (alpha + 1)) / (
        (alpha + 1) * np.diff(times)
    )
    direction[-1] = offsets[-1] ** alpha
    return direction / (T_fwd - times[0]) ** alpha


def _parameters(market: dict, K=None):
    """Collect the model parameters and option strike."""
    parameters = dict(market.get("parameters", {}))
    parameters.update({
        name: market[name]
        for name in ("T", "N", "H", "V0", "nu", "rho", "K")
        if name in market
    })
    parameters["R"] = market.get("R", parameters.get("T_fwd"))
    parameters["K"] = parameters.get("K", 100.0) if K is None else K
    return parameters


def _market_paths(market: dict, N: int):
    """Normalize single-path and batched market arrays without changing inputs."""
    paths = {
        name: np.atleast_2d(np.asarray(market[name]))
        for name in ("S", "F", "V", "dW")
    }
    stock = paths["S"]
    if stock.ndim != 2 or stock.shape[1] != N + 1:
        raise ValueError("S must have shape (N+1,) or (n_paths, N+1).")
    if any(paths[name].shape != stock.shape for name in ("F", "V")):
        raise ValueError("S, F and V must have matching path and date counts.")
    if paths["dW"].shape != (stock.shape[0], N):
        raise ValueError("dW must have shape (N,) or (n_paths, N), matching S.")
    return paths


def _future_noise(n_inner, t_index, parameters, seed, path_id, batch_size):
    """Give each market path and trading date its own inner random stream."""
    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(int(t_index), int(path_id)))
    )
    batches = []
    for start in range(0, n_inner, batch_size):
        count = min(batch_size, n_inner - start)
        batches.append(
            pricing.sample_future_noise(
                count, t_index, parameters["T"], parameters["N"], parameters["H"],
                rng=rng,
            )
        )
    return {
        name: np.concatenate([batch[name] for batch in batches], axis=0)
        for name in ("Y", "dW", "dB")
    }


def make_hedge_price(
    market: dict,
    path_id: int,
    t_index: int,
    n_inner: int = 10_000,
    seed: int = 6969,
    inner_batch_size: int = 1_000,
    K=None,
):
    """Conditional call price for a single-path or batched market.

    Use path_id=0 for a market returned without an explicit n_paths.
    """
    parameters = _parameters(market, K)
    if not 0 <= t_index < parameters["N"]:
        raise ValueError("The hedge date must precede expiry.")
    if n_inner < 1 or inner_batch_size < 1:
        raise ValueError("n_inner and inner_batch_size must be positive.")

    paths = _market_paths(market, parameters["N"])
    times = np.asarray(market["times"])[t_index:]
    theta = pricing.conditional_theta(
        t_index, paths["V"][path_id, t_index],
        paths["dW"][path_id, :t_index], parameters["T"], parameters["N"],
        parameters["H"], parameters["V0"], parameters["nu"],
    )
    direction = kernel_direction(times, parameters["H"], parameters["R"])
    noise = _future_noise(
        n_inner, t_index, parameters, seed, path_id, inner_batch_size
    )

    # sometimes notation co
    def price_fn(S_t, epsilon, *, xp=np):
        shifted_theta = xp.asarray(theta) + epsilon * xp.asarray(direction)
        return pricing.price_from_noise(
            S_t, shifted_theta, noise, times, parameters["H"], parameters["V0"],
            parameters["nu"], parameters["rho"], parameters["K"], xp=xp,
        )

    state = dict(
        spot=float(paths["S"][path_id, t_index]),
        forward=float(paths["F"][path_id, t_index]),
        times=times, theta=theta, direction=direction, noise=noise,
        parameters=parameters,
    )
    return price_fn, state


def autodiff_holdings(price_fn, S_t: float, F_t: float):
    """Stock delta and forward-variance units from the JAX gradient."""

    def price_jax(spot, epsilon):
        return price_fn(spot, epsilon, xp=jnp)

    delta_s, directional = jax.grad(price_jax, argnums=(0, 1))(float(S_t), 0.0)
    return float(delta_s), float(directional / F_t)


def _holdings_one(
    spot, forward, theta, rough_path, dW, dB,
    times, direction, model,
):
    """Differentiate the price for one market path and trading date."""
    H, V0, nu, rho, K = model
    sqrt_variance = jnp.sqrt(V0) * jnp.exp(
        0.5 * (
            theta[:-1] + nu * rough_path[:, :-1]
            - 0.5 * nu**2 * times[:-1] ** (2 * H)
        )
    )
    shocks = jnp.sqrt(1 - rho**2) * dB + rho * dW

    def price_fn(S_t, epsilon):
        factors = 1 + sqrt_variance * jnp.exp(
            0.5 * epsilon * direction[:-1]
        ) * shocks

        def update(stock, factor):
            return stock * factor, None

        terminal, _ = jax.lax.scan(
            update, S_t * jnp.ones(dW.shape[0]), factors.T
        )
        return jnp.mean(jnp.maximum(terminal - K, 0.0))

    delta_s, directional = jax.grad(price_fn, argnums=(0, 1))(spot, 0.0)
    holdings = jnp.array([delta_s, directional / forward])
    return holdings, price_fn(spot, 0.0)


_holdings_batch = jax.jit(
    jax.vmap(
        _holdings_one,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    ),
)


def full_holdings(
    market: dict,
    K=None,
    n_inner: int = 2_000,
    seed: int = 6969,
    path_batch_size: int = 16,
    inner_batch_size: int = 1_000,
    progress: bool = False,
):
    """Compute both hedge positions at dates 0, ..., N-1.

    Accept single-path or batched market arrays. Holdings have shape
    (n_paths, N, 2): stock, then forward variance, with n_paths=1 for a
    single-path market.
    Inner draws use a separate random stream for each path and date.
    Memory scales with path_batch_size * n_inner * remaining dates.
    """
    if min(n_inner, path_batch_size, inner_batch_size) < 1:
        raise ValueError("Monte Carlo and batch counts must be positive.")

    parameters = _parameters(market, K)
    N = parameters["N"]
    paths = _market_paths(market, N)
    stock, forward = paths["S"], paths["F"]
    n_paths = stock.shape[0]
    holdings = np.zeros((n_paths, N, 2))
    prices = np.zeros((n_paths, N))
    model = np.array([
        parameters[name] for name in ("H", "V0", "nu", "rho", "K")
    ])

    for i in range(N):
        times = np.asarray(market["times"])[i:]
        direction = kernel_direction(times, parameters["H"], parameters["R"])
        # Batch
        for start in range(0, n_paths, path_batch_size):
            stop = min(start + path_batch_size, n_paths)
            theta, noise = [], []
            for path_id in range(start, stop):
                theta.append(pricing.conditional_theta(
                    i, paths["V"][path_id, i], paths["dW"][path_id, :i],
                    parameters["T"], N, parameters["H"], parameters["V0"],
                    parameters["nu"],
                ))
                noise.append(_future_noise(
                    n_inner, i, parameters, seed, path_id, inner_batch_size
                ))
            batch_holdings, batch_prices = _holdings_batch(
                stock[start:stop, i], forward[start:stop, i], np.asarray(theta),
                np.stack([item["Y"] for item in noise]),
                np.stack([item["dW"] for item in noise]),
                np.stack([item["dB"] for item in noise]),
                times, direction, model,
            )
            holdings[start:stop, i] = np.asarray(batch_holdings)
            prices[start:stop, i] = np.asarray(batch_prices)
        if progress:
            print(f"Hedged date {i + 1}/{N} ({n_paths:,} paths)", flush=True)

    return dict(
        JAX=holdings, prices=prices, n_inner=n_inner, seed=seed, method="JAX",
        inner_batch_size=inner_batch_size,
    )
