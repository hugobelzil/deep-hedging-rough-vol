"""Settings for the theoretical rBergomi hedge notebook."""

# Model, contract and trading grid.
S0, K = 100.0, 100.0
T, R, N = 30 / 365, 45 / 365, 30
H, V0, nu, rho = 0.10, 0.235**2, 1.9, -0.7

# Monte Carlo
n_outer = 1_000
n_premium = 1_000
n_inner_full = 1_000

pricing_batch_size = 1_000
inner_batch_size = 1_000
path_batch_size = 16

# Hurst parameters used in the full-hedge comparison.
hurst_values = (0.1, 0.2, 0.3, 0.4)

# I put the dates as seed so we can reproduce, will specifiy in paper (maybe)
seeds = dict(
    market=2026110301, premium=2026110302, full_hedge=2026110305,
)
