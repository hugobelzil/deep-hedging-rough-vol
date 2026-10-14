import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import copy

# Riemann-Liouville representation of fBM
def fractional_BM_paths(alpha: float,T: float, N: int):

    if not(-1/2 < alpha <= 0):
        raise('Error : Hurst index must be in ]-1/2, 0]')

    brownian_shocks = []
    path = []
    delta = T/N

    # Helper function for coefficients of the fBM generation
    def b(k: int):
        weight = delta**alpha
        weight *= (k**(alpha + 1) - (k - 1)**(alpha + 1)) * (1 / (alpha + 1))
        return weight

    brownian_shocks = []
    weights = np.array([b(k) for k in range(2, N + 1)])
    path = []
    delta = T/N

    i = 0
    while i != N:
        Ai, dwi = multivariate_normal.rvs(mean = [0, 0], cov = [[(delta**(2*alpha + 1))*(1/(2*alpha + 1)), (delta**(alpha + 1))/(alpha + 1)], [(delta**(alpha + 1))/(alpha + 1), delta]])

        if i == 0:
            brownian_shocks.append(dwi)
            path.append(np.sqrt(2*alpha + 1) * Ai)
            i +=1
            continue

        else:
            Ai, dwi = multivariate_normal.rvs(mean = [0, 0], cov = [[(delta**(2*alpha + 1))*(1/(2*alpha + 1)), (delta**(alpha + 1))/(alpha + 1)], [(delta**(alpha + 1))/(alpha + 1), delta]])
            k = i + 1
            subset_shock = brownian_shocks[:i][::-1]
            Ai += weights[:i] @ subset_shock
            path.append(np.sqrt(2*alpha + 1)* Ai)
            brownian_shocks.append(dwi)
            i += 1

    return np.concatenate(([0.0], path)), brownian_shocks

# rough volatility process

def rough_vol_process(alpha: float, T: float, N: int, V0: float, 
                      nu: float, path: np.array, brownian_shocks: list):
    
    """Converts the paths from a RL fBM to the instantaneous variance process required for rBergomi""" 

    y = np.asarray(path)
    t = np.linspace(0.0, T, N + 1)
    return V0*np.exp(nu * y - 0.5 * nu**2 * t**(2 * alpha + 1))

    return path

def forward_variance_paths(alpha, T, N, V0, nu, T_fwd, brownian_shocks):
    if T_fwd <= T:
        raise ValueError("T_fwd must exceed T.")

    dW = np.asarray(brownian_shocks, dtype=float)
    if dW.shape != (N,):
        raise ValueError("Expected N Brownian increments.")

    t = np.arange(N) * (T / N)  # Left endpoints

    factors = 1 + (
        np.sqrt(2 * alpha + 1)
        * nu
        * (T_fwd - t)**alpha
        * dW
    )

    return V0*np.concatenate(([1.0], np.cumprod(factors)))




