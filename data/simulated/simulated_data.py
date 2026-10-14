import numpy as np
from scipy.stats import norm, multivariate_normal





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
        Ai, dwi = multivariate_normal.rvs(mean = [0, 0], cov = [[(delta**(2*alpha + 1))*(1/(2*alpha + 1)), (delta**(alpha + 1))/(alpha + 1)], [(delta**(alpha + 1))/(alpha + 1)], delta])

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

    return path



