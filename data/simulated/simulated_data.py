import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import copy
import torch
from torch import nn

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
  
  #stock price process with EM scheme
  def stock_price_process(S0: float, T: float, N: int, rho: float, 
                          path: np.array, brownian_shocks: list):
    delta = T/N
    BM = np.asarray(brownian_shocks)
    indep_BM = norm.rvs(scale = np.sqrt(delta), size = N )
    stock_price = np.zeros(N + 1)
    stock_price[0] = S0
    for i in range(N):
      stock_price[i+1] = stock_price[i] + stock_price[i] * np.sqrt(path[i]) *(np.sqrt(1 - rho**2)*indep_BM[i] + rho*BM[i])
    return stock_price

#forward variance path

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

#architecture for each NN - will have 1 per trading/hedging day
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class MyModule(nn.Module):
    def __init__(self, N) -> None:
        super().__init__()
        self.linears = nn.ModuleList([NeuralNetwork() for _ in range(N)])

    #input x is 3d vector with values [path_number, day, data features]
    def forward(self, x):
      N = x.shape[1]
      result = []
      for i in range(N):
        data = x[:, i, :]
        hedge = self.linears[i](data)
        result.append(hedge)
      return torch.stack(result, dim = 1)