import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import copy

# Riemann-Liouville representation of fBM
def fractional_BM_paths(alpha: float,T: float, N: int):
    if not(-1/2 < alpha <= 0):
        raise ValueError('Error : Hurst index must be in ]-1/2, 0]')
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

import torch
from torch import nn

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

#define parameters
T = 30/365
N = 30
T_fwrd = 45/365
S0 = 100
K = 100
rho = -0.7
alpha = - 0.1
V0 = 0.235*0.235
nu = 1.9
batch_size = 1024

# func that generates batch given parameters and size and returns lists holding
def generate_batch(batch_size: int, alpha: float, T: float, N: int, T_fwrd: float, 
                   S0: float, V0: float, nu: float, rho: float):
  batch_features = []
  batch_prices = []
  time_data = np.linspace(0, T, N + 1)

  for _ in range(batch_size):
    path_initial, brownian_shocks = fractional_BM_paths(alpha, T, N)
    path = rough_vol_process(alpha, T, N, V0, nu, path_initial, brownian_shocks)
    stock_data = stock_price_process(S0, T, N, rho, path, brownian_shocks)
    forward_variance_data = forward_variance_paths(alpha, T, N, V0, nu, T_fwrd, brownian_shocks)
    features = np.column_stack((stock_data[:-1], forward_variance_data[:-1], time_data[:-1]))
    batch_features.append(features)
    prices = np.column_stack((stock_data, forward_variance_data))
    batch_prices.append(prices)
  x_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32)
  prices_tensor = torch.tensor(np.array(batch_prices), dtype=torch.float32)

  return x_tensor, prices_tensor
    
x_train, prices_train = generate_batch(batch_size, alpha, T, N, T_fwrd, S0, V0, nu, rho)

# loss function
# weights has shape [batch_size, N, 2]
# prices shape [batch_size, N+1, 2]
def loss_function(p0: float, weights: torch.Tensor, prices: torch.Tensor, K: float):
    price_changes = torch.diff(prices, dim=1) 
    daily_profits = weights * price_changes
    total_hedging_pnl = torch.sum(daily_profits, dim=(1, 2)) 
    S_T = prices[:, -1, 0] 
    Z = torch.nn.functional.relu(S_T - K)
    final_portfolio_value = -Z + p0 + total_hedging_pnl
    loss = torch.mean(final_portfolio_value ** 2)
    return loss

def train_loop(epochs, model, optimizer, batch_size, loss_fn, p0, K, alpha, T, N, T_fwrd, S0, V0, nu, rho):
    model.train()
    for epoch in range(epochs):
      features, prices = generate_batch(batch_size, alpha, T, N, T_fwrd, S0, V0, nu, rho)
      # Compute prediction and loss
      weights = model(features)
      loss = loss_fn(p0, weights, prices, K)

      # Backpropagation
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()


def test_loop(model, loss_fn, test_size: int, alpha: float, T, N, T_fwrd, S0, V0, nu, rho, p0, K):
    model.eval()
    test_loss = 0
    with torch.no_grad():
      features, prices = generate_batch(test_size, alpha, T, N, T_fwrd, S0, V0, nu, rho)
      weights = model(features)
      test_loss += loss_fn(p0, weights, prices, K).item()
    print(f"Avg loss: {test_loss:>8f} \n")

model = MyModule(N)
learning_rate = 5e-3
epochs = 100
test_size = 5000

#change later by simualtion ourselves!!!! this is for H = 0.1
p0 = 2.39

loss_fn = loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loop(epochs, model, optimizer, batch_size, loss_fn, p0, K, alpha, T, N, T_fwrd, S0, V0, nu, rho)
test_loop(model, loss_fn, test_size, alpha, T, N, T_fwrd, S0, V0, nu, rho, p0, K)