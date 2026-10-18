import torch
from torch import nn

class LSTMHedger(nn.Module):
    def __init__(self, hidden_size = 32, strike = 100):
         # what does nn.module do ?
        super().__init__()
        self.cell = nn.LSTMCell(input_size=3, hidden_size=hidden_size)
        self.hidden_to_hedge = nn.Linear(hidden_size, 2)
        self.strike = strike

    def forward(self, prices):
        """assuming the current structure : prices : [batch, N+1, 2]
                                            prices [..., 0] = spot
                                            prices[..., 1] = forward variance 
        and returns : holdings [batch, N, 2]"""

        batch_size, n_dates, _ = prices.shape
        h = prices.new_zeros(batch_size, self.cell.hidden_size)
        c = prices.new_zeros(batch_size, self.cell.hidden_size)

        # scaling : working in log-moneyness units
        spot = prices[:, :, 0]
        forward_var = prices[:, :, 1]

        log_moneyness = torch.log(spot / self.strike)
        inputs = torch.stack((log_moneyness, forward_var), dim=-1)

        holdings = []
        for k in range(n_dates - 1):
            h, c = self.cell(inputs[:, k, :], (h, c))
            holdings.append(self.hidden_to_hedge(h))

        return torch.stack(holdings, dim = -1) # [batch, N, 2]