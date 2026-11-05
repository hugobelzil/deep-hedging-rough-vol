import torch
from torch import nn

class LSTMHedger(nn.Module):
    def __init__(self, hidden_size = 32, strike = 100, log_moneyness = True):
         # what does nn.module do ?
        super().__init__()
        self.cell = nn.LSTMCell(input_size=2, hidden_size=hidden_size)
        self.hidden_to_hedge = nn.Linear(hidden_size, 2)
        self.strike = strike
        self.log_moneyness = bool(log_moneyness) # forces true/false ?

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

        if self.log_moneyness == True:
            spot = torch.log(spot / self.strike)
    
        inputs = torch.stack((spot, forward_var), dim=-1)

        holdings = []
        for k in range(n_dates - 1):
            h, c = self.cell(inputs[:, k, :], (h, c))
            holdings.append(self.hidden_to_hedge(h))

        return torch.stack(holdings, dim=1) # [batch, N, 2]


def hedging_loss(model, prices, strike, option_price):
    """ Quadratic hedging objective from the paper """
    holdings = model(prices)                       # [batch, N, 2]
    price_changes = prices[:, 1:] - prices[:, :-1]  # [batch, N, 2]

    trading_gain = (holdings * price_changes).sum(dim=(1, 2))
    payoff = torch.clamp(prices[:, -1, 0] - strike, min=0.0)

    pnl = option_price + trading_gain - payoff
    return pnl.square().mean()


