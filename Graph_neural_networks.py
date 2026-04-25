import torch
import torch.nn as nn

class GNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, adj):
        # x: (N, D), adj: (N, N)
        agg = adj @ x
        return torch.relu(self.linear(agg))