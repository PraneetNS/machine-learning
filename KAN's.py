import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.splines = nn.Parameter(torch.randn(dim, 10))  # 10 basis functions

    def forward(self, x):
        # x: (B, D)
        # simulate spline interpolation
        basis = torch.sin(x.unsqueeze(-1) * torch.arange(1, 11).float())
        out = (basis * self.splines).sum(dim=-1)
        return out