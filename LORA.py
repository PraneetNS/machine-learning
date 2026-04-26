import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        self.A = nn.Parameter(torch.randn(rank, in_features))
        self.B = nn.Parameter(torch.randn(out_features, rank))

    def forward(self, x):
        base = x @ self.weight.T
        lora = x @ self.A.T @ self.B.T
        return base + lora