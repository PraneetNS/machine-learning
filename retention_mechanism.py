import torch
import torch.nn as nn

class Retention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.gamma = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        # x: (B, L, D)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        retention = 0
        outputs = []
        
        for t in range(x.shape[1]):
            retention = torch.exp(-torch.abs(self.gamma)) * retention + K[:, t, :] * V[:, t, :]
            yt = Q[:, t, :] * retention
            outputs.append(yt.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)