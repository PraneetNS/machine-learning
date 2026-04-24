import torch
import torch.nn as nn
import torch.fft as fft

class HyenaOperator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Learnable filter
        self.filter = nn.Parameter(torch.randn(d_model))
        
        # Gating
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # FFT-based convolution
        x_fft = fft.rfft(x, dim=1)
        filter_fft = fft.rfft(self.filter.unsqueeze(0).unsqueeze(0), n=L, dim=1)
        
        y = fft.irfft(x_fft * filter_fft, n=L, dim=1)
        
        g = torch.sigmoid(self.gate(x))
        return g * y