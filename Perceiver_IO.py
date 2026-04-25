import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, query, context):
        Q = self.q(query)
        K = self.k(context)
        V = self.v(context)
        
        attn = torch.softmax(Q @ K.transpose(-1, -2) / (Q.shape[-1]**0.5), dim=-1)
        return attn @ V


class PerceiverLike(nn.Module):
    def __init__(self, input_dim, latent_dim, num_latents=32):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = CrossAttention(latent_dim)

    def forward(self, x):
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)
        
        latents = self.cross_attn(latents, x)
        return latents