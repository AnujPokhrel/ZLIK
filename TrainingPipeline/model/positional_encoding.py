import math

import torch
import torch.nn as nn
import pdb


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 20, dropout: float = 0.2):
        super().__init__()
        # the data is in shape (B, T, L)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, m):
        torch.nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 20, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]  # broadcast on batch dim
        return self.dropout(x)

class TimePositionalEmbedding(nn.Module):
    def __init__(self, max_len: float = 10.0, d_model: int = 128):
        super().__init__()
        pdb.set_trace()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),     # input is scalar time
            nn.SiLU(),                 # or GELU, ReLU
            nn.Linear(d_model, d_model),
        )
        self.max_len = max_len # usually your horizon in seconds

    def forward(self, t: torch.Tensor):
        """
        t: (batch_size, seq_len) or (seq_len,) → will be normalized to [0,1]
        """
        # Normalize time to [0, 1] over the horizon
        t_normalized = t / self.max_len # now 0.0 to 1.0
        t_normalized = t_normalized.unsqueeze(-1)              # (B, L, 1) or (L, 1)
        return self.mlp(t_normalized.float())       # (B, L, d_model) or (1, L, d_model)

