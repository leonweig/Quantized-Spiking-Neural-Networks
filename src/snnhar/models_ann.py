from __future__ import annotations
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
