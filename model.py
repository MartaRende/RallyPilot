import torch
import torch.nn as nn

from data_loader import SPEED_WEIGHT


class MLP(nn.Module):

    def __init__(self, device):
        super(MLP, self).__init__()
        self.register_parameter(
            "SPEED_WEIGHT",
            torch.nn.Parameter(torch.Tensor([SPEED_WEIGHT]), requires_grad=False),
        )
        self.MLP = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )
        self.device = device
        self.MLP = self.MLP.to(device)

    def forward(self, x):
        out = self.MLP(x)
        return out
