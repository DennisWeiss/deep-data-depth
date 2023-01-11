import torch
import torch.nn as nn


class DeepDataDepthEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, act_fun: nn.Module = nn.ReLU()):
        super(DeepDataDepthEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_fun,
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)