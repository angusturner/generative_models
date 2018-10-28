import torch
import torch.nn as nn

from src.utils import xavier
from src.nn import GLU

class MLP(nn.Module):
    def __init__(self, input=784, hidden=256, output=40, nb_layers=2, dropout=0., **kwargs):
        """
        A simple densely connected neural net, with Gated activations.
        """
        super().__init__()

        # input -> hidden
        layers = [
            xavier(nn.Linear(input, hidden * 2)),
            GLU(),
            nn.Dropout(p=dropout)
        ]

        # hidden -> hidden
        for _ in range(nb_layers - 1):
            layers += [
                xavier(nn.Linear(hidden, hidden * 2)),
                GLU(),
                nn.Dropout(p=dropout)
            ]

        # hidden -> out
        layers.append(
            xavier(nn.Linear(hidden, output))
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: torch.Tensor (batch, input)
        :return: torch.Tensor (batch, output)
        """
        return self.layers(x)