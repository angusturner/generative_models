import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    """
    Gated Linear Unit from Dauphin et al 2016. https://arxiv.org/abs/1612.08083
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, dim=-1):
        """
        :param x: torch.Tensor (*, features)
        :return: torch.Tensor (*, features // 2)
        """
        return F.glu(x, dim=-1)