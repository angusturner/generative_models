import torch
from .initialization import kaiming, xavier

def logsumexp(x, dim, keepdim=False):
    """
    :param x:
    :param dim:
    :param keepdim:
    :return:
    """
    max, _ = torch.max(x, dim=dim, keepdim=True)
    out = max + (x - max).exp().sum(dim=dim, keepdim=keepdim).log()
    return out