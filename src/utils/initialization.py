"""
Thin wrappers on torch.nn.init that allow both layers and tensors
to be passed in.
"""
import torch
from torch.nn.init import xavier_normal, kaiming_normal
from functools import partial

def _init_any(x, init_fn=None, *args, **kwargs):
    if torch.is_tensor(x):
        return init_fn(*args, **kwargs)
    elif hasattr(x, 'weight'):
        init_fn(x.weight)
        return x
    raise Exception("Oops, wrong type passed to init function.")

kaiming = partial(_init_any, init_fn=kaiming_normal)
xavier = partial(_init_any, init_fn=xavier_normal)