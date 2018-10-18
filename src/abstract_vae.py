import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractVAE(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def likelihood(self, x, params, *args, **kwargs):
        """
        Compute the conditional likelihood of a data point p(x|z)
        :param x: data point torch.Tensor
        :param params:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def posterior(self, z, params, *args, **kwargs):
        """
        Compute the log probability of a latent sample z ~ q(z|x) under
        the posterior.
        :param z: latent variable
        :param params: parameters of the posterior distribution
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def prior(self, z, params, *args, **kwargs):
        """
        Get the log probability of a latent sample under the prior p(z)
        :param z:
        :param params:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, x, *args, **kwargs):
        """
        Map a data point to the parameters of the latent posterior.
        Note: if normalising flows are used, the returned parameters
        correspond to the base posterior `q0`
        :param x:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, z, *args, **kwargs):
        """
        Map a latent value to the parameters of the likelihood
        :param z:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, x, mean_only=True, *args, **kwargs):
        """
        Infer the latent value of a given data point x -> z
        :param x: data point
        :param mean_only: whether to use the mean of the posterior or draw a random sample
        :return:
        """
        raise NotImplementedError