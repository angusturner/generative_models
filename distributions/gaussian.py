import numpy as np

log_norm_constant = -0.5 * np.log(2 * np.pi)

def log_gaussian(x, mean=0, logvar=0.):
    """
    Returns the component-wise density of x under the gaussian parameterised
    by `mean` and `logvar`
    :param x: (*) torch.Tensor
    :param mean: float or torch.FloatTensor with dimensions (*)
    :param logvar: float or torch.FloatTensor with dimensions (*)
    :param normalize: include normalisation constant?
    :return: (*) log density
    """
    if type(logvar) == 'float':
        logvar = x.new(1).fill_(logvar)

    a = (x - mean) ** 2
    log_p = -0.5 * (logvar + a / logvar.exp())
    log_p = log_p + log_norm_constant

    return log_p

def reparameterize(mu, logvar):
    """
    Draw a sample z ~ N(mu, std), such that it is differentiable
    with respect to `mu` and `logvar`, by sampling e ~ N(0, I) and
    performing a location-scale transform.
    :param mu: torch.Tensor
    :param logvar: torch.Tensor
    :return:
    """
    e = mu.new_empty(*mu.size()).normal_()
    std = (logvar * 0.5).exp()
    return mu + std * e