import numpy as np
import torch
from torch import lgamma


def beta_binomial_log_pdf(k, n, alpha, beta):
    numer = lgamma(n+1) + lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(alpha + beta)
    denom = lgamma(k+1) + lgamma(n - k + 1) + lgamma(n + alpha + beta) + lgamma(alpha) + lgamma(beta)
    return numer - denom

def generate_beta_binomial_probs(a, b, n):
    ks = torch.tensor(np.arange(n + 1))
    a = a[..., np.newaxis].detach().cpu()
    b = b[..., np.newaxis].detach().cpu()
    probs = np.exp(beta_binomial_log_pdf(ks, n, a, b).numpy())
    # make sure normalised, there are some numerical
    # issues with the exponentiation in the beta binomial
    probs = np.clip(probs, 1e-10, 1.)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = torch.randn(mean.shape, device=mean.device)
        sample = mean + torch.exp(0.5 * logvar) * noise

    return -0.5 * (np.log(2 * np.pi) + logvar + (sample - mean) ** 2 / torch.exp(logvar))


class DiagonalGaussian(object):

    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar
        self.std = torch.exp(0.5 * logvar)

        if sample is None:
            noise = torch.randn(mean.shape, device=mean.device)
            sample = mean + self.std * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)


def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    logp = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
    return logp.sum(dim=(1,2,3))