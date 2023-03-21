import numpy as np
import torch

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
    return torch.sum(logp, dim=[1, 2, 3])


def compute_lowerbound(log_pxz, sum_kl_costs, k=1):
    if k == 1:
        return sum_kl_costs - log_pxz
    # log 1/k \sum p(x | z) * p(z) / q(z | x) = -log(k) + logsumexp(log p(x|z) + log p(z) - log q(z|x))
    log_pxz = torch.reshape(log_pxz, [-1, k])
    sum_kl_costs = torch.reshape(sum_kl_costs, [-1, k])
    return -(-torch.log(torch.tensor(k)) + torch.logsumexp(log_pxz - sum_kl_costs, dim=1))
