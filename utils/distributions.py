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
    return logp.sum(dim=(1,2,3))