import torch
import numpy as np
import utils.torch.modules as modules

def sample_from_logistic(mu, scale, shape, device, bound=1e-5):
    # sample untransformed sample from Logistic distribution (mu=0, scale=1)
    # sample from a Gaussian
    u = torch.rand(shape, device=device)

    # clamp between two bounds to ensure numerical stability
    u = torch.clamp(u, min=bound, max=1 - bound)

    # transform to a sample from the Logistic distribution
    eps = torch.log(u) - torch.log1p(-u)
    
    # reparameterization trick: transform "noise" using a given mean and scale
    sample = mu + scale * eps

    return sample

# function to calculate the log-probability of x under a Logistic(mu, scale) distribution
def logistic_logp(mu, scale, x):
    _y = -(x - mu) / scale
    logp = -_y - torch.log(scale) - 2 * modules.softplus(-_y)
    return logp # (B, C, H, W)

# function to calculate the log-probability of x under a discretized Logistic(mu, scale) distribution
# heavily based on discretized_mix_logistic_loss() in https://github.com/openai/pixel-cnn
def discretized_logistic_logp(mu, scale, x):
    # [0,255] -> [-1.1] (this means bin sizes of 2./255.)
    x_rescaled = (x - 127.5) / 127.5
    invscale = 1. / scale

    x_centered = x_rescaled - mu

    plus_in = invscale * (x_centered + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = invscale * (x_centered - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # log-probability for edge case of 0
    log_cdf_plus = plus_in - modules.softplus(plus_in)

    # log-probability for edge case of 255
    log_one_minus_cdf_min = - modules.softplus(min_in)

    # other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = invscale * x_centered

    # log-probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - torch.log(scale) - 2. * modules.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal case, extremely low-probability case
    cond1 = torch.where(cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-12, max=None)),
                        log_pdf_mid - np.log(127.5))
    cond2 = torch.where(x_rescaled > .999, log_one_minus_cdf_min, cond1)
    logps = torch.where(x_rescaled < -.999, log_cdf_plus, cond2)

    return logps


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape)-1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x-m), dim=axis, keepdim=True))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.shape)-1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x-m2), dim=axis))

def discretized_mix_logistic_logp(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # [0,255] -> [-1.1] (this means bin sizes of 2./255.)
    x = (x - 127.5) / 127.5
    x = x.permute(0, 2, 3, 1) # x:(B, C, H, W) -> (B, H, W, C)
    l = l.permute(0, 2, 3, 1) # l:(B, C * (3 * nr_mix), H, W) -> (B, H, W, C * (3 * nr_mix))
    xs = list(x.size()) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = list(l.size()) # predicted distribution, e.g. (B,32,32,100)

    assert (ls[-1]/xs[-1]) % 3 == 0, \
        f"Error: Invalid shape: {ls}. Expected [batch_size, 3 * num_mixtures * C, H, W]"

    nr_mix = int((ls[-1]/xs[-1]) / 3)
    l = l.reshape(xs + [3 * nr_mix]) # (B, H, W, C, (3 * nr_mix))

    # below: unpacking the params of the mixture of logistics
    logit_probs, means, log_scales = torch.split(l, nr_mix, dim=-1) # (B, H, W, C, nr_mix)
    log_scales = torch.clamp(log_scales, -7.)

    # below: getting the means and adjusting them based on preceding sub-pixels
    x = x.unsqueeze(-1).repeat(1, 1, 1, 1, nr_mix)
    
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - modules.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = - modules.softplus(min_in)
    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * modules.softplus(mid_in)
    
    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)
    log_probs = torch.where(x < -0.999, log_cdf_plus, 
                            torch.where(x > 0.999, log_one_minus_cdf_min, 
                                        torch.where(cdf_delta > 1e-5, 
                                                    torch.log(torch.clamp(cdf_delta, min=1e-12, max=None)), 
                                                    log_pdf_mid - np.log(127.5))))
    
    log_probs = log_probs + log_prob_from_logits(logit_probs) # (B, H, W, C, nr_mix)
    
    return log_sum_exp(log_probs) # (B, H, W, C)


# def mixture_discretized_logistic_cdf(x, l):
#     x = x.permute(0, 2, 3, 1) # x:(B, C, H, W) -> (B, H, W, C)
#     l = l.permute(0, 2, 3, 1) # l:(B, C * (3 * nr_mix), H, W) -> (B, H, W, C * (3 * nr_mix))
#     xs = list(x.size()) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
#     ls = list(l.size()) # predicted distribution, e.g. (B,32,32,100)    nr_mix = int((ls[-1]/xs[-1]) / 3)
#     nr_mix = int((ls[-1]/xs[-1]) / 3)
#     l = l.reshape(xs + [3 * nr_mix]) # (B, H, W, C, (3 * nr_mix))
#     logit_probs, means, log_scales = torch.split(l, nr_mix, dim=-1) # (B, H, W, C, nr_mix)
#     x = x.unsqueeze(-1).repeat(1, 1, 1, 1, nr_mix)
#     centered_x = x - means
#     inv_stdv = torch.exp(-log_scales)
#     plus_in = inv_stdv * (centered_x + 1./255.)
#     cdf_plus = torch.sigmoid(plus_in)
#     log_probs = torch.log(cdf_plus) + log_prob_from_logits(logit_probs)
#     return log_sum_exp(log_probs)

# def sample_from_discretized_mix_logistic(l, nr_mix):
#     ls = l.shape
#     xs = ls[:-1] + (3,)
#     # unpack parameters
#     logit_probs = l[..., :nr_mix]
#     l = l[..., nr_mix:].reshape(xs + (nr_mix*3,))
#     # sample mixture indicator from softmax
#     sel = F.one_hot(torch.argmax(logit_probs - torch.log(-torch.log(torch.rand_like(logit_probs) * (1 - 2e-5) + 1e-5)), dim=3), num_classes=nr_mix).float()
#     sel = sel.reshape(xs[:-1] + (1, nr_mix))
#     # select logistic parameters
#     means = (l[..., :nr_mix] * sel).sum(dim=4)
#     log_scales = torch.clamp((l[..., nr_mix:2*nr_mix] * sel).sum(dim=4), min=-7.)
#     coeffs = torch.tanh(l[..., 2*nr_mix:3*nr_mix]) * sel
#     coeffs = coeffs.sum(dim=4)
#     # sample from logistic & clip to interval
#     # we don't actually round to the nearest 8bit value when sampling
#     u = torch.rand_like(means) * (1 - 2e-5) + 1e-5
#     x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1 - u))
#     x0 = torch.clamp(x[..., 0], min=-1., max=1.)
#     x1 = torch.clamp(x[..., 1] + coeffs[..., 0] * x0, min=-1., max=1.)
#     x2 = torch.clamp(x[..., 2] + coeffs[..., 1] * x0 + coeffs[..., 2] * x1, min=-1., max=1.)
#     return torch.cat([x0.unsqueeze(-1), x1.unsqueeze(-1), x2.unsqueeze(-1)], dim=3)


# function to calculate the CDF of the Logistic(mu, scale) distribution evaluated under x
def logistic_cdf(x, mu, scale):
    return torch.sigmoid((x - mu) / scale)

# function to calculate the inverse CDF (quantile function) of the Logistic(mu, scale) distribution evaluated under x
def logistic_icdf(p, mu, scale):
    return mu + scale * torch.log(p / (1. - p))

def get_pmfs(bin_ends, mu, scale):
    cdfs = logistic_cdf(bin_ends, mu, scale).t() # most expensive calculation?
    pmfs = cdfs[:, 1:] - cdfs[:, :-1]
    pmfs = torch.cat((cdfs[:,0].unsqueeze(1), pmfs, 1. - cdfs[:,-1].unsqueeze(1)), dim=1)
    return pmfs

# class that is used to determine endpoints and centers of discretization bins
# in which every bin has equal mass under some given Logistic(mu, scale) distribution.
# note: the first (-inf) and last (inf) endpoint are not created here, but rather
# accounted for in the compression/decompression loop
class Bins:
    def __init__(self, mu, scale, precision):
        # number of bits used
        self.precision = precision

        # the resulting number of bins from the amount of bits used
        self.nbins = 1 << precision

        # parameters of the Logistic distribution
        self.mu, self.scale = mu, scale

        # datatype used
        self.type = self.mu.dtype

        # device used (GPU/CPU)
        self.device = self.mu.device
        self.shape = list(self.mu.shape)

    def endpoints(self):
        # first uniformly between [0,1]
        # shape: [1 << bits]
        endpoint_probs = torch.arange(1., self.nbins, dtype=self.type, device=self.device) / self.nbins

        # reshape
        endpoint_probs = endpoint_probs[(None,) * len(self.shape)] # shape: [1, 1, 1<<bits]
        endpoint_probs = endpoint_probs.permute([-1] + list(range(len(self.shape)))) # shape: [1 << bits, 1, 1]
        endpoint_probs = endpoint_probs.expand([-1] + self.shape) # shape: [1 << bits] + self.shape

        # put those samples through the inverse CDF
        endpoints = logistic_icdf(endpoint_probs, self.mu, self.scale)

        # reshape
        endpoints = endpoints.permute(list(range(1, len(self.shape) + 1)) + [0]) # self.shape + [1 << bits]
        return endpoints

    def centres(self):
        # first uniformly between [0,1]
        # shape: [1 << bits]
        centre_probs = (torch.arange(end=self.nbins, dtype=self.type, device=self.device) + .5) / self.nbins

        # reshape
        centre_probs = centre_probs[(None,) * len(self.shape)] # shape: [1, 1, 1<<bits]
        centre_probs = centre_probs.permute([-1] + list(range(len(self.shape)))) # shape: [1 << bits, 1, 1]
        centre_probs = centre_probs.expand([-1] + self.shape) # shape: [1 << bits] + self.shape

        # put those samples through the inverse CDF
        centres = logistic_icdf(centre_probs, self.mu, self.scale)

        # reshape
        centres = centres.permute(list(range(1, len(self.shape) + 1)) + [0]) # self.shape + [1 << bits]
        return centres

# class that is used to determine the endpoints and center of bins discretized uniformly between [0,1]
# these bins are used for the discretized Logistic distribution during compression/decompression
# note: the first (-inf) and last (inf) endpoint are not created here, but rather
# accounted for in the compression/decompression loop
class ImageBins:
    def __init__(self, type, device, shape):
        # datatype used
        self.type = type

        # device used (CPU/GPU)
        self.device = device
        self.shape = [shape]

    def endpoints(self):
        endpoints = torch.arange(1, 256, dtype=self.type, device=self.device)
        endpoints = ((endpoints - 127.5) / 127.5) - 1./255.
        endpoints = endpoints[None,].expand(self.shape + [-1])
        return endpoints

    def centres(self):
        centres = torch.arange(0, 256, dtype=self.type, device=self.device)
        centres = (centres - 127.5) / 127.5
        centres = centres[None,].expand(self.shape + [-1])
        return centres