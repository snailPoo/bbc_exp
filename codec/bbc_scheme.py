import numpy as np

from utils.torch.rand import get_pmfs
from .entropy_model import *

class bbc_base:
    def __init__(self, config, model, state, x_bin, z_bin):
        self.model = model
        self.state = state

        self.z_bin_ends, self.z_bin_centres = z_bin
        self.x_bin_ends, self.x_bin_centres = x_bin
        self.xrange = torch.arange(np.prod(model.xdim))
        self.zrange = torch.arange(np.prod(model.zdim))
        self.z_quantbits = config.z_quantbits
        self.x_quantbits = config.x_quantbits
        self.restbits = None
        self.restbits_tag = True
        self.total_xdim = np.prod(self.model.xdim)
        self.prior_mu = torch.zeros(1, device=config.device, dtype=config.type)
        self.prior_scale = torch.ones(1, device=config.device, dtype=config.type)

    def variable_encode(self, to_encode, model_op, i, given, bin_ends, quantbits):
        ops = self.model.generate if model_op == 'gen' else self.model.infer
        mu, scale = ops(i)(given=given)
        pmfs = get_pmfs(bin_ends.t(), mu, scale)
        self.state = ANS(pmfs, quantbits).encode(self.state, to_encode)
        
    def variable_decode(self, model_op, i, given, bin_ends, quantbits):
        ops = self.model.generate if model_op == 'gen' else self.model.infer
        mu, scale = ops(i)(given=given)
        pmfs = get_pmfs(bin_ends.t(), mu, scale)
        self.state, z_symtop = ANS(pmfs, quantbits).decode(self.state)
        return z_symtop
    
    def prior_encode(self, z, bin_ends):
        pmfs = get_pmfs(bin_ends.t(), self.prior_mu, self.prior_scale)
        self.state = ANS(pmfs, self.z_quantbits).encode(self.state, z)
    
    def prior_decode(self, bin_ends):
        pmfs = get_pmfs(bin_ends.t(), self.prior_mu, self.prior_scale)
        self.state, z_symtop = ANS(pmfs, self.z_quantbits).decode(self.state)
        return z_symtop

# vANS mode Logistic_UnifBins, setting lb & ub manually
# P(X > a) = 1 / [1 + (e^(a-mu) / scale)]
# let t is threshold, ub = mu + ln(scale * (1/t - 1))
# lb = mu - (ub - mu)
# should customize encoding/decoding process based on latent variable dependency graph
class BitSwap(bbc_base):
    def __init__(self, config, model, state, x_bin, z_bin):
        # bbc_base.__init__(self, config, model, state, x_bin, z_bin)
        super().__init__(config, model, state, x_bin, z_bin)

    def encoding(self, x):
        x = x.view(self.total_xdim)
        for zi in range(self.model.nz):
            given = self.z_bin_centres[zi - 1, self.zrange, z_sym] if zi > 0 else self.x_bin_centres[self.xrange, x.long()]
            z_symtop = self.variable_decode('inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            
            # save excess bits for calculations
            if self.restbits_tag:
                self.restbits = self.state.copy()
                assert len(self.restbits) > 1, "too few initial bits" # otherwise initial state consists of too few bits
                self.restbits_tag = False            

            given = self.z_bin_centres[zi, self.zrange, z_symtop] # z
            if zi == 0:
                self.variable_encode(x.long(), 'gen', zi, given, self.x_bin_ends, self.x_quantbits)
            else:
                self.variable_encode(z_sym, 'gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)

            z_sym = z_symtop

        # encode prior
        self.prior_encode(z_symtop, self.z_bin_ends[-1])
        return self.state
    
    def decoding(self):
        z_symtop = self.prior_decode(self.z_bin_ends[-1])
        
        for zi in reversed(range(self.model.nz)):
            given = self.z_bin_centres[zi, self.zrange, z_symtop]
            if zi == 0:
                sym = self.variable_decode('gen', zi, given, self.x_bin_ends, self.x_quantbits)
                given = self.x_bin_centres[self.xrange, sym]
            else:
                sym = self.variable_decode('gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)
                given = self.z_bin_centres[zi - 1, self.zrange, sym]

            self.variable_encode(z_symtop, 'inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            
            z_symtop = sym

        return z_symtop, self.state

class BBC(bbc_base):
    def __init__(self, config, model, state, x_bin, z_bin):
        # bbc_base.__init__(self, config, model, state, x_bin, z_bin)
        super(BBC, self).__init__(config, model, state, x_bin, z_bin)
    
    def encoding(self, x):
        x = x.view(self.total_xdim)
        zs = []
        # decode all latent variables
        for zi in range(self.model.nz):
            given = self.z_bin_centres[zi - 1, self.zrange, z_sym] if zi > 0 else self.x_bin_centres[self.xrange, x.long()]
            z_symtop = self.variable_decode('inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            zs.append(z_symtop)
            z_sym = z_symtop

        # save excess bits for calculations
        if self.restbits_tag:
            self.restbits = self.state.copy()
            assert len(self.restbits) > 1, "too few initial bits" # otherwise initial state consists of too few bits
            self.restbits_tag = False

        # decode latent variables and data x
        for zi in range(self.model.nz):
            z_symtop = zs.pop(0)
            given = self.z_bin_centres[zi, self.zrange, z_symtop]
            if zi == 0:
                self.variable_encode(x.long(), 'gen', zi, given, self.x_bin_ends, self.x_quantbits)
            else:
                self.variable_encode(z_sym, 'gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)
            z_sym = z_symtop

        assert zs == []
        self.prior_encode(z_symtop, self.z_bin_ends[-1])

    def decoding(self):
        z_symtop = self.prior_decode(self.z_bin_ends[-1])
        zs = [z_symtop]
        for zi in reversed(range(self.model.nz)):
            given = self.z_bin_centres[zi, self.zrange, z_symtop]
            if zi == 0:
                sym = self.variable_decode('gen', zi, given, self.x_bin_ends, self.x_quantbits)
            else:
                sym = self.variable_decode('gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)
            zs.append(sym)
            z_symtop = sym

        z_symtop = zs.pop(0)
        for zi in reversed(range(self.model.nz)):
            sym = zs.pop(0) if zi > 0 else zs[0]
            given = self.z_bin_centres[zi - 1, self.zrange, sym] if zi > 0 else self.x_bin_centres[self.xrange, sym]
            self.variable_encode(z_symtop, 'inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            z_symtop = sym
        
        return z_symtop, self.state

import craystack as cs
from craystack.bb_ans import BBANS
import torch.nn.functional as F
from craystack.codecs import substack, Uniform, \
    std_gaussian_centres, DiagGaussian_StdBins
from utils.distributions import generate_beta_binomial_probs

def VAE(config, model):
    """
    This codec uses the BB-ANS algorithm to code data which is distributed
    according to a variational auto-encoder (VAE) model. It is assumed that the
    VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
    posterior.
    """
    prior_prec = config.prior_precision
    latent_prec = config.q_precision
    obs_precision = config.obs_precision

    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior = substack(Uniform(prior_prec), z_view)
    prior_inv_cdf = std_gaussian_centres(prior_prec)

    device = config.device

    @torch.no_grad()
    def likelihood(latent_idxs):
        z = prior_inv_cdf[latent_idxs]
        x_alpha, x_beta = model.decode(torch.from_numpy(z.astype(np.float32)).to(device))
        x = (x_alpha.view((config.batch_size,)+model.xdim), x_beta.view((config.batch_size,)+model.xdim))
        obs_codec = lambda x: cs.Categorical(generate_beta_binomial_probs(*x, torch.tensor(255)), obs_precision)
        return substack(obs_codec(x), x_view)
    
    @torch.no_grad()
    def posterior(data):
        x = torch.from_numpy(data.astype(np.float32)).to(device)
        post_mean, post_stdd = model.encode(x)
        return substack(DiagGaussian_StdBins(
                            post_mean.cpu().numpy(), 
                            post_stdd.cpu().numpy(), 
                            latent_prec, prior_prec), 
                        z_view)
    return BBANS(prior, likelihood, posterior)

def BitSwap_vANS(config, model):
    t = config.bound_threshold
    coding_prec = config.coding_prec
    bin_prec = config.bin_prec
    obs_precision = config.obs_precision

    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    device = config.device

    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)
    
    def push(state, data):
        data = (data - 127.5) / 127.5
        x = to_tensor(data)
        state_len_before_pop = get_init_cost(state)
        for zi in range(model.nz):
            mu, scale = model.infer(zi)(given=x if zi == 0 else to_tensor(z))
            # pmfs = get_pmfs(self.z_bin_ends[zi].t(), mu, scale)
            # self.state, z_symtop = ANS(pmfs, self.z_quantbits).decode(self.state)     

            # vANS mode Logistic_UnifBins, setting lb & ub manually
            # P(X > a) = 1 / [1 + (e^(a-mu) / scale)]
            ub = mu + torch.log(scale * (1/t - 1))
            lb = mu - (ub - mu)
            codec = cs.Logistic_UnifBins(mu.detach().cpu().numpy(), 
                                         torch.log(scale).detach().cpu().numpy(),
                                         coding_prec, bin_prec,
                                         bin_lb=torch.min(lb).item(), 
                                         bin_ub=torch.max(ub).item())
            posterior = substack(codec, z_view)
            state, z_next = posterior.pop(state)
            
            if zi == 0:
                state_len_after_pop = get_init_cost(state)
                init_cost = 32 * (state_len_before_pop - state_len_after_pop)
                model.init_cost_record += init_cost
                
                mu, scale = model.generate(zi)(given=to_tensor(z_next))
                # pmfs = get_pmfs(self.x_bin_ends.t(), mu, scale)
                # self.state = ANS(pmfs, self.x_quantbits).encode(self.state, x.long())
                codec = cs.Logistic_UnifBins(mu.detach().cpu().numpy(), 
                                             torch.log(scale).detach().cpu().numpy(),
                                             obs_precision, bin_prec=8,
                                             bin_lb=-1, bin_ub=1)
                likelihood = substack(codec, x_view)
                state, = likelihood.push(state, data)

            else:
                mu, scale = model.generate(zi)(given=to_tensor(z_next))
                # pmfs = get_pmfs(self.z_bin_ends[zi - 1].t(), mu, scale)
                # self.state = ANS(pmfs, self.z_quantbits).encode(self.state, z_sym)

                ub = mu + torch.log(scale * (1/t - 1))
                lb = mu - (ub - mu)
                codec = cs.Logistic_UnifBins(mu.detach().cpu().numpy(), 
                                             torch.log(scale).detach().cpu().numpy(),
                                             coding_prec, bin_prec,
                                             bin_lb=torch.min(lb).item(), 
                                             bin_ub=torch.max(ub).item())
                likelihood = substack(codec, z_view)
                state, = likelihood.push(state, z)

            z = z_next

        # encode prior
        # pmfs = get_pmfs(self.z_bin_ends[-1].t(), self.prior_mu, self.prior_scale)
        # self.state = ANS(pmfs, self.z_quantbits).encode(self.state, z_symtop)
        mu = 0
        scale = 1
        ub = mu + np.log(scale * (1/t - 1))
        lb = mu - (ub - mu)
        codec = cs.Logistic_UnifBins(mu, np.log(scale),
                                     coding_prec, bin_prec,
                                     bin_lb=lb, bin_ub=ub)
        prior = substack(codec, z_view)
        state, = prior.push(state, z)
    
        return state, 
    
    def pop(state):
        mu = 0
        scale = 1
        ub = mu + np.log(scale * (1/t - 1))
        lb = mu - (ub - mu)
        codec = cs.Logistic_UnifBins(mu, np.log(scale),
                                     coding_prec, bin_prec,
                                     bin_lb=lb, bin_ub=ub)
        prior = substack(codec, z_view)
        state, z_next = prior.pop(state)

        for zi in reversed(range(model.nz)):
            if zi == 0:
                mu, scale = model.generate(zi)(given=to_tensor(z_next))
                # pmfs = get_pmfs(self.x_bin_ends.t(), mu, scale)
                # self.state = ANS(pmfs, self.x_quantbits).encode(self.state, x.long())
                codec = cs.Logistic_UnifBins(mu.detach().cpu().numpy(), 
                                             torch.log(scale).detach().cpu().numpy(),
                                             obs_precision, bin_prec=8,
                                             bin_lb=-1, bin_ub=1)
                likelihood = substack(codec, x_view)
                state, data = likelihood.pop(state)

            else:
                mu, scale = model.generate(zi)(given=to_tensor(z_next))
                # pmfs = get_pmfs(self.z_bin_ends[zi - 1].t(), mu, scale)
                # self.state = ANS(pmfs, self.z_quantbits).encode(self.state, z_sym)

                ub = mu + torch.log(scale * (1/t - 1))
                lb = mu - (ub - mu)
                codec = cs.Logistic_UnifBins(mu.detach().cpu().numpy(), 
                                             torch.log(scale).detach().cpu().numpy(),
                                             coding_prec, bin_prec,
                                             bin_lb=torch.min(lb).item(), 
                                             bin_ub=torch.max(ub).item())
                likelihood = substack(codec, z_view)
                state, z = likelihood.pop(state)

            mu, scale = model.infer(zi)(given=x if zi == 0 else to_tensor(z))
            ub = mu + torch.log(scale * (1/t - 1))
            lb = mu - (ub - mu)
            codec = cs.Logistic_UnifBins(mu.detach().cpu().numpy(), 
                                         torch.log(scale).detach().cpu().numpy(),
                                         coding_prec, bin_prec,
                                         bin_lb=torch.min(lb).item(), 
                                         bin_ub=torch.max(ub).item())
            posterior = substack(codec, z_view)
            state, = posterior.push(state, z_next)

            z_next = z

        data = data * 127.5 + 127.5

        return state, data

    return cs.Codec(push, pop)

def get_init_cost(state):
    state = list(state)
    state[0] = np.concatenate((state[0][0].flatten(), state[0][1].flatten()))
    state = tuple(state)
    return len(cs.flatten(state))
    
def ResNetVAE(config, model):
    """
    Codec for a ResNetVAE.
    Assume that the posterior is bidirectional -
    i.e. has a deterministic upper pass but top down sampling.
    Further assume that all latent conditionals are factorised Gaussians,
    both in the generative network p(z_n|z_{n-1})
    and in the inference network q(z_n|x, z_{n-1})

    Assume that everything is ordered bottom up
    """
    prior_prec = config.prior_precision
    latent_prec = config.q_precision
    obs_precision = config.obs_precision

    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior_codec = substack(Uniform(prior_prec), z_view)
    prior_inv_cdf = std_gaussian_centres(prior_prec)

    device = config.device

    def prior_push(state, latents):
        # print('in prior_push')
        # print('')
        # print('')
        # push bottom-up z_1 -> z_L
        latents, _ = latents
        for latent_idx in latents:
            state, = prior_codec.push(state, latent_idx)
        return state,

    @torch.no_grad()
    def prior_pop(state):
        # print('in prior_pop')
        # run the model top-down to get the params and latent vals
        latents = []
        post_params = []
        
        # input initalization
        input = model.h.view(1, -1, 1, 1).expand((config.batch_size, 
                                                  model.h_size, 
                                                  model.xdim[1] // 2, 
                                                  model.xdim[2] // 2)).to(device)

        for layer in reversed(model.layers):
            for zl in reversed(layer):
                h = zl.down_conv_a(F.elu(input))
                pz_mean, pz_logsd, rz_mean, rz_logsd, _, h_det = h.split([zl.z_size] * 4 + [zl.h_size] * 2, 1)

                prior_mean = pz_mean
                prior_stdd = torch.exp(pz_logsd)
                # pop top-down z_L -> z_1
                state, latent_idx = prior_codec.pop(state)
                latent_val = prior_mean + torch.from_numpy(prior_inv_cdf[latent_idx]).to(device) * prior_stdd
                h = torch.cat((latent_val, h_det), 1).float()
                h = zl.down_conv_b(F.elu(h))
                input = input + 0.1 * h

                latents.insert(0, (latent_idx, (prior_mean, prior_stdd)))
                post_params.insert(0, (rz_mean, rz_logsd))

        return state, ((latents, post_params), input)

    @torch.no_grad()
    def posterior(data):
        data = torch.from_numpy(data.astype(np.float32)).to(device)
        # assumes input is in [-0.5, 0.5] 
        x = torch.clamp((data + 0.5) / 256.0, 0.0, 1.0) - 0.5
        h = model.first_conv(x)
        # run deterministic upper-pass to get qz_mean and qz_std (bottom-up)
        for layer in model.layers:
            for sub_layer in layer:
                h = sub_layer.up(h)

        def posterior_push(state, latents):
            # print('in posterior_push')
            # print('')
            # print('')
            (latents, post_params), _ = latents
            
            for l, (latent, post_param) in enumerate(zip(latents, post_params)):
                zl = model.layers[0][l]
                latent_idx, (prior_mean, prior_stdd) = latent
                rz_mean, rz_logsd = post_param
                post_mean = zl.qz_mean + rz_mean
                post_stdd = torch.exp(zl.qz_logsd + rz_logsd)
                codec = substack(cs.DiagGaussian_GaussianBins(post_mean.cpu().numpy(), post_stdd.cpu().numpy(),
                                                              prior_mean.cpu().numpy(), prior_stdd.cpu().numpy(),
                                                              latent_prec, prior_prec),
                                 z_view)
                # push bottom up z_1 -> z_L
                state, = codec.push(state, latent_idx)
            return state,
        
        @torch.no_grad()
        def posterior_pop(state):
            # print('in posterior_pop')
            state_len_before_pop = get_init_cost(state)
            latents = []
            # input initalization
            input = model.h.view(1, -1, 1, 1).expand((config.batch_size, 
                                                      model.h_size, 
                                                      model.xdim[1] // 2, 
                                                      model.xdim[2] // 2)).to(device)
            kl = kl_true = 0.
            p = q = pt = qt = 0.
            for layer in reversed(model.layers):
                for zl in reversed(layer):
                    # down pass
                    h = zl.down_conv_a(F.elu(input))
                    pz_mean, pz_logsd, rz_mean, rz_logsd, _, h_det = h.split([zl.z_size] * 4 + [zl.h_size] * 2, 1)
                    
                    # calculate loss
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    '''
                    prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
                    posterior = DiagonalGaussian(zl.qz_mean + rz_mean, 2 * (zl.qz_logsd + rz_logsd))
                    z = posterior.sample
                    logqs = posterior.logps(z)
                    logps = prior.logps(z)
                    # print(f'{pz_mean=} {pz_logsd=}')
                    # print(f'{zl.qz_mean=} {zl.qz_logsd=}')
                    # print(f'{rz_mean=} {rz_logsd=}')
                    # print(f'log_q:{logqs.sum(dim=(1,2,3)).mean() / (np.log(2.) * 3072)}, log_p:{(-logps).sum(dim=(1,2,3)).mean() / (np.log(2.) * 3072)}')
                    kl += (logqs - logps).sum(dim=(1,2,3))
                    p -= logps.sum(dim=(1,2,3))
                    q += logqs.sum(dim=(1,2,3))
                    '''
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    prior_mean = pz_mean
                    prior_stdd = torch.exp(pz_logsd)
                    post_mean = zl.qz_mean + rz_mean
                    post_stdd = torch.exp(zl.qz_logsd + rz_logsd)
                    codec = substack(cs.DiagGaussian_GaussianBins(post_mean.cpu().numpy(), post_stdd.cpu().numpy(),
                                                                  prior_mean.cpu().numpy(), prior_stdd.cpu().numpy(),
                                                                  latent_prec, prior_prec),
                                     z_view)
                    # pop top-down z_L -> z_1
                    state, latent_idx = codec.pop(state)
                    latent_val = prior_mean + torch.from_numpy(prior_inv_cdf[latent_idx]).to(device) * prior_stdd
                    h = torch.cat((latent_val, h_det), 1).float()
                    h = zl.down_conv_b(F.elu(h))
                    input = input + 0.1 * h
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    '''
                    logqs = posterior.logps(latent_val)
                    logps = prior.logps(latent_val)
                    # print(f'real  log_q:{logqs.sum(dim=(1,2,3)).item() / (np.log(2.) * np.prod(data.shape))}, log_p:{(-logps).sum(dim=(1,2,3)).item() / (np.log(2.) * np.prod(data.shape))}')
                    kl_true += (logqs - logps).sum(dim=(1,2,3))
                    pt -= logps.sum(dim=(1,2,3))
                    qt += logqs.sum(dim=(1,2,3))
                    '''
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    latents.insert(0, latent_idx)
            
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            '''
            x_mu = model.last_conv(F.elu(input))
            x_mu = x_mu.clamp(min = -0.5 + 1. / 512., max = 0.5 - 1. / 512.)

            log_pxlz = discretized_logistic(x_mu, model.dec_log_stdv, sample=x)
            bpd = (kl - log_pxlz) / (np.log(2.) * np.prod(data.shape))
            bpd_true = (kl_true - log_pxlz) / (np.log(2.) * np.prod(data.shape))
            # print(f'log_pxlz:{(-log_pxlz).mean() / (np.log(2.) * np.prod(data.shape))}')
            bpd_p = p / (np.log(2.) * np.prod(data.shape))
            bpd_q = q / (np.log(2.) * np.prod(data.shape))
            bpd_pt = pt / (np.log(2.) * np.prod(data.shape))
            bpd_qt = qt / (np.log(2.) * np.prod(data.shape))
            bpd_x = (- log_pxlz) / (np.log(2.) * np.prod(data.shape))
            # print(f'train bpd:{bpd.item()}, p:{bpd_p.item()}, q:{bpd_q.item()}, x:{bpd_x.item()}')
            # print(f'real bpd:{bpd_true.item()}, p:{bpd_pt.item()}, q:{bpd_qt.item()}, x:{bpd_x.item()}')
            '''
            state_len_after_pop = get_init_cost(state)
            init_cost = 32 * (state_len_before_pop - state_len_after_pop)
            model.init_cost_record += init_cost
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            return state, (latents, input)

        return cs.Codec(posterior_push, posterior_pop)

    @torch.no_grad()
    def likelihood(latents):
        _, h = latents

        x_mu = model.last_conv(F.elu(h))
        x_mu = x_mu.clamp(min = -0.5 + 1. / 512., max = 0.5 - 1. / 512.)
        # print('in likelihood')
        # f.write('in likelihood\n')
        # np.savetxt(f,(x_mu.cpu().numpy().squeeze(0).squeeze(0)+0.5)*256-0.5)
        obs_codec = cs.Logistic_UnifBins(x_mu.cpu().numpy(), 
                                         model.dec_log_stdv.cpu().numpy(),
                                         obs_precision, bin_prec=8,
                                         bin_lb=-0.5, bin_ub=0.5)
        return substack(obs_codec, x_view)

    return BBANS(cs.Codec(prior_push, prior_pop), likelihood, posterior)


from utils.torch.modules import np_lossless_downsample, lossless_downsample, lossless_upsample

def SHVC_BitSwap_vANS(config, model):

    z_view = lambda head: head[0]
    x_view = lambda head: head[1] # one channel of downsampled image

    data_prec = config.data_prec
    latent_bin = config.latent_bin
    latent_prec = config.latent_prec

    device = config.device
    
    z_inv_cdf = std_gaussian_centres(latent_bin)

    @torch.no_grad()
    def encoding(state, data):
        data = np_lossless_downsample(data)
        x = torch.from_numpy(data.astype(np.float32)).to(device)
        x = (x - 127.5) / 127.5

        # ************************************************************************************************
        D_params = model.p_x_l_pre_x(x)
        for i in range(x.size(1)-1, int(model.s), -1):
            y = data[:, i,]#np.expand_dims(data[:, i,], axis=1) # (B, 1, H/2, W/2)
            param = D_params[:, i,] # (B, 15, H/2, W/2)
            logit_probs, means, log_scales = torch.split(param, 5, dim=1)
            likelihood = substack(
                cs.LogisticMixture_UnifBins(means.cpu().numpy(), 
                                            log_scales.cpu().numpy(), 
                                            logit_probs.cpu().numpy(), 
                                            data_prec, bin_prec=8, 
                                            bin_lb=-0.5, bin_ub=0.5), 
                x_view)
            # print(f'autoregressive push channel {i}')
            state, = likelihood.push(state, y)
        
        x[:, int(model.s)+1:,] = 0
        z = None
        # ************************************************************************************************
        for zi in range(model.nz):
            mu, scale = model.infer(zi)(given=x if zi == 0 else z)
            posterior = substack(
                # cs.Logistic_UnifBins(mu.cpu().numpy(), 
                #                      torch.log(scale).cpu().numpy(), 
                #                      latent_prec, latent_bin, 
                #                      bin_lb=-np.inf, bin_ub=np.inf),
                cs.DiagGaussian_StdBins(mu.cpu().numpy(), 
                                        scale.cpu().numpy(), 
                                        latent_prec, latent_bin),
                z_view)
            # state, z_next = posterior.pop(state)
            # print(f'posterior pop z{zi}')
            state, z_next_idx = posterior.pop(state)
            z_next = mu + torch.from_numpy(z_inv_cdf[z_next_idx].astype(np.float32)).to(device) * scale

            if zi == 0:
                # ************************************************************************************************
                D_params = model.p_x_l_pre_x_z1((x, z_next)) # (B, 4C, 15, H/2, W/2)
                for i in range(int(model.s), -1, -1):
                    y = data[:, i,]#np.expand_dims(data[:, i,], axis=1) # (B, 1, H/2, W/2)
                    param = D_params[:, i,] # (B, 15, H/2, W/2)
                    logit_probs, means, log_scales = torch.split(param, 5, dim=1)
                    if i == 0:
                        print(means)
                        # print(log_scales)
                    likelihood = substack(
                        cs.LogisticMixture_UnifBins(means.cpu().numpy(), 
                                                    log_scales.cpu().numpy(), 
                                                    logit_probs.cpu().numpy(), 
                                                    data_prec, bin_prec=8, 
                                                    bin_lb=-0.5, bin_ub=0.5), 
                        x_view)
                    # print(f'autoregressive push channel {i}')
                    state, = likelihood.push(state, y)
                # ************************************************************************************************     
            else:
                mu, scale = model.generate(zi)(given=z_next)
                prior = substack(
                    # cs.Logistic_UnifBins(mu.cpu().numpy(), 
                    #                      torch.log(scale).cpu().numpy(), 
                    #                      latent_prec, latent_bin, 
                    #                      bin_lb=-np.inf, bin_ub=np.inf),
                    cs.DiagGaussian_StdBins(mu.cpu().numpy(), 
                                            scale.cpu().numpy(), 
                                            latent_prec, latent_bin), 
                    z_view)
                # print(f'prior push z{zi-1}')
                # state, = prior.push(state, z)
                state, = prior.push(state, z_idx)

            z = z_next
            z_idx = z_next_idx

        # encode prior
        D_params = model.p_z(z) # z3: (B, c, 2, h, w)
        for i in range(z.size(1)-1, -1, -1):
            y = z_idx[:, i,] # z3: (B, 1, h, w)
            print(f'encode ch.{i} y\n{y}')
            param = D_params[:, i,] # (B, 2, h, w)
            print(f'encode ch.{i} param\n{param}')
            mu, logsd = torch.split(param, 1, dim=1) # (B, 1, h, w)
            mu = mu.squeeze(1).cpu().numpy()
            scale = torch.exp(logsd.squeeze(1)).cpu().numpy()
            prior = substack(
                # cs.Logistic_UnifBins(mu.cpu().numpy(), 
                #                      logsd.cpu().numpy(), 
                #                      latent_prec, latent_bin, 
                #                      bin_lb=-np.inf, bin_ub=np.inf),
                cs.DiagGaussian_StdBins(mu, scale, latent_prec, latent_bin), 
                x_view) # z dim (H & W) the same as x in my case
            state, = prior.push(state, y)
            # print(f'autoregressive push z{model.nz-1} channel {i}')

        return state,
    
    @torch.no_grad()
    def decoding(state):
        z_next_idx = np.zeros((config.batch_size, *model.zdim), dtype=int)
        z_next = torch.zeros((config.batch_size, *model.zdim)).to(device)
        x = torch.zeros((config.batch_size, model.xC, *model.xdim)).to(device)

        # autoregressively decode last latent variable
        hidden = None
        lstm_input = torch.zeros((config.batch_size, 1, 1, model.zdim[-2], model.zdim[-1])).to(device) # z3: (B, c=1, 1, h, w)
        for i in range(0, z_next.size(1)):
            param, hidden = model.p_z.ar_model(lstm_input, hidden=hidden) # z3: (B, c=1, 2, h, w)
            print(f'decode ch.{i} param\n{param}')
            mu, logsd = torch.split(param.squeeze(1), 1, dim=1) # (B, 1, h, w)
            mu = mu.squeeze(1) # (B, h, w)
            scale = torch.exp(logsd.squeeze(1))
            prior = substack(
                cs.DiagGaussian_StdBins(mu.cpu().numpy(), 
                                        scale.cpu().numpy(), 
                                        latent_prec, latent_bin), 
                x_view)
            state, z_C_idx = prior.pop(state)
            z_next_idx[:, i,] = z_C_idx
            z_next[:, i,] = mu + torch.from_numpy(z_inv_cdf[z_C_idx].astype(np.float32)).to(device) * scale # (B, h, w)
            lstm_input = z_next[:, i,].unsqueeze(1).unsqueeze(1)
            print(f'decode ch.{i} pop(=next lstm_input)\n{lstm_input}')
        
        for zi in reversed(range(model.nz)):
            if zi == 0:
                hidden = None
                lstm_input = torch.zeros(config.batch_size, 1, 1, model.xdim[-2], model.xdim[-1]).to(device) # (B, c=1, 1, H/2, W/2)
                z1_embd = model.p_x_l_pre_x_z1.z1_cond_network(z_next).unsqueeze(1) # (B, c=1, 4, H/2, W/2)
                for i in range(0, int(model.s)+1):
                    lstm_input = torch.cat([lstm_input, z1_embd], dim=2) # (B, c=1, 5, H/2, W/2)
                    param, hidden = model.p_x_l_pre_x_z1.ar_model(lstm_input, hidden=hidden) # (B, c=1, 15, H/2, W/2)
                    logit_probs, means, log_scales = torch.split(param.squeeze(1), 5, dim=1) # (B, 5, H/2, W/2)
                    if i == 0:
                        print(means)
                        # print(log_scales)
                    likelihood = substack(
                        cs.LogisticMixture_UnifBins(means.cpu().numpy(), 
                                                    log_scales.cpu().numpy(), 
                                                    logit_probs.cpu().numpy(), 
                                                    data_prec, bin_prec=8, 
                                                    bin_lb=-0.5, bin_ub=0.5), 
                        x_view)
                    # print(f'autoregressive push channel {i}')
                    state, x_C = likelihood.pop(state)
                    x[:, i,] = torch.from_numpy(x_C).to(device)
                    lstm_input = x[:, i,].unsqueeze(1).unsqueeze(1)
            else:
                mu, scale = model.generate(zi)(given=z_next)
                prior = substack(
                    cs.DiagGaussian_StdBins(mu.cpu().numpy(), 
                                            scale.cpu().numpy(), 
                                            latent_prec, latent_bin), 
                    z_view)
                state, z_idx = prior.pop(state)
                z = mu + torch.from_numpy(z_inv_cdf[z_idx].astype(np.float32)).to(device) * scale
                
            mu, scale = model.infer(zi)(given=x if zi == 0 else z)
            posterior = substack(
                cs.DiagGaussian_StdBins(mu.cpu().numpy(), 
                                        scale.cpu().numpy(), 
                                        latent_prec, latent_bin),
                z_view)
            state, = posterior.push(state, z_next_idx)

            z_next = z
            z_next_idx = z_idx

        for i in range(int(model.s)+1, model.xC):
            param, hidden = model.p_x_l_pre_x.ar_model(lstm_input, hidden=hidden) # (B, c=1, 15, H/2, W/2)
            logit_probs, means, log_scales = torch.split(param.squeeze(1), 5, dim=1) # (B, 5, H/2, W/2)
            likelihood = substack(
                cs.LogisticMixture_UnifBins(means.cpu().numpy(), 
                                            log_scales.cpu().numpy(), 
                                            logit_probs.cpu().numpy(), 
                                            data_prec, bin_prec=8, 
                                            bin_lb=-0.5, bin_ub=0.5), 
                x_view)
            state, x_C = likelihood.pop(state)
            x[:, i,] = torch.from_numpy(x_C).to(device)
            lstm_input = x[:, i,].unsqueeze(1).unsqueeze(1)

        x = lossless_upsample(x).cpu().numpy().astype(np.uint64)

        return state, x
    
    return cs.Codec(encoding, decoding)

from utils.torch.rand import get_batch_pmfs, mixture_discretized_logistic_cdf, logistic_cdf

class SHVC_BitSwap_ANS():
    def __init__(self, config, model, state, x_bin, z_bin):
        self.model = model
        self.state = state
        self.batch_size = config.batch_size
        self.z_bin_ends, self.z_bin_centres = z_bin
        self.x_bin_ends, self.x_bin_centres = x_bin
        self.batch_x_bin_ends = self.x_bin_ends[0].view(-1, 1, 1, 1).expand(-1, self.batch_size, np.prod(self.model.xdim[1:]), 5) # (255, ) -> (255, B, H * W * C, nr_mix)
        self.xrange = torch.arange(np.prod(model.xdim))
        self.zrange = torch.arange(np.prod(model.zdim))
        self.z_quantbits = config.z_quantbits
        self.x_quantbits = config.x_quantbits
        self.total_xdim = np.prod(self.model.xdim)

    @torch.no_grad()
    def encoding(self, data):
        data = lossless_downsample(data).long()
        x = (data - 127.5) / 127.5

        # ************************************************************************************************
        D_params = self.model.p_x_l_pre_x(x)
        for i in range(x.size(1)-1, int(self.model.s), -1):
            y = torch.flatten(data[:, i,], start_dim=1) # (B, H/2 * W/2)
            param = D_params[:, i,] # (B, 15, H/2, W/2)
            cdfs = mixture_discretized_logistic_cdf(self.batch_x_bin_ends, param, 5) # (B, H/2 * W/2, 255)
            pmfs = get_batch_pmfs(cdfs)
            self.state = batch_ANS(pmfs, self.x_quantbits).encode(self.state, y) # batch execute ANS 
        
        x[:, int(self.model.s)+1:,] = 0
        data[:, int(self.model.s)+1:,] = 0
        z = None
        # ************************************************************************************************

        for zi in range(self.model.nz):
            if zi > 0:
                given = self.z_bin_centres[zi - 1, self.zrange, z]
            else:
                given = self.x_bin_centres[self.xrange, torch.flatten(data, start_dim=1)]
            mu, scale = self.model.infer(zi)(given=given)
            z_bin_ends = self.z_bin_ends[zi].t().unsqueeze(1).expand(-1, self.batch_size, -1)
            cdfs = logistic_cdf(z_bin_ends, mu, scale)
            pmfs = get_batch_pmfs(cdfs)
            self.state, z_next = batch_ANS(pmfs, self.z_quantbits).decode(self.state)

            given = self.z_bin_centres[zi, self.zrange, z_next] # z
            if zi == 0:
                # encode x_i ~ p(x_i|x_1:i-1, z1), i = s, ..., 1
                D_params = self.p_x_l_pre_x_z1((x, given.view(self.batch_size, self.model.zdim))) # (B, 4C, 15, H/2, W/2)
                for i in range(int(self.s), -1, -1):        
                    y = torch.flatten(data[:, i,], start_dim=1) # (B, H/2 * W/2)
                    param = D_params[:, i,] # (B, 15, H/2, W/2)
                    cdfs = mixture_discretized_logistic_cdf(self.batch_x_bin_ends, param, 5) # (B, H/2 * W/2, 255)
                    pmfs = get_batch_pmfs(cdfs)
                    self.state = batch_ANS(pmfs, self.x_quantbits).encode(self.state, y) # batch execute ANS 
            else:
                mu, scale = self.model.generate(zi)(given=given)
                z_bin_ends = self.z_bin_ends[zi - 1].t().unsqueeze(1).expand(-1, self.batch_size, -1)
                cdfs = logistic_cdf(z_bin_ends, mu, scale)
                pmfs = get_batch_pmfs(cdfs)
                self.state = batch_ANS(pmfs, self.z_quantbits).encode(self.state, z)

            z = z_next

        # autoregressively encode prior
        D_params = self.p_z(z) # z3: (B, c, 2, h, w)
        for i in range(z.size(1)-1, -1, -1):
            y = z[:, i,].unsqueeze(1) # z3: (B, 1, h, w)
            param = D_params[:, i,] # (B, 2, h, w)
            mu, logsd = torch.split(param, 1, dim=1) # (B, 1, h, w)
            # ------------------------ logistic ------------------------
            scale = torch.exp(logsd) # 若要修改，discretization.py 也要動
            # scale = 0.1 + 0.9 * modules.softplus(torch.exp(logsd) + np.log(np.exp(1.) - 1.))

        return self.state
    
    def decoding(self):
        z_symtop = self.prior_decode(self.z_bin_ends[-1])
        
        for zi in reversed(range(self.model.nz)):
            given = self.z_bin_centres[zi, self.zrange, z_symtop]
            if zi == 0:
                sym = self.variable_decode('gen', zi, given, self.x_bin_ends, self.x_quantbits)
                given = self.x_bin_centres[self.xrange, sym]
            else:
                sym = self.variable_decode('gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)
                given = self.z_bin_centres[zi - 1, self.zrange, sym]

            self.variable_encode(z_symtop, 'inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            
            z_symtop = sym

        return z_symtop, self.state