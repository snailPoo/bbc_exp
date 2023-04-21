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

# should customize encoding/decoding process based on latent variable dependency graph
class BitSwap(bbc_base):
    def __init__(self, config, model, state, x_bin, z_bin):
        # bbc_base.__init__(self, config, model, state, x_bin, z_bin)
        super().__init__(config, model, state, x_bin, z_bin)

    def encoding(self, x):
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

def tensor_to_ndarray(tensor):
    if type(tensor) is tuple:
        return tuple(tensor_to_ndarray(t) for t in tensor)
    else:
        return tensor.detach().cpu().numpy()

def ndarray_to_tensor(arr):
    if type(arr) is tuple:
        return tuple(ndarray_to_tensor(a) for a in arr)
    elif type(arr) is torch.Tensor:
        return arr
    else:
        return torch.from_numpy(np.float32(arr))
    
def torch_fun_to_numpy_fun(fun):
    def numpy_fun(*args, **kwargs):
        torch_args = ndarray_to_tensor(args)
        return tensor_to_ndarray(fun(*torch_args, **kwargs))
    return numpy_fun

def VAE(gen_net, rec_net, obs_codec, prior_prec, latent_prec, device):
    """
    This codec uses the BB-ANS algorithm to code data which is distributed
    according to a variational auto-encoder (VAE) model. It is assumed that the
    VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
    posterior.
    """
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior = substack(Uniform(prior_prec), z_view)

    def likelihood(latent_idxs):
        z = std_gaussian_centres(prior_prec)[latent_idxs]
        x = gen_net(ndarray_to_tensor(z).to(device))
        return substack(obs_codec(x), x_view)

    def posterior(data):
        x = ndarray_to_tensor(data).to(device)
        post_mean, post_stdd = rec_net(x)
        return substack(DiagGaussian_StdBins(
                            tensor_to_ndarray(post_mean), 
                            tensor_to_ndarray(post_stdd), 
                            latent_prec, prior_prec), 
                        z_view)
    return BBANS(prior, likelihood, posterior)

from utils.distributions import DiagonalGaussian, discretized_logistic

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
            # state_before_pop = cs.flatten(state)
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
            # flat_state = cs.flatten(state)
            # init_cost = 32 * (len(state_before_pop) - len(flat_state))
            # print("Initial cost used {} bits.".format(init_cost))
            # print("This is {:.2f} bits per dim.".format(init_cost / (np.prod(input) * model.n_blocks)))
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

def custom_ResNetVAE(config, model):
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

    prior = substack(Uniform(prior_prec), z_view)
    prior_inv_cdf = std_gaussian_centres(prior_prec)

    device = config.device
    
    @torch.no_grad()
    def encoding(state, data):
        # assumes input is in [-0.5, 0.5] 
        x = torch.from_numpy(data.astype(np.float32)).to(device)
        x = torch.clamp((x + 0.5) / 256.0, 0.0, 1.0) - 0.5
        h = model.first_conv(x)
        # run deterministic upper-pass to get qz_mean and qz_std (bottom-up)
        for layer in model.layers:
            for sub_layer in layer:
                h = sub_layer.up(h)

        latents = []
        # input initalization
        input = model.h.view(1, -1, 1, 1).expand((config.batch_size, 
                                                  model.h_size, 
                                                  model.xdim[1] // 2, 
                                                  model.xdim[2] // 2)).to(device)

        for layer in reversed(model.layers):
            for zl in reversed(layer):
                # down pass
                h = zl.down_conv_a(F.elu(input))
                pz_mean, pz_logsd, rz_mean, rz_logsd, _, h_det = h.split([zl.z_size] * 4 + [zl.h_size] * 2, 1)
                # ***************************************** posterior pop z *****************************************
                prior_mean = pz_mean
                prior_stdd = torch.exp(pz_logsd)
                post_mean = zl.qz_mean + rz_mean
                post_stdd = torch.exp(zl.qz_logsd + rz_logsd)
                posterior = substack(cs.DiagGaussian_GaussianBins(post_mean.cpu().numpy(), post_stdd.cpu().numpy(),
                                                                  prior_mean.cpu().numpy(), prior_stdd.cpu().numpy(),
                                                                  latent_prec, prior_prec),
                                     z_view)
                # pop top-down z_L -> z_1
                state, latent_idx = posterior.pop(state)
                latent_val = prior_mean + torch.from_numpy(prior_inv_cdf[latent_idx]).to(device) * prior_stdd
                latents.insert(0, latent_idx)
                # ***************************************************************************************************
                h = torch.cat((latent_val, h_det), 1).float()
                h = zl.down_conv_b(F.elu(h))
                input = input + 0.1 * h
        
        x_mu = model.last_conv(F.elu(input))
        x_mu = x_mu.clamp(min = -0.5 + 1. / 512., max = 0.5 - 1. / 512.)
        # *********************** likelihood push x ***********************
        obs_codec = cs.Logistic_UnifBins(x_mu.cpu().numpy(), 
                                         model.dec_log_stdv.cpu().numpy(),
                                         obs_precision, bin_prec=8,
                                         bin_lb=-0.5, bin_ub=0.5)
        likelihood = substack(obs_codec, x_view)
        state, = likelihood.push(state, data)
        # *****************************************************************
        # *********************** prior push z **********************
        for latent_idx in latents: # push bottom-up z_1 -> z_L
            state, = prior.push(state, latent_idx)
        # ***********************************************************
        return state, 

    @torch.no_grad()
    def decoding(state):
        # run the down pass to top-down get params and latent vals
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
                # ***************************************** prior pop z *****************************************
                prior_mean = pz_mean
                prior_stdd = torch.exp(pz_logsd)
                # pop top-down z_L -> z_1
                state, latent_idx = prior.pop(state)
                latent_val = prior_mean + torch.from_numpy(prior_inv_cdf[latent_idx]).to(device) * prior_stdd
                latents.insert(0, (latent_idx, (prior_mean, prior_stdd)))
                post_params.insert(0, (rz_mean, rz_logsd))
                # ***********************************************************************************************
                h = torch.cat((latent_val, h_det), 1).float()
                h = zl.down_conv_b(F.elu(h))
                input = input + 0.1 * h

        x_mu = model.last_conv(F.elu(input))
        x_mu = x_mu.clamp(min = -0.5 + 1. / 512., max = 0.5 - 1. / 512.)
        # *********************** likelihood pop x ***********************
        obs_codec = cs.Logistic_UnifBins(x_mu.cpu().numpy(), 
                                         model.dec_log_stdv.cpu().numpy(),
                                         obs_precision, bin_prec=8,
                                         bin_lb=-0.5, bin_ub=0.5)
        likelihood = substack(obs_codec, x_view)
        state, data = likelihood.pop(state)
        # ****************************************************************
        x = torch.from_numpy(data.astype(np.float32)).to(device)
        x = torch.clamp((x + 0.5) / 256.0, 0.0, 1.0) - 0.5
        h = model.first_conv(x)
        # run deterministic upper-pass to get qz_mean and qz_std (bottom-up)
        for layer in model.layers:
            for zl, latent, post_param in zip(layer, latents, post_params):
                h = zl.up(h)
                # *************************************** posterior push z ***************************************
                latent_idx, (prior_mean, prior_stdd) = latent
                rz_mean, rz_logsd = post_param
                post_mean = zl.qz_mean + rz_mean
                post_stdd = torch.exp(zl.qz_logsd + rz_logsd)
                posterior = substack(cs.DiagGaussian_GaussianBins(post_mean.cpu().numpy(), post_stdd.cpu().numpy(),
                                                                  prior_mean.cpu().numpy(), prior_stdd.cpu().numpy(),
                                                                  latent_prec, prior_prec),
                                     z_view)
                # push bottom up z_1 -> z_L
                state, = posterior.push(state, latent_idx)
                # ************************************************************************************************
        return state, data

    return cs.Codec(encoding, decoding)