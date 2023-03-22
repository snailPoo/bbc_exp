import numpy as np

import torch
from torch import nn
from torchvision import *
import torch.nn.functional as F

# Bit-Swap
import utils.torch.modules as modules
import utils.torch.rand as random

# HiLLoC
from utils.distributions import DiagonalGaussian, discretized_logistic
from utils.torch.modules import WnConv2d, WnDeConv2d, ArMulticonv2d


class ResNet_VAE(nn.Module):
    def __init__(self, xdim):
        super().__init__()
        # default: disable compressing mode
        # if activated, tensors will be flattened
        self.compressing = False
        self.logger = None
        self.global_step = 0
        
        # hyperparameters
        self.xdim = xdim#(3, 32, 32) # data shape
        self.nz = 8  # number of latent variables
        self.zchannels = 8  # number of channels for the latent variables
        self.nprocessing = 4  # number of processing layers
        # latent height/width is always 16,
        # the number of channels depends on the dataset
        self.zdim = (self.zchannels, 16, 16)
        
        self.resdepth = 8  # number of ResNet blocks
        self.reswidth = 256  # number of channels in the convolutions in the ResNet blocks
        self.kernel_size = 3  # size of the convolutional filter (kernel) in the ResNet blocks
        dropout_p = 0.3

        # apply these two factors (i.e. on the ELBO) in sequence and it results in "bits/dim"
        # factor to convert "nats" to bits
        self.bitsscale = np.log2(np.e)
        # factor to divide by the data dimension
        self.perdimsscale = 1. / np.prod(self.xdim)

        # calculate processing layers convolutions options
        # kernel/filter is 5, so in order to ensure same-size outputs, we have to pad by 2
        padding_proc = (5 - 1) / 2
        assert padding_proc.is_integer()
        padding_proc = int(padding_proc)

        # calculate other convolutions options
        padding = (self.kernel_size - 1) / 2
        assert padding.is_integer()
        padding = int(padding)

        # set-up current "best elbo"
        self.best_elbo = np.inf

        # distribute ResNet blocks over latent layers
        resdepth = [0] * (self.nz)
        i = 0
        for _ in range(self.resdepth):
            i = 0 if i == (self.nz) else i
            resdepth[i] += 1
            i += 1

        # reduce initial variance of distributions corresponding
        # to latent layers if latent nz increases
        scale = 1.0 / (self.nz ** 0.5)

        # activations
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ELU()
        self.actresnet = nn.ELU()

        # Below we build up the main model architecture of the inference- and generative-models
        # All the architecure components are built up from different custom are existing PyTorch modules

        # <===== INFERENCE MODEL =====>
        # the bottom (zi=1) inference model
        self.infer_in = nn.Sequential(
            # shape: [1,32,32] -> [4,16,16]
            modules.Squeeze2d(factor=2),

            # shape: [4,16,16] -> [32,16,16]
            modules.WnConv2d(4 * self.xdim[0],
                             self.reswidth,
                             5,
                             1,
                             padding_proc,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )
        self.infer_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.infer_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        # shape: [32,16,16] -> [1,16,16]
        self.infer_mu = modules.WnConv2d(self.reswidth,
                                         self.zchannels,
                                         self.kernel_size,
                                         1,
                                         padding,
                                         init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # shape: [32,16,16] -> [1,16,16]
        self.infer_std = modules.WnConv2d(self.reswidth,
                                          self.zchannels,
                                          self.kernel_size,
                                          1,
                                          padding,
                                          init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # <===== DEEP INFERENCE MODEL =====>
        # the deeper (zi > 1) inference models
        self.deepinfer_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepinfer_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepinfer_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        self.deepinfer_std = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        # <===== DEEP GENERATIVE MODEL =====>
        # the deeper (zi > 1) generative models
        self.deepgen_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepgen_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepgen_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        self.deepgen_std = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding, init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        # <===== GENERATIVE MODEL =====>
        # the bottom (zi = 1) inference model
        self.gen_in = nn.Sequential(
            # shape: [1,16,16] -> [32,16,16]
            modules.WnConv2d(self.zchannels,
                             self.reswidth,
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )

        self.gen_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        self.gen_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.gen_mu = nn.Sequential(
            # shape: [32,16,16] -> [4,16,16]
            modules.WnConv2d(self.reswidth,
                             4 * self.xdim[0],
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=0.1),
            # shape: [4,16,16] -> [1,32,23]
            modules.UnSqueeze2d(factor=2)
        )

        # the scale parameter of the bottom (zi = 1) generative model is modelled unconditional
        self.gen_std = nn.Parameter(torch.Tensor(*self.xdim))
        nn.init.zeros_(self.gen_std)

    # function to set the model to compression mode
    def compress_mode(self, compress=True):
        self.compressing = compress

    # function that only takes in the layer number and returns a distribution based on that
    def infer(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            h = given

            # if compressing, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()

            # bottom latent layer
            if i == 0:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.xdim)
                # also, when NOT compressing, the input is not scaled from [0,255] to [-1,1]
                else:
                    h = (h - 127.5) / 127.5

                # input convolution
                h = self.infer_in(h)

                # processing ResNet blocks
                h = self.infer_res0(h)

                # other ResNet blocks
                h = self.infer_res1(h)

                # mu parameter of the conditional Logistic distribution
                mu = self.infer_mu(h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.infer_std(h) + 2.)

            # deeper latent layers
            else:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.zdim)

                # input convolution
                h = self.deepinfer_in[i - 1](h)

                # other ResNet blocks
                h = self.deepinfer_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepinfer_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.deepinfer_std[i - 1](h) + 2.)

            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.zdim)).type(type)
                scale = scale.view(np.prod(self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that only takes in the layer number and returns a distribution based on that
    def generate(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            h = given

            # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
            # also, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()
                h = h.view((-1,) + self.zdim)

            # bottom latent layer
            if i == 0:
                # input convolution
                h = self.gen_in(h)

                # processing ResNet blocks
                h = self.gen_res1(h)

                # other ResNet blocks
                h = self.gen_res0(h)

                # mu parameter of the conditional Logistic distribution
                mu = self.gen_mu(h)

                # scale parameter of the conditional Logistic distribution
                # set a minimal value for the scale parameter of the bottom generative model
                scale = ((2. / 255.) / 8.) + modules.softplus(self.gen_std)

            # deeper latent layers
            else:
                # input convolution
                h = self.deepgen_in[i - 1](h)

                # other ResNet blocks
                h = self.deepgen_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepgen_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * modules.softplus(self.deepgen_std[i - 1](h) + np.log(np.exp(1.) - 1.))


            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.xdim if i == 0 else self.zdim)).type(type)
                scale = scale.view(np.prod(self.xdim if i == 0 else self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that takes as input the data and outputs all the components of the ELBO + the latent samples
    def loss(self, x, tag):
        # tensor to store inference model losses
        logenc = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the generative model losses
        logdec = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the latent samples
        zsamples = torch.zeros((self.nz, x.shape[0], np.prod(self.zdim)), device=x.device)

        for i in range(self.nz):
            # inference model
            # get the parameters of inference distribution i given x (if i == 0) or z (otherwise)
            mu, scale = self.infer(i)(given=x if i == 0 else z)

            z_next = random.sample_from_logistic(mu, scale, mu.shape, device=mu.device)

            # store the inference model loss
            zsamples[i] = z_next.flatten(1)
            logq = torch.sum(random.logistic_logp(mu, scale, z_next), dim=2)
            logenc[i] += logq

            # generative model
            # get the parameters of inference distribution i given z
            mu, scale = self.generate(i)(given=z_next)

            # store the generative model loss
            if i == 0:
                # if bottom (zi = 1) generative model, evaluate loss using discretized Logistic distribution
                logp = torch.sum(random.discretized_logistic_logp(mu, scale, x), dim=1)
                logrecon = logp

            else:
                logp = torch.sum(random.logistic_logp(mu, scale, x if i == 0 else z), dim=2)
                logdec[i - 1] += logp

            z = z_next

        # store the prior loss
        logp = torch.sum(random.logistic_logp(torch.zeros(1, device=x.device), torch.ones(1, device=x.device), z), dim=2)
        logdec[self.nz - 1] += logp

        # convert from "nats" to bits
        logenc = torch.mean(logenc, dim=1) * self.bitsscale
        logdec = torch.mean(logdec, dim=1) * self.bitsscale
        logrecon = torch.mean(logrecon) * self.bitsscale

        # construct the ELBO
        if tag == 'train':
            # free bits technique, in order to prevent posterior collapse
            bits_pc = 1.
            kl = torch.sum(torch.max(-logdec + logenc, 
                                     bits_pc * torch.ones((self.nz, self.zdim[0]), device=x.device)))
            elbo = -logrecon + kl
        else:
            elbo = -logrecon + torch.sum(-logdec + logenc)
        
        # scale by image dimensions to get "bits/dim"
        elbo *= self.perdimsscale
        
        # log
        # compute the inference- and generative-model loss
        entdec = (-1) * torch.sum(logdec, dim=1) * self.perdimsscale
        entenc = (-1) * torch.sum(logenc, dim=1) * self.perdimsscale
        entrecon = (-1) * logrecon * self.perdimsscale
        kl = entdec - entenc

        self.logger.add_scalar(f'elbo/{tag}',             elbo,     self.global_step)
        self.logger.add_scalar(f'x/reconstruction/{tag}', entrecon, self.global_step)
        for i in range(0, logdec.shape[0]):
            self.logger.add_scalar(f'z{i+1}/encoder/{tag}', entenc[i], self.global_step)
            self.logger.add_scalar(f'z{i+1}/decoder/{tag}', entdec[i], self.global_step)
            self.logger.add_scalar(f'z{i+1}/KL/{tag}',      kl[i],     self.global_step)

        return elbo, zsamples

    # function to sample from the model (using the generative model)
    def sample(self, device, epoch, num=64):
        # sample "num" latent variables from the prior
        z = random.sample_from_logistic(0, 1, ((num,) + self.zdim), device=device)

        # sample from the generative distribution(s)
        for i in reversed(range(self.nz)):
            mu, scale = self.generate(i)(given=z)
            z_prev = random.sample_from_logistic(mu, scale, mu.shape, device=device)
            z = z_prev

        # scale up from [-1,1] to [0,255]
        x_cont = (z * 127.5) + 127.5

        # ensure that [0,255]
        x = torch.clamp(x_cont, 0, 255)

        # scale from [0,255] to [0,1] and convert to right shape
        x_sample = x.float() / 255.
        x_sample = x_sample.view((num,) + self.xdim)

        # make grid out of "num" samples
        x_grid = utils.make_grid(x_sample)

        # log
        self.logger.add_image('x_sample', x_grid, epoch)

    # function to sample a reconstruction of input data
    def reconstruct(self, x_orig, device, epoch):
        # take only first 32 datapoints of the input
        # otherwise the output image grid may be too big for visualization
        x_orig = x_orig[:32, :, :, :].to(device)

        # sample from the bottom (zi = 1) inference model
        mu, scale = self.infer(0)(given=x_orig)
        z = random.sample_from_logistic(mu, scale, mu.shape, device=device) # sample zs
        
        # sample from the bottom (zi = 1) generative model
        mu, scale = self.generate(0)(given=z)
        x_cont = random.sample_from_logistic(mu, scale, mu.shape, device=device)

        # scale up from [-1.1] to [0,255]
        x_cont = (x_cont * 127.5) + 127.5

        # esnure that [0,255]
        x_sample = torch.clamp(x_cont, 0, 255)

        # scale from [0,255] to [0,1] and convert to right shape
        x_sample = x_sample.float() / 255.
        x_orig = x_orig.float() / 255.

        # concatenate the input data and the sampled reconstructions for comparison
        x_with_recon = torch.cat((x_orig, x_sample))

        # make a grid out of the original data and the reconstruction samples
        x_with_recon = x_with_recon.view((2 * x_orig.shape[0],) + self.xdim)
        x_grid = utils.make_grid(x_with_recon)

        # log
        self.logger.add_image('x_reconstruct', x_grid, epoch)


class IAFLayer(torch.nn.Module):
    def __init__(self, mode, xdim):
        super().__init__()
        self.z_channel = 32
        self.h_size = 160 # Size of resnet block.
        self.kl_min = torch.tensor(0.1) # Number of "free bits/nats".

        self.bidirectional = True # True for bidirectional, False for bottom-up inference
        self.enable_iaf = False # True for IAF, False for Gaussian posterior

        self.mode = mode

        self.act = nn.ELU()

        self.in_dim = xdim[0]
        self.up_split_conv2d   = WnConv2d(in_dim = self.h_size, 
                                          out_dim = 2 * self.z_channel + 2 * self.h_size, 
                                          kernel_size = 3, 
                                          stride = 1, 
                                          padding = 1)
        
        self.up_merge_conv2d   = WnConv2d(in_dim = self.h_size, 
                                          out_dim = self.h_size, 
                                          kernel_size = 3, 
                                          stride = 1, 
                                          padding = 1)
        
        self.down_split_conv2d = WnConv2d(in_dim = self.h_size, 
                                          out_dim = 4 * self.z_channel + self.h_size * 2,
                                          kernel_size = 3, 
                                          stride = 1, 
                                          padding = 1)
        
        self.down_merge_conv2d = WnConv2d(in_dim = self.h_size + self.z_channel, 
                                          out_dim = self.h_size, 
                                          kernel_size = 3, 
                                          stride = 1, 
                                          padding = 1)
        
        if self.enable_iaf:
            self.ar_multiconv2d = ArMulticonv2d(self.z_channel, 
                                              [(self.h_size), (self.h_size)],
                                               [self.z_channel, self.z_channel])

    def up(self, input, **_):
        self.qz_mean, self.qz_logsd, self.up_context, h = self.up_split(input)
        return self.up_merge(h, input)

    def up_split(self, input):
        x = self.act(input)
        x = self.up_split_conv2d(x)
        return torch.split(x, [self.z_channel, self.z_channel, self.h_size, self.h_size], dim=1)

    def up_merge(self, h, input):
        h = nn.ELU()(h)
        h = self.up_merge_conv2d(h)
        return input + 0.1 * h

    def down(self, input):
        h_det, posterior, prior, ar_context = self.down_split(
            input, self.qz_mean, self.qz_logsd, self.up_context)

        if self.mode in ["init", "sample"]:
            z = prior.sample
        else:
            z = posterior.sample
        
        batch_size = z.shape[0]
        if self.mode == "sample":
            kl_cost = kl_obj = torch.zeros([batch_size])
        else:
            logqs = posterior.logps(z)
            if self.enable_iaf:
                x = self.ar_multiconv2d(z, ar_context)
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                z = (z - arw_mean) / torch.exp(arw_logsd)
                logqs += arw_logsd

            logps = prior.logps(z)

            kl_cost = logqs - logps

            if self.kl_min > 0:
                # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
                kl_ave = torch.mean(torch.sum(kl_cost, [2, 3]), [0], keepdim=True)
                kl_ave = torch.max(kl_ave, self.kl_min)
                kl_ave = kl_ave.repeat(batch_size, 1)
                kl_obj = torch.sum(kl_ave, [1])
            else:
                kl_obj = torch.sum(kl_cost, [1, 2, 3])

            kl_cost = torch.sum(kl_cost, [1, 2, 3])

        return self.down_merge(h_det, input, z), kl_obj, kl_cost

    def down_split(self, input, qz_mean, qz_logsd, up_context):
        x = self.act(input)
        x = self.down_split_conv2d(x)

        pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = torch.split(x, [self.z_channel] * 4 + [self.h_size] * 2, dim=1)
        
        prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
        posterior = DiagonalGaussian(
            qz_mean + (rz_mean if self.bidirectional else 0),
            2 * (qz_logsd + (rz_logsd if self.bidirectional else 0)))
        
        return h_det, posterior, prior, up_context + down_context

    def down_merge(self, h_det, input, z):
        h = torch.cat([z, h_det], dim=1)
        h = self.act(h)
        h = self.down_merge_conv2d(h)
        return input + 0.1 * h


class Convolutional_VAE(nn.Module):
    def __init__(self, xdim):
        super().__init__()
        self.h_size = 160 # Size of resnet block.
        self.num_blocks = 24 # Number of resnet blocks for each downsampling layer.
        self.xdim = xdim

        self.mode = None
        self.best_elbo = np.inf
        
        self.logger = None
        self.global_step = 0

        # model 
        self.dec_log_stdv = nn.Parameter(torch.zeros(1))
        self.h_top = nn.Parameter(torch.zeros(self.h_size), requires_grad=True)

        self.layers = nn.ModuleList([IAFLayer("init", self.xdim) 
                                     for _ in range(self.num_blocks)])
        
        self.downsample_conv2d = WnConv2d(in_dim = self.xdim[0], 
                                          out_dim = self.h_size, 
                                          kernel_size = 5, 
                                          stride = 2, 
                                          padding = 2)
        
        self.upsample_deconv2d = WnDeConv2d(in_dim = self.h_size, 
                                            out_dim = self.xdim[0], 
                                            kernel_size = 4, 
                                            stride = 2, 
                                            padding = 1)

    def loss(self, x, tag="train"):

        kl_cost = kl_obj = 0.0

        input = self.preprocess_and_downsample(x)

        for layer in self.layers:
            input = layer.up(input)

        input = self.initialize_input(x.shape).to(x.device)

        for layer in reversed(self.layers):
            input, cur_obj, cur_cost = layer.down(input)
            kl_obj += cur_obj
            kl_cost += cur_cost

        x_out = self.upsample_and_postprocess(input)

        log_pxz = discretized_logistic(x_out, self.dec_log_stdv, sample=x)
        
        obj = torch.sum(kl_obj - log_pxz)
        loss = torch.sum(kl_cost - log_pxz)

        bits_per_dim = loss / (np.log(2.) * np.prod(x.shape))
        
        self.logger.add_scalar(f'log_pxz/{tag}', (-log_pxz).mean(), self.global_step)
        self.logger.add_scalar(f'kl_cost/{tag}',    kl_cost.mean(), self.global_step)
        self.logger.add_scalar( f'kl_obj/{tag}',     kl_obj.mean(), self.global_step)
        self.logger.add_scalar(    f'bpd/{tag}',      bits_per_dim, self.global_step)

        return obj, x_out

    def initialize_input(self, size):
        return torch.tile(self.h_top.view(1, -1, 1, 1),
                         [size[0], 1, size[2] // 2, size[3] // 2])

    def upsample_and_postprocess(self, input):
        x = nn.ELU()(input)
        x = self.upsample_deconv2d(x)
        x = torch.clamp(x, -0.5 + 1 / 512., 0.5 - 1 / 512.)
        return x

    def preprocess_and_downsample(self, x):
        x = x.float()
        x = torch.clamp((x + 0.5) / 256.0, 0.0, 1.0) - 0.5
        h = self.downsample_conv2d(x)  # -> [16, 16]
        return h