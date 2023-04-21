import numpy as np

import torch
from torch import nn
from torchvision import *
import torch.nn.functional as F

## BB-ANS
from torch.distributions import Normal, Beta, Binomial
from torchvision.utils import save_image

from utils.distributions import beta_binomial_log_pdf


# Bit-Swap
import utils.torch.modules as modules
import utils.torch.rand as random


# HiLLoC
from utils.distributions import discretized_logistic


class BetaBinomialVAE(nn.Module):
    def __init__(self, hparam):
        super().__init__()
        self.hidden_dim = hparam.h_size
        self.latent_dim = hparam.z_size
        self.xdim = hparam.xdim
        self.x_flat = int(np.prod(self.xdim))
        self.dataset = hparam.dataset

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        self.register_buffer('n', torch.ones(128, self.x_flat) * 255.)

        self.fc1 = nn.Linear(self.x_flat, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.bn21 = nn.BatchNorm1d(self.latent_dim)
        self.bn22 = nn.BatchNorm1d(self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)

        self.fc4 = nn.Linear(self.hidden_dim, self.x_flat*2)

        self.best_elbo = np.inf
        self.logger = None

    def encode(self, x):
        """Return mu, sigma on latent"""
        h = x / 255.  # otherwise we will have numerical issues
        h = F.relu(self.bn1(self.fc1(h)))
        return self.bn21(self.fc21(h)), torch.exp(self.bn22(self.fc22(h)))

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = F.relu(self.bn3(self.fc3(z)))
        h = self.fc4(h)
        log_alpha, log_beta = torch.split(h, self.x_flat, dim=1)
        return torch.exp(log_alpha), torch.exp(log_beta)

    def loss(self, x, tag):
        z_mu, z_std = self.encode(x.view(-1, self.x_flat))
        z = self.reparameterize(z_mu, z_std)  # sample zs

        x_alpha, x_beta = self.decode(z)
        l = beta_binomial_log_pdf(x.view(-1, self.x_flat), self.n.to(x.device),
                                  x_alpha, x_beta)
        l = torch.sum(l, dim=1)
        p_z = torch.sum(Normal(self.prior_mean, self.prior_std).log_prob(z), dim=1)
        q_z = torch.sum(Normal(z_mu, z_std).log_prob(z), dim=1)
        return -torch.mean(l + p_z - q_z) * np.log2(np.e) / self.x_flat, None

    def sample(self, device, epoch, num=64):
        sample = torch.randn(num, self.latent_dim).to(device)
        x_alpha, x_beta = self.decode(sample)
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(255, p)
        x_sample = binomial.sample()
        x_sample = x_sample.float() / 255.
        save_image(x_sample.view(num, *self.xdim),
                   'results/epoch_{}_samples.png'.format(epoch))

    def reconstruct(self, x, device, epoch):
        x = x.view(-1, self.x_flat).float().to(device)
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)  # sample zs
        x_alpha, x_beta = self.decode(z)
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(255, p)
        x_recon = binomial.sample()
        x_recon = x_recon.float() / 255.
        x_with_recon = torch.cat((x, x_recon))
        save_image(x_with_recon.view(64, *self.xdim),
                   'results/epoch_{}_recon.png'.format(epoch))

from torch import nn

class BetaBinomial_Conv_VAE(nn.Module):
    def __init__(self, hparam):
        super().__init__()
        h_dim = hparam.h_size
        self.xdim = hparam.xdim
        self.x_flat = np.prod(self.xdim)

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        self.n = torch.ones(hparam.batch_size, *self.xdim) * 255.

        self.conv11 = nn.Conv2d(self.xdim[0], h_dim, 4, 2, 1)
        self.conv12 = nn.Conv2d(h_dim, h_dim, 3, 1, 1)
        self.conv13 = nn.Conv2d(h_dim, h_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(h_dim, self.xdim[0] * 2, 3, 1, 1)
        self.conv31 = nn.Conv2d(3, h_dim, 3, 1, 1)
        self.conv32 = nn.Conv2d(h_dim, h_dim, 3, 1, 1)
        self.conv33 = nn.Conv2d(h_dim, h_dim, 3, 1, 1)
        self.conv4 = nn.ConvTranspose2d(h_dim, self.xdim[0] * 2, 4, 2, 1)

        self.best_elbo = np.inf
        self.logger = None

    def encode(self, x):
        """Return mu, sigma on latent"""
        h = x / 255.  # otherwise we will have numerical issues
        h = self.conv13(self.conv12((self.conv11(h))))
        h = self.conv2(h)
        mean, logsd = h.split([self.xdim[0],self.xdim[0]], 1)
        return mean, torch.exp(logsd)

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self.conv33(self.conv32((self.conv31(z))))
        h = self.conv4(h)
        log_alpha, log_beta= h.split([self.xdim[0],self.xdim[0]], 1)
        return torch.exp(log_alpha), torch.exp(log_beta)

    def loss(self, x, tag):
        z_mu, z_std = self.encode(x)
        z = self.reparameterize(z_mu, z_std)  # sample zs

        x_alpha, x_beta = self.decode(z)
        l = beta_binomial_log_pdf(x, self.n.to(x.device),
                                  x_alpha, x_beta)
        l = torch.sum(l, dim=(1,2,3))
        p_z = torch.sum(Normal(self.prior_mean, self.prior_std).log_prob(z), dim=(1,2,3))
        q_z = torch.sum(Normal(z_mu, z_std).log_prob(z), dim=(1,2,3))
        return -torch.mean(l + p_z - q_z) * np.log2(np.e) / self.x_flat, None


class ResNet_VAE(nn.Module):
    def __init__(self, hparam):
        super().__init__()
        # default: disable compressing mode
        # if activated, tensors will be flattened
        self.compressing = False
        self.logger = None
        self.global_step = 0
        
        # hyperparameters
        self.xdim = hparam.xdim  # (3, 32, 32) # data shape
        self.nz = hparam.nz  # number of latent variables
        self.zchannels = hparam.zchannels  # number of channels for the latent variables
        self.nprocessing = hparam.nprocessing  # number of processing layers
        
        # latent height/width is always 16,
        # the number of channels depends on the dataset
        self.zdim = (self.zchannels, 16, 16)
        
        self.resdepth = hparam.resdepth  # number of ResNet blocks
        self.reswidth = hparam.reswidth  # number of channels in the convolutions in the ResNet blocks
        self.kernel_size = hparam.kernel_size  # size of the convolutional filter (kernel) in the ResNet blocks
        dropout_p = hparam.dropout_p

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
                logp = torch.sum(random.logistic_logp(mu, scale, z), dim=2)
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


'''
class Convolutional_VAE(nn.Module):
    def __init__(self, xdim):
        super().__init__()
        self.z_channel = 32
        self.h_size = 160 # Size of resnet block.
        self.num_blocks = 4 # Number of resnet blocks for each downsampling layer.

        self.xdim = xdim
        self.num_pixels = np.prod(self.xdim)

        self.best_elbo = np.inf
        # self.register_buffer("best_elbo", torch.tensor([float('inf')))
        
        self.logger = None
        self.global_step = 0

        # model 
        self.dec_log_stdv = nn.Parameter(torch.Tensor([0.]))
        self.h_top = nn.Parameter(torch.zeros(self.h_size), requires_grad=True)

        self.layers = nn.ModuleList([modules.IAFLayer(self.xdim, self.h_size, self.z_channel) 
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

        kl, obj = 0., 0.

        input = self.preprocess_and_downsample(x)

        for layer in self.layers:
            input = layer.up(input)

        input = self.initialize_input(x.shape).to(x.device)

        for layer in reversed(self.layers):
            input, cur_kl, cur_obj = layer.down(input)
            kl  += cur_kl
            obj += cur_obj

        x_out = self.upsample_and_postprocess(input)

        logp_x = discretized_logistic(x_out, self.dec_log_stdv, sample=x)
        loss = (obj - logp_x).sum() / x.size(0)
        elbo = (kl  - logp_x)
        bpd  = elbo / (np.log(2.) * self.num_pixels)


        self.logger.add_scalar(f'kl/{tag}',              kl.mean(), self.global_step)
        self.logger.add_scalar(f'obj/{tag}',            obj.mean(), self.global_step)
        self.logger.add_scalar(f'bpd/{tag}',            bpd.mean(), self.global_step)
        self.logger.add_scalar(f'elbo/{tag}',          elbo.mean(), self.global_step)
        self.logger.add_scalar(f'log p(x|z)/{tag}', -logp_x.mean(), self.global_step)

        return loss, x_out

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
'''

# git@github.com:pclucas14/iaf-vae.git
class Convolutional_VAE(nn.Module):
    def __init__(self, hparam):
        super(Convolutional_VAE, self).__init__()

        self.h_size = hparam.h_size
        self.depth = hparam.depth
        self.n_blocks = hparam.n_blocks
        self.xdim = hparam.xdim

        self.best_elbo = np.inf
        self.num_pixels = np.prod(self.xdim)

        self.logger = None
        self.global_step = 0

        self.register_parameter('h', nn.Parameter(torch.zeros(self.h_size)))
        self.register_parameter('dec_log_stdv', nn.Parameter(torch.Tensor([0.])))

        layers = []
        # build network
        for i in range(self.depth):
            layer = []

            for j in range(self.n_blocks):
                downsample = (i > 0) and (j == 0)
                layer += [modules.IAFLayer(hparam, downsample)]

            layers += [nn.ModuleList(layer)]

        self.layers = nn.ModuleList(layers) 
        
        self.first_conv = nn.Conv2d(self.xdim[0], self.h_size, 4, 2, 1)
        self.last_conv = nn.ConvTranspose2d(self.h_size, self.xdim[0], 4, 2, 1)

    def loss(self, x, tag):
        # assumes input is \in [-0.5, 0.5] 
        x = torch.clamp((x + 0.5) / 256.0, 0.0, 1.0) - 0.5
        h = self.first_conv(x)

        for layer in self.layers:
            for sub_layer in layer:
                h = sub_layer.up(h)

        kl_cost = kl_obj = 0.0
        self.hid_shape = h[0].size()
        h = self.h.view(1, -1, 1, 1).expand_as(h)

        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, cur_obj, cur_cost = sub_layer.down(h)
                kl_obj += cur_obj
                kl_cost += cur_cost

        h = F.elu(h)
        x_mu = self.last_conv(h)
        x_mu = x_mu.clamp(min = -0.5 + 1. / 512., max = 0.5 - 1. / 512.)

        log_pxlz = discretized_logistic(x_mu, self.dec_log_stdv, sample=x)
        loss = (kl_obj  - log_pxlz).sum() / x.size(0)
        elbo = (kl_cost - log_pxlz)
        bpd  = elbo / (np.log(2.) * self.num_pixels)
        # if tag == "test":
        #     print(f'log_pxlz:{(-log_pxlz).mean() / (np.log(2.) * self.num_pixels)}, bpd:{bpd.mean()}')
        #     return loss, x
        self.logger.add_scalar(f'kl/{tag}',           kl_cost.mean(), self.global_step)
        self.logger.add_scalar(f'obj/{tag}',           kl_obj.mean(), self.global_step)
        self.logger.add_scalar(f'bpd/{tag}',              bpd.mean(), self.global_step)
        self.logger.add_scalar(f'elbo/{tag}',            elbo.mean(), self.global_step)
        self.logger.add_scalar(f'log p(x|z)/{tag}', -log_pxlz.mean(), self.global_step)

        return loss, x


    def sample(self, n_samples=64):
        h = self.h.view(1, -1, 1, 1)
        h = h.expand((n_samples, *self.hid_shape))
        
        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, _, _ = sub_layer.down(h, sample=True)

        x = F.elu(h)
        x = self.last_conv(x)
        
        return x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
    
    
    def cond_sample(self, input):
        # assumes input is \in [-0.5, 0.5] 
        x = self.first_conv(input)
        kl, kl_obj = 0., 0.

        h = self.h.view(1, -1, 1, 1)

        for layer in self.layers:
            for sub_layer in layer:
                x = sub_layer.up(x)

        h = h.expand_as(x)
        self.hid_shape = x[0].size()

        outs = []

        current = 0
        for i, layer in enumerate(reversed(self.layers)):
            for j, sub_layer in enumerate(reversed(layer)):
                h, curr_kl, curr_kl_obj = sub_layer.down(h)
                
                h_copy = h
                again = 0
                # now, sample the rest of the way:
                for layer_ in reversed(self.layers):
                    for sub_layer_ in reversed(layer_):
                        if again > current:
                            h_copy, _, _ = sub_layer_.down(h_copy, sample=True)
                        
                        again += 1
                        
                x = F.elu(h_copy)
                x = self.last_conv(x)
                x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
                outs += [x]

                current += 1

        return outs

import torch.nn.utils.weight_norm as wn

class Simple_SHVC(nn.Module):
	def __init__(self, hparam):
		super(Simple_SHVC, self).__init__()

		self.C, self.H, self.W = hparam.xdim

		# self.z_dim = [(), (32, H >> 2, W >> 2), (24, H >> 3, W >> 3), (16, H >> 4, W >> 4), (8, H >> 5, W >> 5)]
		self.z_dim = [12, 32, 24, 16, 8]

		self.register_parameter('s', nn.Parameter(torch.tensor(6.)))

		self.p_x_given_ARx_z1 = modules.Conv1x1Net(self.z_dim[0]-1 + self.z_dim[1], 3 * 5)

		self.q_z1_given_x  = modules.Conv1x1Net(self.z_dim[0]-1, 2 * self.z_dim[1], "down")
		self.q_z2_given_z1 = modules.Conv1x1Net(self.z_dim[1]  , 2 * self.z_dim[2], "down")
		self.q_z3_given_z2 = modules.Conv1x1Net(self.z_dim[2]  , 2 * self.z_dim[3], "down")

		self.p_z1_given_z2 = modules.Conv1x1Net(self.z_dim[2], 2 * self.z_dim[1], "up")
		self.p_z2_given_z3 = modules.Conv1x1Net(self.z_dim[3], 2 * self.z_dim[2], "up")
		self.p_z3 = modules.Conv1x1Net(self.z_dim[3]-1, 2)

		self.z_up = wn(nn.ConvTranspose2d(in_channels=self.z_dim[1], out_channels=self.z_dim[1], 
									      kernel_size=4, stride=2, padding=1))
		self.lamb = 0.1
		self.nat2bit = np.log2(np.e)
		self.num_pixels = np.prod(hparam.xdim)
		self.logger = None
		self.best_elbo = np.inf
		# self.c_prior = ChannelPriorMultiScale(batch_size,3,32,32,L,mog=False,dp_rate=0,num_layers=3,hidden_size=32)
		

	def loss(self, x, tag):
		batch_size = x.shape[0]
		x = modules.lossless_downsample(x)
		init_save = log_p = log_q = torch.Tensor([0.] * batch_size).to(x.device)
		
		# encode x_i ~ p(x_i|x_1:i-1), i = 12, ..., s+1
		for i in range(1, self.z_dim[0]-int(self.s)+1):
			y = x[:, -i, :, :].unsqueeze(1)
			pad = torch.zeros((batch_size, self.z_dim[1] + (i - 1), self.H >> 1, self.W >> 1)).to(x.device)
			h = torch.cat((x[:, :-i, :, :], pad), dim=1)
			# mu_xi, logsd_xi = torch.split(self.p_x_given_ARx_z1(h), 5, dim=1)
			# p_xi = Logistic_Mixture(mu_xi, logsd_xi)
			# init_save += p_xi.prob(y)
			channel_save = torch.sum(random.discretized_mix_logistic_logp(y, self.p_x_given_ARx_z1(h)), dim=1)
			init_save += channel_save
			log_p += channel_save

		# decode z^1 ~ p(z^1|x_1:s)
		pad = torch.zeros((batch_size, self.z_dim[0]-int(self.s)-1, self.H >> 1, self.W >> 1)).to(x.device)
		h = torch.cat((x[:, :-int(self.s), :, :], pad), dim=1)
		mu_z, logsd_z = torch.split(self.q_z1_given_x(h), self.z_dim[1], dim=1)
		# q_z = Logistic(mu_z, logsd_z)
		# z1 = q_z.sample
		# init_cost = q_z.prob(z1)
		z1 = random.sample_from_logistic(mu_z, logsd_z, mu_z.shape, device=mu_z.device)
		init_cost = torch.sum(random.logistic_logp(mu_z, logsd_z, z1), dim=(1,2))
		log_q += init_cost

		# encode x_i ~ p(x_i|x_1:i-1, z1), i = s, ..., 1
		up_z1 = self.z_up(z1)
		for i in range(int(self.s), 0, -1):
			y = x[:, i, :, :].unsqueeze(1)
			pad = torch.zeros((batch_size, self.z_dim[0] - i - 1, self.H >> 1, self.W >> 1)).to(x.device)
			h = torch.cat((x[:, :i, :, :], pad, up_z1), dim=1)
			# mu_xi, logsd_xi = torch.split(self.p_x_given_ARx_z1(h), 5, dim=1)
			# p_xi = Logistic_Mixture(mu_xi, logsd_xi)
			# log_p += p_xi.prob(y)
			log_p += torch.sum(random.discretized_mix_logistic_logp(y, self.p_x_given_ARx_z1(h)), dim=1)
		
		# decode z^2 ~ q(z^2|z^1)
		mu_z, logsd_z = torch.split(self.q_z2_given_z1(z1), self.z_dim[2], dim=1)
		# q_z = Logistic(mu_z, logsd_z)
		# z2 = q_z.sample
		# log_q += q_z.prob(z2)
		z2 = random.sample_from_logistic(mu_z, logsd_z, mu_z.shape, device=mu_z.device)
		log_q += torch.sum(random.logistic_logp(mu_z, logsd_z, z2), dim=(1,2))

		# encode z^1 ~ p(z^1|z^2)
		mu_z, logsd_z = torch.split(self.p_z1_given_z2(z2), self.z_dim[1], dim=1)
		# p_z = Logistic(mu_z, logsd_z)
		# log_p += p_z.prob(z1)
		log_p += torch.sum(random.logistic_logp(mu_z, logsd_z, z1), dim=(1,2))


		# decode z^3 ~ q(z^3|z^2)
		mu_z, logsd_z = torch.split(self.q_z3_given_z2(z2), self.z_dim[3], dim=1)
		# q_z = Logistic(mu_z, logsd_z)
		# z3 = q_z.sample
		# log_q += q_z.prob(z3)
		z3 = random.sample_from_logistic(mu_z, logsd_z, mu_z.shape, device=mu_z.device)
		log_q += torch.sum(random.logistic_logp(mu_z, logsd_z, z3), dim=(1,2))

		# encode z^2 ~ p(z^2|z^3)
		mu_z, logsd_z = torch.split(self.p_z2_given_z3(z3), self.z_dim[2], dim=1)
		# p_z = Logistic(mu_z, logsd_z)
		# log_p += p_z.prob(z2)
		log_p += torch.sum(random.logistic_logp(mu_z, logsd_z, z2), dim=(1,2))

		# encode z^3
		for i in range(1, self.z_dim[3]):
			y = z3[:, -i, :, :]
			if i > 1:
				pad = torch.zeros((batch_size, i - 1, self.H >> 4, self.W >> 4)).to(x.device)
				h = torch.cat((z3[:, :-i, :, :], pad), dim=1)
			else:
				h = z3[:, :-i, :, :]
			mu_z, logsd_z = torch.split(self.p_z3(h), 1, dim=1)
			# p_z3 = Logistic(mu_z, logsd_z)
			# log_p += p_z3.prob(y)
			log_p += torch.sum(random.logistic_logp(mu_z, logsd_z, y), dim=(1,2))

		loss = torch.mean(log_q - log_p) * self.nat2bit / self.num_pixels + self.lamb * torch.mean(torch.max(torch.tensor(0.), - init_cost + init_save))

		return loss, None
