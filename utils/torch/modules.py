from contextlib import contextmanager

from torch.nn import Module, Parameter, Sequential, Dropout, ELU
from torch.nn import init
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from utils.distributions import DiagonalGaussian

_WN_INIT_STDV = 0.05
_SMALL = 1e-10

_INIT_ENABLED = False

@contextmanager
def init_mode():
    global _INIT_ENABLED
    assert not _INIT_ENABLED
    _INIT_ENABLED = True
    yield
    _INIT_ENABLED = False

# PyTorch module that applies Data Dependent Initialization + Weight Normalization
class WnModule(Module):
    """
    Module with data-dependent initialization
    """

    def __init__(self):
        super().__init__()

    def _init(self, *args, **kwargs):
        """
        Data-dependent initialization. Will be called on the first forward()
        """
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        """
        The standard forward pass
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Calls _init (with no_grad) if not initialized.
        If initialized already, calls _forward.
        """
        if _INIT_ENABLED:
            with torch.no_grad():  # no gradients for the init pass
                return self._init(*args, **kwargs)
        return self._forward(*args, **kwargs)

# Data-Dependent Initialization + Weight Normalization extension of a "Conv2D" module of PyTorch
class WnConv2d(WnModule):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, init_scale=1.0, loggain=True, bias=True, mask=None):
        super().__init__()
        self.in_dim, self.out_dim, self.kernel_size, self.stride, self.padding = in_dim, out_dim, kernel_size, stride, padding
        self.bias = bias
        self.init_scale = init_scale
        self.loggain = loggain
        self.v = Parameter(torch.Tensor(out_dim, in_dim, self.kernel_size, self.kernel_size))
        self.gain = Parameter(torch.Tensor(out_dim))
        self.b = Parameter(torch.Tensor(out_dim), requires_grad=True if self.bias else False)

        init.normal_(self.v, 0., _WN_INIT_STDV)
        if self.loggain:
            init.zeros_(self.gain)
        else:
            init.ones_(self.gain)
        init.zeros_(self.b)

        if mask is not None:
            self.v = mask * self.v

    def _init(self, x):
        # calculate unnormalized activations
        y_bchw = self._forward(x)
        assert len(y_bchw.shape) == 4 and y_bchw.shape[:2] == (x.shape[0], self.out_dim)

        # set g and b so that activations are normalized
        y_c = y_bchw.transpose(0, 1).reshape(self.out_dim, -1)
        m = y_c.mean(dim=1)
        s = self.init_scale / (y_c.std(dim=1) + _SMALL)
        assert m.shape == s.shape == self.gain.shape == self.b.shape

        if self.loggain:
            loggain = torch.clamp(torch.log(s), min=-10., max=None)
            self.gain.data.copy_(loggain)
        else:
            self.gain.data.copy_(s)

        if self.bias:
            self.b.data.sub_(m * s)

        # forward pass again, now normalized
        return self._forward(x)

    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = self.v.view(self.out_dim, -1).norm(p=2, dim=1)
        assert vnorm.shape == self.gain.shape == self.b.shape
        w = self.v * (g / (vnorm + _SMALL)).view(self.out_dim, 1, 1, 1)
        return F.conv2d(x, w, self.b, stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, kernel_size={}, stride={}, padding={}, init_scale={}, loggain={}'.format(self.in_dim, self.out_dim, self.kernel_size, self.stride, self.padding, self.init_scale, self.loggain)


class IAFLayer(nn.Module):
    def __init__(self, hparam, downsample):
        super(IAFLayer, self).__init__()
        
        self.z_size        = hparam.z_size
        self.h_size        = hparam.h_size
        self.enable_iaf    = hparam.enable_iaf
        self.free_bits     = hparam.free_bits
        # posterior is bidirectional - i.e. has a deterministic upper pass but top down sampling.
        self.bidirectional = hparam.bidirectional # True for bidirectional, False for bottom-up inference
        self.ds            = downsample

        if downsample:
            stride, padding, filter_size = 2, 1, 4
            self.down_conv_b = wn(nn.ConvTranspose2d(self.h_size + self.z_size, self.h_size, 4, 2, 1))
        else:
            stride, padding, filter_size = 1, 1, 3
            self.down_conv_b = wn(nn.Conv2d(self.h_size + self.z_size, self.h_size, 3, 1, 1))

        # create modules for UP pass: 
        self.up_conv_a = wn(nn.Conv2d(self.h_size, self.h_size * 2 + self.z_size * 2, filter_size, stride, padding))
        self.up_conv_b = wn(nn.Conv2d(self.h_size, self.h_size, 3, 1, 1))

        # create modules for DOWN pass: 
        self.down_conv_a  = wn(nn.Conv2d(self.h_size, 4 * self.z_size + 2 * self.h_size, 3, 1, 1))

        if self.enable_iaf:
            self.down_ar_conv = ARMultiConv2d([self.h_size] * 2, [self.z_size] * 2, args)


    def up(self, input):
        x = F.elu(input)
        out_conv = self.up_conv_a(x)
        self.qz_mean, self.qz_logsd, self.up_context, h = out_conv.split([self.z_size] * 2 + [self.h_size] * 2, 1)

        h = F.elu(h)
        h = self.up_conv_b(h)

        if self.ds:
            input = F.upsample(input, scale_factor=0.5)

        return input + 0.1 * h
        

    def down(self, input, sample=False):
        x = F.elu(input)
        x = self.down_conv_a(x)
        
        pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = x.split([self.z_size] * 4 + [self.h_size] * 2, 1)
        # prior = D.Normal(pz_mean, torch.exp(pz_logsd))
        prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
            
        if sample:
            # z = prior.rsample()
            z = prior.sample
            kl_cost = kl_obj = torch.zeros(input.size(0)).to(input.device)
        else:
            if not self.bidirectional:
                rz_mean = rz_logsd = 0

            # posterior = D.Normal(rz_mean + self.qz_mean, torch.exp(rz_logsd + self.qz_logsd))
            posterior = DiagonalGaussian(self.qz_mean + rz_mean, 2 * (self.qz_logsd + rz_logsd))
            
            # z = posterior.rsample()
            # logqs = posterior.log_prob(z) 
            # logps = prior.log_prob(z) 
            z = posterior.sample
            logqs = posterior.logps(z)
            logps = prior.logps(z)
            # print(f'train log_q:{logqs.sum(dim=(1,2,3)).mean().item() / (np.log(2.) * 3072)}, log_p:{(-logps).sum(dim=(1,2,3)).mean().item() / (np.log(2.) * 3072)}')

            if self.enable_iaf:
                context = self.up_context + down_context
                x = self.down_ar_conv(z, context) 
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                z = (z - arw_mean) / torch.exp(arw_logsd)
            
                # the density at the new point is the old one + determinant of transformation
                logqs += arw_logsd

            kl_cost = logqs - logps
            # print(f'{pz_mean=} {pz_logsd=}')
            # print(f'{self.qz_mean=} {self.qz_logsd=}')
            # print(f'{rz_mean=} {rz_logsd=}')
            # print(f'log_q:{logqs.sum(dim=(1,2,3)).mean() / (np.log(2.) * 3072)}, log_p:{(-logps).sum(dim=(1,2,3)).mean() / (np.log(2.) * 3072)}')
            # free bits (doing as in the original repo, even if weird)
            kl_obj = kl_cost.sum(dim=(-2, -1)).mean(dim=0, keepdim=True)
            kl_obj = kl_obj.clamp(min=self.free_bits)
            kl_obj = kl_obj.expand(kl_cost.size(0), -1)
            kl_obj = kl_obj.sum(dim=1)

            # sum over all the dimensions, but the batch
            kl_cost = kl_cost.sum(dim=(1,2,3))

        h = torch.cat((z, h_det), 1)
        h = F.elu(h)

        if self.ds:
            input = F.upsample(input, scale_factor=2.)
        
        h = self.down_conv_b(h)

        return input + 0.1 * h, kl_obj, kl_cost

'''
class IAFLayer(nn.Module):
    def __init__(self, xdim, h_size, z_channel):
        super().__init__()
        self.z_channel = z_channel
        self.h_size = h_size # Size of resnet block.

        # posterior is bidirectional - i.e. has a deterministic upper pass but top down sampling.
        self.bidirectional = True # True for bidirectional, False for bottom-up inference
        # self.enable_iaf = False # True for IAF, False for Gaussian posterior
        self.kl_min = torch.tensor(0.1) # Number of "free bits/nats".

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
                                          out_dim = 4 * self.z_channel + 2 * self.h_size,
                                          kernel_size = 3, 
                                          stride = 1, 
                                          padding = 1)
        
        self.down_merge_conv2d = WnConv2d(in_dim = self.h_size + self.z_channel, 
                                          out_dim = self.h_size, 
                                          kernel_size = 3, 
                                          stride = 1, 
                                          padding = 1)


    def up(self, input, **_):
        self.qz_mean, self.qz_logsd, self.up_context, h = self.up_split(input)
        return self.up_merge(h, input)

    def up_split(self, input):
        x = self.act(input)
        x = self.up_split_conv2d(x)
        return torch.split(x, [self.z_channel] * 2 + [self.h_size] * 2, dim=1)

    def up_merge(self, h, input):
        h = nn.ELU()(h)
        h = self.up_merge_conv2d(h)
        return input + 0.1 * h

    def down(self, input):
        h_det, posterior, prior, ar_context = self.down_split(
            input, self.qz_mean, self.qz_logsd, self.up_context)

        z = posterior.sample
        logqs = posterior.logps(z)

        logps = prior.logps(z)

        kl = logqs - logps

        # free bits
        kl_obj = kl.sum(dim=(-2, -1)).mean(dim=0, keepdim=True)
        kl_obj = kl_obj.clamp(min=self.kl_min.to(kl_obj.device))
        kl_obj = kl_obj.expand(kl.size(0), -1)
        kl_obj = kl_obj.sum(dim=1)

        # sum over all the dimensions, but the batch
        kl = kl.sum(dim=(1,2,3))


        return self.down_merge(h_det, input, z), kl, kl_obj 

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
'''


class SimpleNet(nn.Module):
    def __init__(self, in_dim, out_dim, up_down=None):
        super(SimpleNet, self).__init__()
        hidden_size = 128
        # image shape is 3 * 32 * 32
        self.conv1 = wn(nn.Conv2d(in_channels=in_dim, out_channels=hidden_size, 
                                  kernel_size=3, stride=1, padding=1))
        self.conv2 = wn(nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, 
                                  kernel_size=3, stride=1, padding=1))
        self.conv3 = wn(nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, 
                                  kernel_size=3, stride=1, padding=1))
        self.conv4 = wn(nn.Conv2d(in_channels=hidden_size, out_channels=out_dim, 
                                  kernel_size=3, stride=1, padding=1))
        self.down  = wn(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, 
                                  kernel_size=4, stride=2, padding=1))
        self.up = wn(nn.ConvTranspose2d(in_channels=in_dim, out_channels=in_dim, 
                                        kernel_size=4, stride=2, padding=1))
        self.act = nn.PReLU()
        self.up_down = up_down

    def forward(self, x, up_down=None):
        if up_down == "up":
            x = self.act(self.up(x))
        elif up_down == "down":
            x = self.act(self.down(x))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        return x

class Conv1x1Net(nn.Module):
    def __init__(self, in_dim, out_dim, up_down=None):
        super(Conv1x1Net, self).__init__()
        hidden_size = 128
        # image shape is 3 * 32 * 32
        self.conv1 = wn(nn.Conv2d(in_channels=in_dim, out_channels=hidden_size, 
                                  kernel_size=1, stride=1, padding=0))
        self.conv2 = wn(nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, 
                                  kernel_size=1, stride=1, padding=0))
        self.conv3 = wn(nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, 
                                  kernel_size=1, stride=1, padding=0))
        self.conv4 = wn(nn.Conv2d(in_channels=hidden_size, out_channels=out_dim, 
                                  kernel_size=1, stride=1, padding=0))
        self.down  = wn(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, 
                                  kernel_size=4, stride=2, padding=1))
        self.up = wn(nn.ConvTranspose2d(in_channels=in_dim, out_channels=in_dim, 
                                        kernel_size=4, stride=2, padding=1))
        self.act = nn.PReLU()
        self.up_down = up_down
        
    def forward(self, x):
        if self.up_down == "up":
            x = self.act(self.up(x))
        elif self.up_down == "down":
            x = self.act(self.down(x))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        return x

# x with shape (C, H, W) lossless downsampled to (4*C, H/2, W/2)
def lossless_downsample(input, factor=2):
	#assert factor >= 1 and isinstance(factor, int)
	if factor == 1:
		return input
	size = input.size()
	B = size[0]
	C = size[1]
	H = size[2]
	W = size[3]
	assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
	x = input.view(B, C, H // factor, factor, W // factor, factor)
	x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
	x = x.view(B, C * factor * factor, H // factor, W // factor)
	permute = (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
	x = x[:, permute]
	return x

from utils.torch.module_shvc import Conv2dLSTM

class ConvSeqEncoder(nn.Module):
	def __init__(self, input_ch, out_ch, embed_ch, kernel_size=5, dilation=1, num_layers=1, bidirectional=False, dropout=0.0):
		super().__init__()
		self.lstm = Conv2dLSTM(in_channels=embed_ch,  # Corresponds to input size
								   out_channels=embed_ch,  # Corresponds to hidden size
								   kernel_size=kernel_size,  # Int or List[int]
								   num_layers=num_layers,
								   bidirectional=bidirectional,
								   dilation=dilation, stride=1, dropout=0.0,
								   batch_first=True)

		
		self.conv_embed = nn.Conv2d(input_ch, embed_ch, kernel_size, stride=1, padding=(1 if kernel_size==3 else 2))
		self.conv_out1 = nn.Conv2d(embed_ch * (2 if bidirectional else 1), out_ch, 3, stride=1, padding=1 )
		self.embed_ch = embed_ch
		self.out_ch = out_ch
		self.dropout = dropout
		self.conv_dropout = nn.Dropout2d(dropout)

	def td_conv(self,x,conv_fn,out_ch):
		x = x.contiguous()
		batch_size = x.size(0)
		time_steps = x.size(1)
		x = x.view(batch_size*time_steps,x.size(2),x.size(3),x.size(4))
		x = conv_fn(x)
		if self.dropout > 0:
			x = self.conv_dropout(x)	
		x = x.view(batch_size,time_steps,out_ch,x.size(2),x.size(3))
		return x
		
	def forward(self, x, hidden = None):
		x2 = self.td_conv(x,self.conv_embed,self.embed_ch)
		
		outputs, hidden = self.lstm(x2, hidden)
			
		output = self.td_conv(outputs,self.conv_out1,self.out_ch)
		return output, hidden


# taken from https://github.com/jzbontar/pixelcnn-pytorch
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ARMultiConv2d(nn.Module):
    def __init__(self, n_h, n_out, args, nl=F.elu):
        super(ARMultiConv2d, self).__init__()
        self.nl = nl

        convs, out_convs = [], []

        for i, size in enumerate(n_h):
            convs     += [MaskedConv2d('A' if i == 0 else 'B', self.z_size if i == 0 else self.h_size, self.h_size, 3, 1, 1)]
        for i, size in enumerate(n_out):
            out_convs += [MaskedConv2d('B', self.h_size, self.z_size, 3, 1, 1)]

        self.convs = nn.ModuleList(convs)
        self.out_convs = nn.ModuleList(out_convs)


    def forward(self, x, context):
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x)
            if i == 0: 
                x += context
            x = self.nl(x)

        return [conv_layer(x) for conv_layer in self.out_convs]

    
# numerically stable version of the "softplus" function
def softplus(x):
    ret = -F.logsigmoid(-x)
    return ret

# class used to store two sets of parameters
# 1. parameters that are the result of EMA (for evaluation)
# 2. parameters not affected by EMA (for training)
# and to apply EMA to (1.)
class EMA(Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        # decay parameter
        self.mu = mu

        # parameters affected by EMA
        self.shadow = {}

        # "default" parameters
        self.default = {}

    # set parameters affected by EMA
    def register_ema(self, name, val):
        self.shadow[name] = val.clone()

    # set "default parameters
    def register_default(self, name, val):
        self.default[name] = val.clone()

    # return parameters affected by EMA
    def get_ema(self, name):
        assert name in self.shadow
        return self.shadow[name].clone()

    # return "default" parameters
    def get_default(self, name):
        assert name in self.default
        return self.default[name].clone()

    # apply exponential moving average on parameters stored in self.shadow
    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

# PyTorch module that is used to only pass through values
class Pass(Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x

    def inverse(self, x):
        return x

# PyTorch module used to squeeze from [C, H, W] to [C * factor^2, H // factor, W // factor]
class Squeeze2d(Module):
    def __init__(self, factor=2):
        super(Squeeze2d, self).__init__()
        assert factor >= 2
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        shape = x.shape
        height = int(shape[2])
        width = int(shape[3])
        n_channels = int(shape[1])
        assert height % self.factor == 0 and width % self.factor == 0
        x = x.view(-1, n_channels, height//self.factor, self.factor, width//self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(-1, n_channels*self.factor*self.factor, height//self.factor, width // self.factor)
        return x

    def extra_repr(self):
        return 'factor={}'.format(self.factor)

# PyTorch module used to squeeze from [C, H, W] to [C / factor^2, H * factor, W * factor]
class UnSqueeze2d(Module):
    def __init__(self, factor=2):
        super(UnSqueeze2d, self).__init__()
        assert factor >= 2
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        shape = x.shape
        height = int(shape[2])
        width = int(shape[3])
        n_channels = int(shape[1])
        x = x.view(-1, int(n_channels/self.factor**2), self.factor, self.factor, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(-1, int(n_channels/self.factor**2), int(height*self.factor), int(width*self.factor))
        return x

    def extra_repr(self):
        return 'factor={}'.format(self.factor)

# PyTorch module used to build a ResNet layer
class ResNetLayer(Module):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, dropout_p=0., act=ELU()):
        super(ResNetLayer, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.dropout_p = dropout_p
        self.stride = stride
        self.act = act

        self.conv1 = WnConv2d(inchannels, outchannels, kernel_size=kernel_size, stride=1,
                                      padding=padding, init_scale=1.0, loggain=True)
        self.dropout = Dropout(dropout_p)
        self.conv2 = WnConv2d(outchannels,  outchannels, kernel_size=kernel_size,
                                      stride=1, padding=padding, init_scale=0., loggain=False)

    def forward(self, x):
        # first convolution preceded and followed by an activation
        c1 = self.act(self.conv1(self.act(x)))

        # dropout layer
        if self.dropout_p > 0.:
            c1 = self.dropout(c1)

        # second convolution
        c2 = self.conv2(c1)

        # residual connection
        return x + c2

# PyTorch module used to build a sequence of ResNet layers
class ResNetBlock(Sequential):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, nlayers=1, dropout_p=0.,
                 act=ELU()):
        super(ResNetBlock, self).__init__()
        for i in range(nlayers):
            layer = ResNetLayer(inchannels, outchannels, kernel_size, stride, padding, dropout_p, act)
            self.add_module('res{}layer{}'.format(inchannels, i + 1), layer)


def main():
    global _INIT_ENABLED
    print('Outside:', _INIT_ENABLED)
    with init_mode():
        print('Inside:', _INIT_ENABLED)
    print('Outside:', _INIT_ENABLED)
