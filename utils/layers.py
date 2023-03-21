import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch.modules import WnConv2d


# class Conv2d(nn.Module):
#     def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, 
#                  init_scale=0.1, mask=None, dtype=torch.float32):
        
#         super(Conv2d, self).__init__()
#         self.in_dim, self.out_dim, self.kernel_size, self.stride, self.padding = in_dim, out_dim, kernel_size, stride, padding
#         self.init_scale, self.mask = init_scale, mask
#         self.kernel_shape = (out_dim, in_dim, self.kernel_size, self.kernel_size)

#         self.v = nn.Parameter(torch.randn(self.kernel_shape))
#         self.g = nn.Parameter(torch.ones(self.out_dim))
#         self.b = nn.Parameter(torch.zeros(self.out_dim))
#         if self.mask is not None:
#             self.v = self.mask * self.v

#     def init(self, x):
#         self.v = nn.Parameter(torch.randn(self.kernel_shape) * 0.05)
#         if self.mask is not None:
#             v = self.mask * v
#         v_norm = F.normalize(self.v, dim=(1, 2, 3))
#         x_init = F.conv2d(x, v_norm, stride=self.stride, padding=self.padding)
#         m_init, v_init = torch.mean(x_init, dim=(0, 2, 3)), torch.var(x_init, dim=(0, 2, 3))
#         scale_init = self.init_scale / torch.sqrt(v_init + 1e-10)
#         self.g = nn.Parameter(torch.log(scale_init) / 3.0)
#         self.b = nn.Parameter(-m_init * scale_init)
#         return torch.reshape(torch.exp(self.g), (1, self.out_dim, 1, 1)) * (x_init - torch.reshape(m_init, (1, self.out_dim, 1, 1)))

#     def forward(self, x):
#         w = torch.reshape(torch.exp(self.g), (self.out_dim, 1, 1, 1)) * F.normalize(self.v, dim=(1, 2, 3))
#         return F.conv2d(x, w, self.b, self.stride, self.padding)


# def conv2d(x, num_kernel, padding=1, kernel_size=(3, 3), stride=(1, 1), 
#            init_scale=0.1, init=False, mask=None):

#     kernel_shape = (num_kernel, int(x.shape[1]), *kernel_size)
#     if init:
#         v = Parameter(torch.randn(kernel_shape) * 0.05)
#         if mask is not None:
#             v = mask * v
#         v_norm = F.normalize(v, dim=(1, 2, 3))
#         x_init = F.conv2d(x, v_norm, stride=stride, padding=padding)
#         m_init, v_init = torch.mean(x_init, dim=(0, 2, 3)), torch.var(x_init, dim=(0, 2, 3))
#         scale_init = init_scale / torch.sqrt(v_init + 1e-10)
#         g = Parameter(torch.log(scale_init) / 3.0)
#         b = Parameter(-m_init * scale_init, requires_grad=True)
#         return torch.reshape(torch.exp(g), (1, num_kernel, 1, 1)) * (x_init - torch.reshape(m_init, (1, num_kernel, 1, 1)))
#     else:
#         v = Parameter(torch.randn(kernel_shape, device=x.device))
#         g = Parameter(torch.ones(num_kernel, device=x.device))
#         b = Parameter(torch.zeros(num_kernel, device=x.device))
#         if mask is not None:
#             v = mask * v
#         # use weight normalization (Salimans & Kingma, 2016)
#         w = torch.reshape(torch.exp(g), (num_kernel, 1, 1, 1)) * F.normalize(v, dim=(1, 2, 3))
#         return F.conv2d(x, w, b, stride, padding)


# def deconv2d(x, num_kernel, padding=1, kernel_size=(3, 3), stride=(2, 2), 
#             init_scale=0.1, init=False, mask=None):
    
#     kernel_shape = (int(x.shape[1]), num_kernel, stride[0]+2, stride[1]+2)
#     if init:
#         v = Parameter(torch.randn(kernel_shape) * 0.05)
#         if mask is not None:
#             v = mask * v
#         v_norm = F.normalize(v, dim=(1, 2, 3))

#         x_init = F.conv_transpose2d(x, v_norm, stride=stride, padding=padding)
#         m_init, v_init = torch.mean(x_init, dim=(0, 2, 3)), torch.var(x_init, dim=(0, 2, 3))
#         scale_init = init_scale / torch.sqrt(v_init + 1e-10)
#         g = Parameter(torch.log(scale_init) / 3.0)
#         b = Parameter(-m_init * scale_init)
#         return torch.reshape(torch.exp(g), (1, num_kernel, 1, 1)) * (x_init - torch.reshape(m_init, (1, num_kernel, 1, 1)))
#     else:
#         v = Parameter(torch.randn(kernel_shape))
#         g = Parameter(torch.randn(num_kernel))
#         b = Parameter(torch.randn(num_kernel))
#         if mask is not None:
#             v = mask * v
#         # use weight normalization (Salimans & Kingma, 2016)
#         w = torch.reshape(torch.exp(g), (1, num_kernel, 1, 1)) * F.normalize(v, dim=(1, 2, 3))
#         return F.conv_transpose2d(x, w, b, stride, padding)


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[l, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return torch.tensor(mask).permute((3,2,0,1))


class ArMulticonv2d(nn.Module):
    def __init__(self, n_in, n_h, n_out):
        super(ArMulticonv2d, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()

        kernel_size = 3
        stride = 1
        padding = 1

        zerodiagonal = False
        for i, size in enumerate(n_h):
            mask = get_conv_ar_mask(kernel_size, kernel_size, 
                                    n_in, size, zerodiagonal)
            conv_layer = WnConv2d(in_dim=n_in if i == 0 else n_h[i-1], 
                                  out_dim=size, kernel_size=kernel_size, 
                                  stride=stride, padding=padding, mask=mask)
            self.conv_layers.append(conv_layer)

        zerodiagonal = True
        mask = torch.tensor(get_conv_ar_mask(kernel_size, kernel_size, 
                                                n_h[-1], size, zerodiagonal))
        for i, size in enumerate(n_out):
            out_layer = WnConv2d(in_dim=n_h[-1], out_dim=size, 
                                 kernel_size=kernel_size, stride=stride, 
                                 padding=padding, mask=mask)
            self.out_layers.append(out_layer)
        
        self.nl = nn.ELU()

    def forward(self, x, context):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i == 0:
                x += context
            x = self.nl(x)
        out = []
        for i, out_layer in enumerate(self.out_layers):
            out.append(out_layer(x))
        return out