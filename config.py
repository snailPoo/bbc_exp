import os
import torch

from utils.discretization import *

class Config_bbans(object):
    def __init__(self):
        self.seed = 100   # seed for dataset generation
        self.log_interval = 100
        self.eval_freq = 2

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'imagenet_full'#'cifar'#'mnist'#'imagenet'#'imagenet_full'

        self.warmup = False
        self.epochs = 150
        self.lr = 1e-3
        self.decay = 0.99
        self.batch_size = 256

        self.model_name = "bbans"
        self.bbc_scheme = 'BBC'
        self.discretization_method = None

        class Model_hparam:
            def __init__(self, batch_size):
                self.batch_size = batch_size
                self.z_size = 50
                self.h_size = 200
                self.xdim = None

        self.model_hparam = Model_hparam(self.batch_size)

        self.model_dir = f'model/params/{self.dataset}'
        self.model_pt = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"model/log/{self.dataset}/{self.model_name}"

        class Compress_hparam:
            def __init__(self, device):
                self.prior_precision = 8
                self.obs_precision = 14
                self.q_precision = 14
                self.batch_size = 1
                self.initial_bits = int(1e5)
                self.device = device

        self.compress_hparam = Compress_hparam(self.device)


class Config_bitswap(object):
    def __init__(self):
        self.seed = 99
        self.log_interval = 3000  # interval for logging/printing of relevant values
        self.eval_freq = 5

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'mnist'#'cifar'#'mnist'#'imagenet'#'imagenet_full'

        self.warmup = False
        self.epochs = 1400
        self.lr = 1e-3
        self.decay = 0.999
        self.batch_size = 256

        self.model_name = 'bitswap'
        self.bbc_scheme = 'bitswap'
        self.discretization_method = 'posterior_sampling'

        class Model_hparam:
            def __init__(self, dataset):
                if dataset == 'mnist':
                    self.nz = 8 # number of latent variables
                    self.zchannels = 1 # number of channels for the latent variables
                    self.reswidth = 61 # number of channels in the convolutions in the ResNet blocks
                    self.dropout_p = 0.2
                elif dataset == 'cifar':
                    self.nz = 8
                    self.zchannels = 8
                    self.reswidth = 256
                    self.dropout_p = 0.3
                else:
                    self.nz = 4
                    self.zchannels = 8
                    self.reswidth = 254
                    self.dropout_p = 0

                self.nprocessing = 4  # number of processing layers
                self.resdepth = 8  # number of ResNet blocks
                self.kernel_size = 3  # size of the convolutional filter (kernel) in the ResNet blocks
                self.xdim = None
        
        self.model_hparam = Model_hparam(self.dataset)

        self.model_dir = f'model/params/{self.dataset}'
        self.model_pt = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"model/log/{self.dataset}/{self.model_name}"

        # used in compression stage
        self.ansbits = 31 # ANS precision
        self.init_state_size = 1000
        self.z_quantbits = 10
        self.x_quantbits = 8
        self.type = torch.float64 # datatype throughout compression
        self.state_dir = f"bitstreams/{self.dataset}"
        self.discretization_dir = f"bins/{self.dataset}"
        self.compression_batch_size = 1

        if self.discretization_method == 'posterior_sampling':
            self.discretization = posterior_sampling


class Config_hilloc(object):
    def __init__(self):
        self.seed = 91   # seed for dataset generation
        self.log_interval = 1000
        self.eval_freq = 5

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'mnist'#'cifar'#'mnist'#'imagenet'#'imagenet_full'

        self.warmup = False
        self.epochs = 10000
        self.lr = 1.5e-5
        self.decay = 1
        self.batch_size = 128

        self.model_name = "hilloc"
        self.bbc_scheme = 'BBC'
        self.discretization_method = 'conditional_prior'

        class Model_hparam:
            def __init__(self, dataset):
                if dataset == 'mnist':
                    self.n_blocks = 2
                    self.z_size = 28
                    self.h_size = 32
                else:
                    self.n_blocks = 24
                    self.z_size = 32
                    self.h_size = 160
                
                self.free_bits = 0
                self.bidirectional = True
                self.xdim = None

        self.model_hparam = Model_hparam(self.dataset)

        self.model_dir = f'model/params/{self.dataset}'
        self.model_pt = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"model/log/{self.dataset}/{self.model_name}"

        class Compress_hparam:
            def __init__(self, device):
                self.prior_precision = 10
                self.obs_precision = 24
                self.q_precision = 18
                self.batch_size = 1
                self.device = device
                self.compression_exclude_sizes = False
                # self.n_flif = 0   # number of images to compress with FLIF to start the bb chain (bbans mode)
                self.initial_bits = int(1e5)#1e8  # if n_flif == 0 then use a random message with this many bits

        self.compress_hparam = Compress_hparam(self.device)


class Config_shvc(object):
    def __init__(self):
        self.seed = 199
        self.log_interval = 500
        self.eval_freq = 1

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'cifar'#'cifar'#'mnist'#'imagenet'#'imagenet_full'

        self.warmup = True
        self.epochs = 1000
        self.lr = 9e-4#5e-4
        self.decay = 0.999#0.9961
        self.batch_size = 128

        self.model_name = "shvc"
        self.bbc_scheme = 'bitswap'
        self.discretization_method = ''

        class Model_hparam:
            def __init__(self):
                self.nz = 3  # number of latent variables
                self.zchannels = 32  # number of channels for the latent variables
                self.nprocessing = 3  # number of processing layers
                self.resdepth = 6#8  # number of ResNet blocks
                self.reswidth = 64#256  # number of channels in the convolutions in the ResNet blocks
                self.kernel_size = 3  # size of the convolutional filter (kernel) in the ResNet blocks
                self.dropout_p = 0.3
                self.lamb = 0.01
                self.xdim = None

        self.model_hparam = Model_hparam()

        self.model_dir = f'model/params/{self.dataset}'
        self.model_pt = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"model/log/{self.dataset}/{self.model_name}"

        class Compress_hparam:
            def __init__(self, device):
                self.data_prec = 24
                self.latent_bin = 10
                # bitseap num_sample_per_bin=30 ≈ latent_prec=latent_bin+5 for 2^5 = 32
                self.latent_prec = 18
                self.batch_size = 1
                self.device = device
                self.initial_bits = int(1e5)  # if n_flif == 0 then use a random message with this many bits

        self.compress_hparam = Compress_hparam(self.device)