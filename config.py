import os
import torch

from utils.discretization import *

class Config_bbans(object):
    def __init__(self):
        self.seed=100   # seed for dataset generation
        self.log_interval=100
        self.eval_freq = 5

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'imagenet'#cifar #mnist #imagenet

        self.epochs = 600
        self.lr=2e-3
        self.decay=0.95
        self.batch_size=32

        self.model_name = "bbans"
        self.bbc_scheme = 'BBC'
        self.discretization_method = None

        class Model_hparam:
            def __init__(self, batch_size, dataset):
                self.batch_size=batch_size
                if dataset == "mnist":
                    self.z_size=50
                    self.h_size=200
                else:
                    self.h_size=128
                self.xdim = None

        self.model_hparam = Model_hparam(self.batch_size, self.dataset)

        self.model_dir = f'model/params/{self.dataset}'
        self.model_pt = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"model/log/{self.dataset}/{self.model_name}"

        class Compress_hparam:
            def __init__(self, device):
                self.prior_precision=8
                self.obs_precision=14
                self.q_precision=14
                self.batch_size = 1
                self.device = device

        self.compress_hparam = Compress_hparam(self.device)


class Config_bitswap(object):
    def __init__(self):
        self.seed = 99
        self.log_interval = 100  # interval for logging/printing of relevant values
        self.eval_freq = 5

        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'cifar'#'cifar'#'mnist'#'imagenet'

        self.epochs = 600
        self.lr = 5e-4
        self.decay = 1
        self.batch_size = 64

        self.model_name = 'bitswap'
        self.bbc_scheme = 'bitswap'
        self.discretization_method = 'posterior_sampling'

        class Model_hparam:
            def __init__(self):
                self.nz=8  # number of latent variables
                self.zchannels=8  # number of channels for the latent variables
                self.nprocessing=4  # number of processing layers
                self.resdepth=8  # number of ResNet blocks
                self.reswidth=256  # number of channels in the convolutions in the ResNet blocks
                self.kernel_size=3  # size of the convolutional filter (kernel) in the ResNet blocks
                self.dropout_p=0.3
                self.xdim = None
        
        self.model_hparam = Model_hparam()


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
        self.seed=199   # seed for dataset generation
        self.log_interval=500
        self.eval_freq = 5

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#torch.device('cpu')
        self.dataset = 'mnist'#cifar #mnist #imagenet

        self.epochs = 600
        self.lr=2e-3
        self.decay=0.995
        self.batch_size=64

        self.model_name = "hilloc"
        self.bbc_scheme = 'BBC'
        self.discretization_method = 'conditional_prior'

        class Model_hparam:
            def __init__(self):
                self.n_blocks=4#24
                self.depth=1 # should be 1
                self.z_size=32
                self.h_size=160
                self.enable_iaf=False
                self.free_bits=0.1
                self.bidirectional = True
                self.xdim = None

        self.model_hparam = Model_hparam()

        self.model_dir = f'model/params/{self.dataset}'
        self.model_pt = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"model/log/{self.dataset}/{self.model_name}"

        class Compress_hparam:
            def __init__(self, device):
                self.prior_precision=10
                self.obs_precision=24
                self.q_precision=18
                self.batch_size=1
                self.device = device
                self.compression_exclude_sizes=False
                self.n_flif=0   # number of images to compress with FLIF to start the bb chain (bbans mode)
                self.initial_bits=int(1e5)#1e8  # if n_flif==0 then use a random message with this many bits

        self.compress_hparam = Compress_hparam(self.device)


class Config_shvc(object):
    def __init__(self):
        self.seed=199   # seed for dataset generation
        self.log_interval=500
        self.eval_freq = 5

        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')#torch.device('cpu')
        self.dataset = 'cifar'#cifar #mnist #imagenet

        self.epochs = 1000
        self.lr=5e-4
        self.decay=0.9961
        self.batch_size=1

        self.model_name = "shvc"
        self.bbc_scheme = 'bitswap'
        self.discretization_method = ''

        class Model_hparam:
            def __init__(self):
                # self.n_blocks=8
                # self.h_size=128
                self.xdim = None

        self.model_hparam = Model_hparam()

        self.model_dir = f'model/params/{self.dataset}'
        self.model_pt = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"model/log/{self.dataset}/{self.model_name}"

        class Compress_hparam:
            def __init__(self, device):
                self.prior_precision=10
                self.obs_precision=24
                self.q_precision=18
                self.batch_size=1
                self.device = device
                self.initial_bits=int(1e5)#1e8  # if n_flif==0 then use a random message with this many bits

        self.compress_hparam = Compress_hparam(self.device)