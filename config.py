import os
import torch

from utils.discretization import *

class Config_hilloc(object):
    def __init__(self):
        self.seed=0   # seed for dataset generation
        self.log_interval=100
        self.eval_freq = 5

        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'cifar'

        self.epochs = 400
        self.lr=0.002   # Learning rate.
        self.decay=0
        self.batch_size=64   # Batch size on one GPU.

        self.model_name = "hilloc"
        self.bbc_scheme = 'BBC'
        self.discretization_method = 'conditional_prior'

        self.model_dir = f'params/{self.dataset}'
        self.model_param = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"log/{self.dataset}/{self.model_name}"

        self.compression_always_variable=False
        self.compression_exclude_sizes=False
        self.n_flif=5   # number of images to compress with FLIF to start the bb chain (bbans mode)
        self.initial_bits=int(1e8)  # if n_flif==0 then use a random message with this many bits
        self.mode="train"   # Whether to run 'train' or 'eval' model.

        # self.z_size=32   # Size of z variables.
        # self.h_size=160   # Size of resnet block.
        # self.kl_min=0.1   # Number of "free bits/nats".
        # self.depth=1   # Number of downsampling blocks.
        # self.num_blocks=24   # Number of resnet blocks for each downsampling layer.
        # self.k=1   # Number of samples for IS objective.
        # self.image_size=None   # Image size, automatically determined (input is ignored).
        # self.enable_iaf=False   # True for IAF, False for Gaussian posterior
        # self.bidirectional=True   # True for bidirectional, False for bottom-up inference

class Config_bitswap(object):
    def __init__(self):
        self.seed = 99
        self.log_interval = 100  # interval for logging/printing of relevant values
        self.eval_freq = 5

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'imagenet'#'cifar'#'mnist'#'imagenet'

        self.epochs = 600
        self.lr = 2e-4
        self.decay = 0.99#0.999995  # learning rate decay
        self.batch_size = 16

        self.model_name = 'bitswap'
        self.bbc_scheme = 'bitswap'
        self.discretization_method = 'posterior_sampling'

        self.model_dir = f'params/{self.dataset}'
        self.model_param = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        self.log_dir = f"log/{self.dataset}/{self.model_name}"

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
