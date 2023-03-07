from torch import optim
from torchvision import transforms, datasets

from model.model_pool import *
from utils.discretization import *
import os


# create class that scales up the data to [0,255] if called
class ToInt:
    def __call__(self, pic):
        return pic * 255
# set data pre-processing transforms
transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
# learning rate schedule
def lr_step(curr_lr, decay=0.99995, min_lr=5e-4):
    # only decay after certain point
    # and decay down until minimal value
    if curr_lr > min_lr:
        curr_lr *= decay
        return curr_lr
    return curr_lr

class Config(object):
    def __init__(self):
        self.seed = 99
        self.log_interval = 100  # interval for logging/printing of relevant values

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'cifar'

        self.epochs = 10000000000
        self.lr = 2e-3
        self.decay = 0.999995  # decay of the learning rate when using learning rate schedule
        self.batch_size = 16

        self.model_name = 'bitswap'
        self.bbc_scheme = 'bitswap'
        self.discretization_method = 'posterior_sampling'

        self.model_param = f"model/params/{self.dataset}/{self.model_name}_best.pt"
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


    def load_data(self):
        print("load data")
        if self.dataset == "cifar":
            self.train_set = datasets.CIFAR10(root="data/cifar", train=True, transform=transform_ops, download=True)
            self.test_set = datasets.CIFAR10(root="data/cifar", train=False, transform=transform_ops, download=True)
        
        elif self.dataset == "imagenet" or self.dataset == "imagenetcrop":
            self.train_set = modules.ImageNet(root='data/imagenet/train', file='train.npy', transform=transform_ops)
            self.test_set = modules.ImageNet(root='data/imagenet/test', file='test.npy', transform=transform_ops)
        
        else:
            self.train_set = datasets.MNIST(root="model/data/mnist", train=True, download=True, 
                                            transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
                                           )
    def load_model(self):
        print("load model")
        if self.model_name == 'bitswap':
            self.model = ResNet_VAE()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = lr_step
        
        self.model.to(self.device)

        if os.path.exists(self.model_param):
            print('load pre-trained weights')
            param = torch.load(self.model_param)
            self.model.load_state_dict(param['model_params'])
            self.model.best_elbo = param['elbo']
        else:
            print('weights not founnd')