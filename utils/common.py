import glob
import numpy as np
import os
import torch
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from model.model_pool import *


def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# learning rate schedule "bitswap used"
def lr_step(curr_lr, decay=0.99995, min_lr=5e-4):
    # only decay after certain point
    # and decay down until minimal value
    if curr_lr > min_lr:
        curr_lr *= decay
        return curr_lr
    return curr_lr


class ImageNetDataset(Dataset):

    def __init__(self, data_dir, data_shape=(32, 32, 3), transform=None):

        self.data = None

        data_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        for path in data_paths:
            batch = np.load(path)['data']
            self.data = batch if self.data is None else np.concatenate((self.data, batch), axis=0)

        self.data_shape = data_shape
        self.transform = transform


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].reshape(self.data_shape)
        if self.transform:
            x = self.transform(x)
        label = -1
        return (x, label)

class ImageNet64Dataset(Dataset):

    def __init__(self, data_dir, data_shape=(64, 64, 3), transform=None):

        self.data = None

        data_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        for path in data_paths:
            batch = np.load(path)['data']
            self.data = batch if self.data is None else np.concatenate((self.data, batch), axis=0)

        self.data_shape = data_shape
        self.transform = transform


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        real_idx = idx//4
        mod = idx % 4
        x = self.data[real_idx].reshape(self.data_shape)
        x = extract_blocks(x)[mod]
        if self.transform:
            x = self.transform(x)
        label = -1
        return (x, label)
    
def extract_blocks(arr, block_size=(32, 32)):
    _h, _w = block_size
    b, c, h, w = arr.shape
    if h % _h != 0:
        h -= h % _h
        arr = arr[:,:,:h]
    if w % _w != 0:
        w -= w % _w
        arr = arr[:,:,:,:w]
    arr = arr.reshape(b, c, h//_h, _h, -1, _w)
    arr = arr.swapaxes(3,4).reshape(b, c, -1, _h, _w)
    arr = arr.swapaxes(1,2).reshape(b * (h//_h) * (w//_w), c, _h, _w)
    return np.split(arr, arr.shape[0], 0)

def extract_blocks2(input, factor=2):
    B, C, H, W = input.shape
    H_, W_ = H // factor, W // factor
    x = input.reshape(B, C, H_, factor, W_, factor)
    x = np.transpose(x, (0, 3, 5, 1, 2, 4)).copy()
    x = x.reshape(B, factor * factor, C, H_, W_)
    x = x.reshape(B * factor * factor, C, H_, W_)
    return np.split(x, x.shape[0], 0)

# create class that scales up the data to [0,255] if called
class ToInt:
    def __call__(self, pic):
        return pic * 255

def load_data(dataset, model_name, load_train=True):
    # set data pre-processing transforms
    transform = transforms.Compose([transforms.ToTensor(), ToInt()])
    transform_28 = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
    print("load data")
    train_set = None
    if dataset == "cifar":
        if load_train:
            train_set = datasets.CIFAR10(root="./data/cifar",  train=True, transform=transform, download=True)
        test_set  = datasets.CIFAR10(root="./data/cifar", train=False, transform=transform, download=True)
    
    elif dataset == "imagenet" or dataset == "imagenet_full":
        if load_train:
            train_set = ImageNetDataset('./data/imagenet/train_32', transform=transform)
        test_set  = ImageNetDataset('./data/imagenet/val_32', transform=transform)

    elif dataset == "imagenet64":
        test_set  = ImageNetDataset('./data/imagenet/val_64', data_shape=(64, 64, 3), transform=transform)

    elif dataset == "mnist":
        if model_name == 'bitswap' or model_name == 'shvc':
            transform = transform_28
        if load_train:
            train_set = datasets.MNIST(root="./data/mnist", train=True, download=True, 
                                       transform=transform
                                      )
        test_set = datasets.MNIST(root="./data/mnist", train=False, download=True, 
                                  transform=transform
                                 )
    return train_set, test_set


def load_model(model_name, model_pt, hparam, lr, decay):
    print("load model")

    if model_name == 'bbans':
        model = BetaBinomialVAE(hparam)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay)

    if model_name == 'bitswap':
        model = ResNet_VAE(hparam)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = lr_step
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay)
        
    elif model_name == 'hilloc':
        model = Convolutional_VAE(hparam)
        optimizer = optim.Adamax(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay)

    elif model_name == 'shvc':
        model = My_SHVC_VAE(hparam)#SHVC_VAE(hparam) #Simple_SHVC(hparam)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay, last_epoch=-1, verbose=True)

    if os.path.exists(model_pt):
        print('load pre-trained weights')
        param = torch.load(model_pt, map_location=torch.device('cpu'))
        model.load_state_dict(param['model_params'])
        model.best_elbo = param['elbo']
    else:
        print('weights not founnd')

    return model, optimizer, scheduler

from codec.bbc_scheme import VAE, BitSwap_vANS, ResNetVAE, SHVC_BitSwap_ANS
def load_scheme(model_name, cf, model):
    if model_name == 'bbans':
        return VAE(cf, model)
    elif model_name == 'bitswap':
        return BitSwap_vANS(cf, model)
    elif model_name == 'hilloc':
        return ResNetVAE(cf, model)
    elif model_name == 'shvc':
        return SHVC_BitSwap_ANS(cf, model)