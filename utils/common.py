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


# create class that scales up the data to [0,255] if called
class ToInt:
    def __call__(self, pic):
        return pic * 255
# set data pre-processing transforms
transform_ori = transforms.Compose([transforms.ToTensor(), ToInt()])
transform_ori_28 = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
transform_nor = transforms.Compose([transforms.ToTensor(), lambda x : x - 0.5])
transform_nor_28 = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), lambda x : x - 0.5])
def load_data(dataset):
    print("load data")

    if dataset == "cifar":
        train_set = datasets.CIFAR10(root="../data/cifar",  train=True, transform=transform_nor, download=True)
        test_set  = datasets.CIFAR10(root="../data/cifar", train=False, transform=transform_nor, download=True)
    
    elif dataset == "imagenet" or dataset == "imagenetcrop":
        train_set = ImageNetDataset('../data/imagenet/train_32', transform=transform_ori)
        test_set  = ImageNetDataset('../data/imagenet/val_32'  , transform=transform_ori)
    
    elif dataset == "mnist":
        train_set = datasets.MNIST(root="../data/mnist", train=True, download=True, 
                                   transform=transform_nor_28
                                  )
        test_set = datasets.MNIST(root="../data/mnist", train=False, download=True, 
                                   transform=transform_nor_28
                                  )
    return train_set, test_set


def load_model(model_name, model_pt, hparam, lr, decay):
    print("load model")

    if model_name == 'bitswap':
        model = ResNet_VAE(hparam)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = lr_step
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay)
        
    elif model_name == 'hilloc':
        model = Convolutional_VAE(hparam)
        optimizer = optim.Adamax(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay)

    if os.path.exists(model_pt):
        print('load pre-trained weights')
        param = torch.load(model_pt, map_location=torch.device('cpu'))
        model.load_state_dict(param['model_params'])
        model.best_elbo = param['elbo']
    else:
        print('weights not founnd')

    return model, optimizer, scheduler
