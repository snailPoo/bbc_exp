# export PYTHONPATH=$PYTHONPATH:/tmp2/raaa/Compression/bbc_exp
import numpy as np
import os
import time

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import *

from tensorboardX import SummaryWriter

from contextlib import contextmanager
from utils.common import same_seed, load_model, load_data
from config import *

cf = Config_shvc()# Config_bbans # Config_bitswap # Config_hilloc # Config_shvc

_INIT_ENABLED = False
@contextmanager
def init_mode():
    global _INIT_ENABLED
    assert not _INIT_ENABLED
    _INIT_ENABLED = True
    yield
    _INIT_ENABLED = False

def warmup(model, data_loader, warmup_batches):
    # convert model to evaluation mode (no Dropout etc.)
    model.eval()

    # prepare initialization batch
    for batch_idx, (x, _) in enumerate(data_loader):
        # # stack image with to current stack
        # warmup_images = torch.cat((warmup_images, x), dim=0) \
        #     if batch_idx != 0 else x
        # do one 'special' forward pass to initialize parameters
        with init_mode():
            elbo, _ = model.loss(x.to(cf.device), 'warmup')
        # stop stacking batches if reaching limit
        if batch_idx + 1 == warmup_batches:
            break

    # set the stack to current device
    # warmup_images = warmup_images.to(cf.device)

    # # do one 'special' forward pass to initialize parameters
    # with init_mode():
    #     elbo, _ = model.loss(warmup_images, 'warmup')

    print(f'====> Epoch: 0 Average loss: {elbo:.4f}')


def train(epoch, data_loader, model, optimizer):
    # convert model to train mode (activate Dropout etc.)
    model.train()

    # get number of batches
    num_batch = len(data_loader)

    # setup training metrics
    elbos = 0

    start_time = time.time()

    # enumerate over the batches
    for batch_idx, (x, _) in enumerate(data_loader):
        # keep track of the global step
        model.global_step = (epoch - 1) * num_batch + (batch_idx + 1)
            
        x = x.to(cf.device)

        # empty all the gradients stored
        optimizer.zero_grad()

        # evaluate the data under the model and calculate ELBO components
        elbo, _ = model.loss(x, 'train')

        # calculate gradients
        elbo.backward()

        # take gradient step
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
        optimizer.step()

        # log gradient norm
        model.logger.add_scalar('gradient norm', total_norm, model.global_step)

        elbos += elbo

        # log and save parameters
        if batch_idx % cf.log_interval == 0 and cf.log_interval < num_batch:

            # print metrics to console
            print(f'Train Epoch: {epoch} [{batch_idx}/{num_batch} ({100. * batch_idx / num_batch:.0f}%)]\t Loss: {elbo.item():.6f}\t Gradient norm: {total_norm:.2f}\t Steps/sec: {(time.time() - start_time) / (batch_idx + 1):.3f}')

            # log
            model.logger.add_scalar('step-sec', (time.time() - start_time) / (batch_idx + 1), model.global_step)
            model.logger.add_scalar('lr', optimizer.param_groups[0]['lr'], model.global_step)


    # print the average loss of the epoch to the console
    elbo = (elbos / num_batch).item()
    print(f'====> Epoch: {epoch} Average loss: {elbo:.4f}')


def eval(epoch, data_loader, model):
    # convert model to evaluation mode (no Dropout etc.)
    model.eval()

    # setup the reconstruction dataset
    num_batch = len(data_loader)
    recon_batch_idx = int(torch.Tensor(1).random_(0, num_batch - 1))

    elbos = 0

    # enumerate over the batches
    for batch_idx, (x, _) in enumerate(data_loader):
        x = x.to(cf.device)
        # save batch for reconstruction
        if batch_idx == recon_batch_idx:
            recon_dataset = x

        with torch.no_grad():
            # evaluate the data under the model and calculate ELBO components
            elbo, _ = model.loss(x, 'eval')

        elbos += elbo

    elbo = (elbos / num_batch).item()

    # print metrics to console and Tensorboard
    print(f'\nEpoch: {epoch}\tTest loss: {elbo:.6f}')
    # model.logger.add_scalar('elbo/test', elbo, epoch)

    # if the current ELBO is better than the ELBO's before, save parameters
    if elbo < model.best_elbo and not np.isnan(elbo):
        print("best result\n")
        model.logger.add_scalar('elbo/besttest', elbo, epoch)
        if not os.path.exists(cf.model_dir):
            os.makedirs(cf.model_dir)
        to_saved = {
            'model_params' : model.state_dict(),
            'elbo' : elbo,
        }
        torch.save(to_saved, cf.model_pt)
        model.best_elbo = elbo

        if cf.model_name == 'bitswap':
            model.sample(cf.device, epoch)
            model.reconstruct(recon_dataset, cf.device, epoch)

def count_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of trainable parameters in model: {num_parameters}')
    model.logger.add_text(f'hyperparams', '{num_parameters}', 0)


if __name__ == '__main__':
    print(f"Model:{cf.model_name}; Dataset:{cf.dataset}")

    # Set seed for reproducibility
    same_seed(cf.seed)

    # set up dataloader
    train_set, test_set = load_data(cf.dataset, cf.model_name)
    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(
        dataset=train_set, 
        sampler=None, batch_size=cf.batch_size, 
        shuffle=True, drop_last=True, **kwargs)
    test_loader = DataLoader(
        dataset=test_set, 
        sampler=None, batch_size=cf.batch_size, 
        shuffle=False, drop_last=True, **kwargs)
    cf.model_hparam.xdim = train_set[0][0].shape

    # set up model and optimizer
    model, optimizer, scheduler = load_model(cf.model_name, cf.model_pt, 
                                             cf.model_hparam, cf.lr, cf.decay)
    # create loggers
    model.logger = SummaryWriter(log_dir=cf.log_dir)

    model.to(cf.device)
    count_num_params(model)

    # data-dependent initialization
    # warmup(model, train_loader, 25)

    # initial test loss
    eval(0, test_loader, model)

    print("Training")
    for epoch in range(1, cf.epochs + 1):
        train(epoch, train_loader, model, optimizer)
        if scheduler is not None:
            scheduler.step()
        if epoch % cf.eval_freq == 0:
            eval(epoch, test_loader, model)
