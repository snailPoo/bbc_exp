# export PYTHONPATH=$PYTHONPATH:/tmp2/raaa/Compression/bbc_exp
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import *
import os
import time
import numpy as np

from tensorboardX import SummaryWriter

import utils.torch.modules as modules
from config import Config

config = Config()

# setup seeds to maintain experiment reproducibility
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)
torch.backends.cudnn.deterministic = True

def warmup(data_loader, warmup_batches):
    # convert model to evaluation mode (no Dropout etc.)
    config.model.eval()

    # prepare initialization batch
    for batch_idx, (image, _) in enumerate(data_loader):
        # stack image with to current stack
        warmup_images = torch.cat((warmup_images, image), dim=0) \
            if batch_idx != 0 else image

        # stop stacking batches if reaching limit
        if batch_idx + 1 == warmup_batches:
            break

    # set the stack to current device
    warmup_images = warmup_images.to(config.device)

    # do one 'special' forward pass to initialize parameters
    with modules.init_mode():
        logrecon, logdec, logenc, _ = config.model.loss(warmup_images)

    # log
    logdec = torch.sum(logdec, dim=1)
    logenc = torch.sum(logenc, dim=1)

    elbo = -logrecon + torch.sum(-logdec + logenc)

    elbo = elbo.detach().cpu().numpy() * config.model.perdimsscale
    entrecon = -logrecon.detach().cpu().numpy() * config.model.perdimsscale
    entdec = -logdec.detach().cpu().numpy() * config.model.perdimsscale
    entenc = -logenc.detach().cpu().numpy() * config.model.perdimsscale

    kl = entdec - entenc

    print(f'====> Epoch: {0} Average loss: {elbo:.4f}')
    config.logger.add_text('architecture', f"{config.model}", 0)
    config.logger.add_scalar('elbo/train', elbo, 0)
    config.logger.add_scalar('x/reconstruction/train', entrecon, 0)
    for i in range(1, logdec.shape[0] + 1):
        config.logger.add_scalar(f'z{i}/encoder/train', entenc[i - 1], 0)
        config.logger.add_scalar(f'z{i}/decoder/train', entdec[i - 1], 0)
        config.logger.add_scalar(f'z{i}/KL/train', kl[i - 1], 0)


def train(epoch, data_loader, optimizer):
    # convert model to train mode (activate Dropout etc.)
    config.model.train()

    # get number of batches
    num_batch = len(data_loader)

    # setup training metrics
    elbos = 0

    start_time = time.time()

    # allocate memory for data
    data = torch.zeros((data_loader.batch_size,) + config.model.xdim, device=config.device)

    # enumerate over the batches
    for batch_idx, (batch, _) in enumerate(data_loader):
        # keep track of the global step
        # global_step = (epoch - 1) * num_batch + (batch_idx + 1)

        # update the learning rate according to schedule
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.scheduler(param_group['lr'], decay=config.decay)

        # empty all the gradients stored
        optimizer.zero_grad()

        # copy the mini-batch in the pre-allocated data-variable
        data.copy_(batch)

        # evaluate the data under the model and calculate ELBO components
        elbo, _ = config.model.loss(data, 'train')

        # calculate gradients
        elbo.backward()

        # take gradient step
        total_norm = nn.utils.clip_grad_norm_(config.model.parameters(), 1., norm_type=2)
        optimizer.step()

        # log
        elbos += elbo

        # log and save parameters
        if batch_idx % config.log_interval == 0 and config.log_interval < num_batch:
            # print metrics to console
            print(f'Train Epoch: {epoch} [{batch_idx}/{num_batch} ({100. * batch_idx / num_batch:.0f}%)]\tLoss: {elbo.item():.6f}\tGnorm: {total_norm:.2f}\tSteps/sec: {(time.time() - start_time) / (batch_idx + 1):.3f}')


            # config.logger.add_scalar('step-sec', (time.time() - start_time) / (batch_idx + 1), global_step)

            # log
            # config.logger.add_scalar('elbo/train', elbo, global_step)
            # for param_group in optimizer.param_groups:
            #     lr = param_group['lr']
            # config.logger.add_scalar('lr', lr, global_step)


    # print the average loss of the epoch to the console
    elbo = elbos.item() / num_batch
    print(f'====> Epoch: {epoch} Average loss: {elbo:.4f}')
    # config.logger.add_scalar('elbo/train', elbo, epoch)


def eval(epoch, data_loader):
    # convert model to evaluation mode (no Dropout etc.)
    config.model.eval()

    # setup the reconstruction dataset
    recon_dataset = None
    num_batch = len(data_loader)
    recon_batch_idx = int(torch.Tensor(1).random_(0, num_batch - 1))

    elbos = 0

    # allocate memory for the input data
    data = torch.zeros((data_loader.batch_size,) + config.model.xdim, device=config.device)

    # enumerate over the batches
    for batch_idx, (batch, _) in enumerate(data_loader):
        # save batch for reconstruction
        if batch_idx == recon_batch_idx:
            recon_dataset = data

        # copy the mini-batch in the pre-allocated data-variable
        data.copy_(batch)

        with torch.no_grad():
            # evaluate the data under the model and calculate ELBO components
            elbo, _ = config.model.loss(data, 'eval')

        elbos += elbo

    elbo = elbos.item() / num_batch

    # print metrics to console and Tensorboard
    print(f'\nEpoch: {epoch}\tTest loss: {elbo:.6f}')
    config.logger.add_scalar('elbo/test', elbo, epoch)

    # if the current ELBO is better than the ELBO's before, save parameters
    if elbo < config.model.best_elbo and not np.isnan(elbo):
        print("best result\n")
        config.logger.add_scalar('elbo/besttest', elbo, epoch)
        save_dir = f'model/params/{config.dataset}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        to_saved = {
            'model_params' : config.model.state_dict(),
            'elbo' : elbo,
        }
        torch.save(to_saved, save_dir+f"{config.model_name}_best.pt")
        config.model.best_elbo = elbo

        config.model.sample(config.device, epoch)
        config.model.reconstruct(recon_dataset, config.device, epoch)


if __name__ == '__main__':

    # create loggers
    config.logger = SummaryWriter(log_dir=config.log_dir)

    # build model
    config.load_model()

    # set up optimizer
    optimizer = config.optimizer

    # print and log amount of parameters
    model_parameters = filter(lambda p: p.requires_grad, config.model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of trainable parameters in model: {num_parameters}')
    config.logger.add_text(f'hyperparams', '{num_parameters}', 0)

    # set up dataloader
    config.load_data()
    kwargs = {'num_workers': 8, 'pin_memory': False}
    train_loader = DataLoader(
        dataset=config.train_set, 
        sampler=None, batch_size=config.batch_size, 
        shuffle=True, drop_last=True, **kwargs)
    test_loader = DataLoader(
        dataset=config.test_set, 
        sampler=None, batch_size=config.batch_size, 
        shuffle=False, drop_last=True, **kwargs)

    # used by bitswap : data-dependent initialization
    # warmup(train_loader, 25)

    # setup exponential moving average decay (EMA) for the parameters.
    # This basically means maintaining two sets of parameters during training/testing:
    # 1. parameters that are the result of EMA
    # 2. parameters not affected by EMA
    # The (1)st parameters are only active during test-time.
    # ema = modules.EMA(0.999)
    # with torch.no_grad():
    #     for name, param in config.model.named_parameters():
    #         # only parameters optimized using gradient-descent are relevant here
    #         if param.requires_grad:
    #             # register (1) parameters
    #             ema.register_ema(name, param.data)
    #             # register (2) parameters
    #             ema.register_default(name, param.data)

    # initial test loss
    eval(0, test_loader)

    # do the training loop and run over the test-set 1/5 epochs.
    print("Training")
    for epoch in range(1, config.epochs + 1):
        train(epoch, train_loader, optimizer)
        if epoch % 5 == 0:
            eval(epoch, test_loader)
