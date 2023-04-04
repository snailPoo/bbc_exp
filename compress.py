import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from bbc_exp.config import *
from codec.codec import Codec
from utils.common import same_seed, load_model, load_data

cf = Config_bitswap()

# seed for replicating experiment and stability
# same_seed(cf.seed)
np.random.seed(cf.seed)
random.seed(cf.seed)
torch.manual_seed(cf.seed)
torch.cuda.manual_seed(cf.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    # ******* data ********
    train_set, test_set = load_data(cf.dataset)
    train_loader = DataLoader(
                    dataset=train_set, 
                    batch_size=128, 
                    shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, 
                             batch_size=cf.compression_batch_size, 
                             shuffle=False, drop_last=True)
    cf.model_hparam.xdim = train_set[0][0].shape
    # *********************

    # ******* model *******
    model, _, _ = load_model(cf.model_name, cf.model_pt, 
                             cf.model_hparam, cf.lr, cf.decay)
    model.logger = SummaryWriter(log_dir=cf.log_dir)
    model.eval().to(cf.device)
    # *********************

    # ******* state *******
    # fill state list with 'random' bits
    state = list(map(int, np.random.randint(low=1 << 16, high=(1 << 32) - 1, 
                                            size=cf.init_state_size, 
                                            dtype=np.uint32)))
    state[-1] = state[-1] << 32
    # *********************

    codec = Codec(cf, model, (train_loader, test_loader), state)
    codec.compress()
    decompressed_data = codec.decompress()

    # check if decompressed_data == original
    datapoints = list(test_loader)
    for i, x in enumerate(decompressed_data):
        assert torch.equal(x, datapoints[i][0][0].to(torch.int64).to(cf.device))