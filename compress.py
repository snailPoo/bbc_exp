import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import *
from codec.codec import Codec
from utils.common import same_seed, load_model, load_data

cf = Config_bitswap()

# seed for replicating experiment and stability
same_seed(cf.seed)
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    print(f"Model:{cf.model_name}; Dataset:{cf.dataset}")

    # ******* data ********
    train_set, test_set = load_data(cf.dataset, cf.model_name)
    train_loader = DataLoader(
                    dataset=train_set, 
                    batch_size=cf.batch_size, 
                    shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, 
                             batch_size=cf.compression_batch_size, 
                             shuffle=False, drop_last=True)
    cf.model_hparam.xdim = train_set[0][0].shape
    # *********************

    # ******* model *******
    model, _, _ = load_model(cf.model_name, cf.model_pt, 
                             cf.model_hparam, cf.lr, cf.decay)
    # model.logger = SummaryWriter(log_dir=cf.log_dir)
    model.eval().to(cf.device)
    # *********************

    # ******* state *******
    # fill state list with 'random' bits
    state = list(map(int, np.random.randint(low=1 << 16, high=(1 << 32) - 1, 
                                            size=cf.init_state_size, 
                                            dtype=np.uint32)))
    state[-1] = state[-1] << 32
    # *********************

    num_images = 1000#test_loader.__len__()
    codec = Codec(cf, model, (train_loader, test_loader), state, num_images)

    encode_t0 = time.time()
    codec.compress()
    encode_t = time.time() - encode_t0
    print("All encoded in {:.2f}s.".format(encode_t))
    print("Average singe image encoding time: {:.2f}s.".format(encode_t / num_images))

    decode_t0 = time.time()
    decompressed_data = codec.decompress()
    decode_t = time.time() - decode_t0
    print('All decoded in {:.2f}s.'.format(decode_t))
    print("Average singe image decoding time: {:.2f}s.".format(decode_t / num_images))

    # check if decompressed_data == original
    datapoints = list(test_loader)
    for i, x in enumerate(decompressed_data):
        assert torch.equal(x, datapoints[i][0][0].to(torch.int64).to(cf.device))