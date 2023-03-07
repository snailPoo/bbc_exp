import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from config import Config
from codec.codec import Codec

config = Config()

# seed for replicating experiment and stability
np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    
    # ******* model *******
    config.load_model()
    config.model.eval()
    # *********************

    # ******* data ********
    config.load_data()
    config.data_to_compress = config.test_set
    dataloader = DataLoader(dataset=config.data_to_compress, 
                            batch_size=config.compression_batch_size, 
                            shuffle=False, drop_last=True)
    # *********************

    # ******* state *******
    # fill state list with 'random' bits
    state = list(map(int, np.random.randint(low=1 << 16, high=(1 << 32) - 1, 
                                            size=config.init_state_size, 
                                            dtype=np.uint32)))
    state[-1] = state[-1] << 32
    # *********************

    codec = Codec(config, dataloader, state)
    codec.compress()
    decompressed_data = codec.decompress()

    # check if decompressed_data == original
    datapoints = list(dataloader)
    for i, x in enumerate(decompressed_data):
        assert torch.equal(x, datapoints[i][0][0].to(torch.int64).to(config.device))