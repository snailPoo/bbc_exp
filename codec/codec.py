from tqdm import tqdm
import pickle
import os

from .bbc_scheme import *
from utils.torch.rand import ImageBins

'''
# pseudo code
# param : model, dataset, BBC scheme
1. load model
2. discretization
3. load dataset, create dataloader
4. compression
    if BBC:
        loop:
            prior model
            decode z
        model generation
        encode x
        model inference
        encode z
    if bit-swap:
        loop:
            prior model
            decode z
            model generation
            encode x or z
        model inference
        encode z
'''
class Codec:
    def __init__(self, config, dataloader, state):
        self.cf = config
        self.model = self.cf.model
        self.dataloader = dataloader
        self.state = state
        self.initialstate = self.state.copy()
        self.decoded_data = []
        
        self.total_xdim = np.prod(self.model.xdim)
        
        print("discretizing")
        self.z_bin = self.cf.discretization(self.cf)
        xbin = ImageBins(self.cf.type, self.cf.device, self.total_xdim)
        self.x_bin = xbin.endpoints(), xbin.centres()

    def compress(self):
        reswidth = 252 # why?

        # metrics for the results
        nets  = np.zeros((len(self.cf.data_to_compress), ), dtype=np.float64)
        elbos = np.zeros((len(self.cf.data_to_compress), ), dtype=np.float64)
        cma   = np.zeros((len(self.cf.data_to_compress), ), dtype=np.float64)
        total = np.zeros((len(self.cf.data_to_compress), ), dtype=np.float64)

        print("Start compressing images")
        self.model.compress_mode(True)

        # ******* sender *******
        iterator = tqdm(self.dataloader, total=len(self.dataloader), desc="Sender")
        for i, (x, _) in enumerate(iterator):
            # print(f'{i}-th {len(self.state)=}')
            if i == 10:
                break
            x = x.to(self.cf.device).view(self.total_xdim)

            # calculate ELBO
            with torch.no_grad():
                self.model.compress_mode(False)
                elbo, _ = self.model.loss(x.view((-1,) + self.model.xdim), 'test')
                self.model.compress_mode(True)

            if self.cf.bbc_scheme == 'bitswap':
                scheme = BitSwap(self.cf, self.model, self.state, self.x_bin, self.z_bin)
            else:
                scheme = BBC(self.cf, self.model, self.state, self.x_bin, self.z_bin)
            
            self.state = scheme.encoding(x)

            # calculating bits
            total_added_bits = (len(self.state) - len(self.initialstate)) * 32
            totalbits = (len(self.state) - (len(scheme.restbits) - 1)) * 32

            # logging
            nets[i]  = (total_added_bits / self.total_xdim) - nets[:i].sum()
            elbos[i] = elbo.item() / self.total_xdim
            cma[i]   = totalbits / (self.total_xdim * (i + 1))
            total[i] = totalbits

            iterator.set_postfix_str(s=f"N:{nets[:i+1].mean():.2f}±{nets[:i+1].std():.2f}, " + 
                                       f"D:{nets[:i+1].mean()-elbos[:i+1].mean():.4f}, " +
                                       f"C: {cma[:i+1].mean():.2f}, " +
                                       f"T: {totalbits:.0f}", refresh=False)

        # write state to file
        if not os.path.exists(self.cf.state_dir):
            os.makedirs(self.cf.state_dir)
        with open(os.path.join(self.cf.state_dir, f"{self.cf.model_name}.pt"), "wb") as fp:
            pickle.dump(self.state, fp)
        
        print(f"N:{nets.mean():.4f}±{nets.std():.2f}, " +
              f"E:{elbos.mean():.4f}±{elbos.std():.2f}, " +
              f"D:{nets.mean() - elbos.mean():.6f}")
        
        print('compression complete')
        return

    def decompress(self):
        # read state file
        with open(os.path.join(self.cf.state_dir, f"{self.cf.model_name}.pt"), "rb") as fp:
            self.state = pickle.load(fp)
        
        print("Start decompressing")
        # ****** receiver ******
        iterator = tqdm(range(len(self.dataloader)), desc="Receiver")
        for i in iterator:
            # print(f'{i}-th {len(self.state)=}')
            if i == 10:
                break
            scheme = BitSwap if self.cf.bbc_scheme == 'bitswap' else BBC
            x, self.state = scheme(self.cf, self.model, self.state, self.x_bin, self.z_bin).decoding()

            self.decoded_data.insert(0, x.view(self.model.xdim))

        # check if the initial state matches the output state
        assert self.initialstate == self.state
        return self.decoded_data