import torch
import numpy as np

class ANS:
    def __init__(self, pmfs, quantbits):
        self.device = pmfs.device
        self.bits = 31
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << self.bits) - 1

        # normalization constants
        self.lbound = 1 << 32
        self.tail_bits = (1 << 32) - 1

        self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        # add remnant to the maimum value of the probabilites
        self.pmfs[torch.arange(0, self.seq_len),torch.argmax(self.pmfs, dim=1)] += ((1 << self.bits) - self.pmfs.sum(1))

        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=1) # compute CDF (scaled up to 2**n)
        self.cdfs = torch.cat([torch.zeros([self.cdfs.shape[0], 1], dtype=torch.long, device=self.device), self.cdfs], dim=1) # pad with 0 at the beginning

        # move cdf's and pmf's the cpu for faster encoding and decoding
        self.cdfs = self.cdfs.cpu().numpy()
        self.pmfs = self.pmfs.cpu().numpy()

        assert self.cdfs.shape == (self.seq_len, self.support + 1)
        assert np.all(self.cdfs[:,-1] == (1 << self.bits))

    def encode(self, x, symbols):
        for i, s in enumerate(symbols):
            pmf = int(self.pmfs[i,s])
            if x[-1] >= ((self.lbound >> self.bits) << 32) * pmf:
                x.append(x[-1] >> 32)
                x[-2] = x[-2] & self.tail_bits
            x[-1] = ((x[-1] // pmf) << self.bits) + (x[-1] % pmf) + int(self.cdfs[i, s])
        return x

    def decode(self, x):
        sequence = np.zeros((self.seq_len,), dtype=np.int64)
        for i in reversed(range(self.seq_len)):
            masked_x = x[-1] & self.mask
            s = np.searchsorted(self.cdfs[i,:-1], masked_x, 'right') - 1
            sequence[i] = s
            x[-1] = int(self.pmfs[i,s]) * (x[-1] >> self.bits) + masked_x - int(self.cdfs[i, s])
            if x[-1] < self.lbound:
                x[-1] = (x[-1] << 32) | x.pop(-2)
        sequence = torch.from_numpy(sequence).to(self.device)
        return x, sequence

# still in development
class batch_ANS:
    def __init__(self, pmfs, quantbits):
        self.device = pmfs.device
        self.bits = 31
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << self.bits) - 1

        # normalization constants
        self.lbound = 1 << 32
        self.tail_bits = (1 << 32) - 1

        self.batch_size, self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        # add remnant to the maimum value of the probabilites
        self.pmfs[:, torch.arange(0, self.seq_len), torch.argmastate(self.pmfs, dim=2)] += ((1 << self.bits) - self.pmfs.sum(2))

        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=2) # compute CDF (scaled up to 2**n)
        self.cdfs = torch.cat([torch.zeros((*self.cdfs.shape[:2], 1), dtype=torch.long, device=self.device), self.cdfs], dim=2) # pad with 0 at the beginning

        # move cdf's and pmf's the cpu for faster encoding and decoding
        # self.cdfs = self.cdfs.cpu().numpy()
        # self.pmfs = self.pmfs.cpu().numpy()

        assert self.cdfs.shape == (self.batch_size, self.seq_len, self.support + 1)
        assert np.all(self.cdfs[:,-1] == (1 << self.bits))

    def encode(self, state, symbols): # (B, ), (B, seq_len)
        for i in range(self.seq_len):
            s = symbols[:, i] # (B, )
            pmf = self.pmfs[:, i, s] # (B, )
            if state[-1] >= ((self.lbound >> self.bits) << 32) * pmf:
                state.append(state[-1] >> 32)
                state[-2] = state[-2] & self.tail_bits
            state[-1] = ((state[-1] // pmf) << self.bits) + (state[-1] % pmf) + int(self.cdfs[:, i, s])
        return state

    def decode(self, state):
        sequence = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)
        for i in reversed(range(self.seq_len)):
            masked_state = state[-1] & self.mask
            s = np.searchsorted(self.cdfs[:, i,:-1], masked_state, 'right') - 1
            sequence[i] = s
            state[-1] = int(self.pmfs[:, i, s]) * (state[-1] >> self.bits) + masked_state - int(self.cdfs[:, i, s])
            if state[-1] < self.lbound:
                state[-1] = (state[-1] << 32) | state.pop(-2)
        sequence = torch.from_numpy(sequence).to(self.device)
        return state, sequence