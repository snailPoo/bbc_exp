import numpy as np

from utils.torch.rand import get_pmfs
from .entropy_model import *

class bbc_base:
    def __init__(self, config, model, state, x_bin, z_bin):
        self.model = model
        self.state = state

        self.z_bin_ends, self.z_bin_centres = z_bin
        self.x_bin_ends, self.x_bin_centres = x_bin
        self.xrange = torch.arange(np.prod(model.xdim))
        self.zrange = torch.arange(np.prod(model.zdim))
        self.z_quantbits = config.z_quantbits
        self.x_quantbits = config.x_quantbits
        self.restbits = None
        self.restbits_tag = True

        self.prior_mu = torch.zeros(1, device=config.device, dtype=config.type)
        self.prior_scale = torch.ones(1, device=config.device, dtype=config.type)

    def variable_encode(self, to_encode, model_op, i, given, bin_ends, quantbits):
        ops = self.model.generate if model_op == 'gen' else self.model.infer
        mu, scale = ops(i)(given=given)
        pmfs = get_pmfs(bin_ends.t(), mu, scale)
        self.state = ANS(pmfs, quantbits).encode(self.state, to_encode)
        
    def variable_decode(self, model_op, i, given, bin_ends, quantbits):
        ops = self.model.generate if model_op == 'gen' else self.model.infer
        mu, scale = ops(i)(given=given)
        pmfs = get_pmfs(bin_ends.t(), mu, scale)
        self.state, z_symtop = ANS(pmfs, quantbits).decode(self.state)
        return z_symtop
    
    def prior_encode(self, z, bin_ends):
        pmfs = get_pmfs(bin_ends.t(), self.prior_mu, self.prior_scale)
        self.state = ANS(pmfs, self.z_quantbits).encode(self.state, z)
    
    def prior_decode(self, bin_ends):
        pmfs = get_pmfs(bin_ends.t(), self.prior_mu, self.prior_scale)
        self.state, z_symtop = ANS(pmfs, self.z_quantbits).decode(self.state)
        return z_symtop

# should customize encoding/decoding process based on latent variable dependency graph
class BitSwap(bbc_base):
    def __init__(self, config, model, state, x_bin, z_bin):
        # bbc_base.__init__(self, config, model, state, x_bin, z_bin)
        super().__init__(config, model, state, x_bin, z_bin)

    def encoding(self, x):
        for zi in range(self.model.nz):
            given = self.z_bin_centres[zi - 1, self.zrange, z_sym] if zi > 0 else self.x_bin_centres[self.xrange, x.long()]
            z_symtop = self.variable_decode('inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            
            # save excess bits for calculations
            if self.restbits_tag:
                self.restbits = self.state.copy()
                assert len(self.restbits) > 1, "too few initial bits" # otherwise initial state consists of too few bits
                self.restbits_tag = False            

            given = self.z_bin_centres[zi, self.zrange, z_symtop] # z
            if zi == 0:
                self.variable_encode(x.long(), 'gen', zi, given, self.x_bin_ends, self.x_quantbits)
            else:
                self.variable_encode(z_sym, 'gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)

            z_sym = z_symtop

        # encode prior
        self.prior_encode(z_symtop, self.z_bin_ends[-1])
        return self.state
    
    def decoding(self):
        z_symtop = self.prior_decode(self.z_bin_ends[-1])
        
        for zi in reversed(range(self.model.nz)):
            given = self.z_bin_centres[zi, self.zrange, z_symtop]
            if zi == 0:
                sym = self.variable_decode('gen', zi, given, self.x_bin_ends, self.x_quantbits)
                given = self.x_bin_centres[self.xrange, sym]
            else:
                sym = self.variable_decode('gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)
                given = self.z_bin_centres[zi - 1, self.zrange, sym]

            self.variable_encode(z_symtop, 'inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            
            z_symtop = sym

        return z_symtop, self.state

class BBC(bbc_base):
    def __init__(self, config, model, state, x_bin, z_bin):
        # bbc_base.__init__(self, config, model, state, x_bin, z_bin)
        super(BBC, self).__init__(config, model, state, x_bin, z_bin)
    
    def encoding(self, x):
        zs = []
        # decode all latent variables
        for zi in range(self.model.nz):
            given = self.z_bin_centres[zi - 1, self.zrange, z_sym] if zi > 0 else self.x_bin_centres[self.xrange, x.long()]
            z_symtop = self.variable_decode('inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            zs.append(z_symtop)
            z_sym = z_symtop

        # save excess bits for calculations
        if self.restbits_tag:
            self.restbits = self.state.copy()
            assert len(self.restbits) > 1, "too few initial bits" # otherwise initial state consists of too few bits
            self.restbits_tag = False

        # decode latent variables and data x
        for zi in range(self.model.nz):
            z_symtop = zs.pop(0)
            given = self.z_bin_centres[zi, self.zrange, z_symtop]
            if zi == 0:
                self.variable_encode(x.long(), 'gen', zi, given, self.x_bin_ends, self.x_quantbits)
            else:
                self.variable_encode(z_sym, 'gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)
            z_sym = z_symtop

        assert zs == []
        self.prior_encode(z_symtop, self.z_bin_ends[-1])

    def decoding(self):
        z_symtop = self.prior_decode(self.z_bin_ends[-1])
        zs = [z_symtop]
        for zi in reversed(range(self.model.nz)):
            given = self.z_bin_centres[zi, self.zrange, z_symtop]
            if zi == 0:
                sym = self.variable_decode('gen', zi, given, self.x_bin_ends, self.x_quantbits)
            else:
                sym = self.variable_decode('gen', zi, given, self.z_bin_ends[zi - 1], self.z_quantbits)
            zs.append(sym)
            z_symtop = sym

        z_symtop = zs.pop(0)
        for zi in reversed(range(self.model.nz)):
            sym = zs.pop(0) if zi > 0 else zs[0]
            given = self.z_bin_centres[zi - 1, self.zrange, sym] if zi > 0 else self.x_bin_centres[self.xrange, sym]
            self.variable_encode(z_symtop, 'inf', zi, given, self.z_bin_ends[zi], self.z_quantbits)
            z_symtop = sym
        
        return z_symtop, self.state
