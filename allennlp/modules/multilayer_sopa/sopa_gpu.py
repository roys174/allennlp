
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from collections import namedtuple
from pynvrtc.compiler import Program
from cupy.cuda import function
import numpy as np
from multilayer_sopa.cuda.utils  import *
from multilayer_sopa.cuda.sopa  import *
from multilayer_sopa.cuda.sopa_bidir  import *

class SOPA_Compute_GPU(Function):

    _SOPA_PROG = Program((UTIL+SOPA+SOPA_BIDIR).encode('utf-8'), 'sopa_prog.cu'.encode())
    _SOPA_PTX = _SOPA_PROG.compile()
    _DEVICE2FUNC = {}

    def __init__(self, d_out, k, bidirectional=False):
        super(SOPA_Compute_GPU, self).__init__()
        self.d_out = d_out
        self.k = k
        self.bidirectional = bidirectional

    def compile_functions(self):
        device = torch.cuda.current_device()
        print ('SOPA loaded for gpu {}'.format(device))
        mod = function.Module()
        mod.load(bytes(self._SOPA_PTX.encode()))
        fwd_func = mod.get_function('sopa_fwd')
        bwd_func = mod.get_function('sopa_bwd')
        bifwd_func = mod.get_function('sopa_bifwd')
        bibwd_func = mod.get_function('sopa_bibwd')

        Stream = namedtuple('Stream', ['ptr'])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        self._DEVICE2FUNC[device] = (
            current_stream, fwd_func, bwd_func,
            bifwd_func, bibwd_func,
        )
        return current_stream, fwd_func, bwd_func, bifwd_func, bibwd_func,

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()

    def forward(self, u, c1_init=None, c2_init=None, d_init=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k - 1
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c1_init is None:
            assert False
            # assert d_init is None
            # c1_init = u.new(ncols).zero_()
            # c2_init = u.new(ncols).zero_()
            # d_init = u.new(ncols).zero_()
        # init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, bidir, dim)
        c1s = u.new(*size)
        c2s = u.new(*size)
        # size = (length, batch, dim*bidir)
        d = u.new(*size)
        stream, fwd_func, _, bifwd_func, _ = self.get_functions()
        FUNC = fwd_func if bidir == 1 else bifwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            d_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k-1),
            c1s.data_ptr(),
            c2s.data_ptr(),
            d.data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, c1_init, c2_init, d_init)
        self.intermediate_c1s, self.intermediate_c2s = c1s, c2s
        self.intermediate_d = d
        if self.bidirectional:
            last_c1, last_c2, last_d \
                = torch.cat((c1s[-1,:,0,:], c1s[0,:,1,:]), dim=1), \
                  torch.cat((c2s[-1,:,0,:], c2s[0,:,1,:]), dim=1), \
                  torch.cat((d[-1,:,0,:], d[0,:,1,:]), dim=1)
        else:
            last_c1 = c1s[-1,...].view(batch, -1)
            last_c2 = c2s[-1,...].view(batch, -1) 
            last_d = d[-1,...].view(batch, -1)
        return c1s, c2s, last_c1, last_c2, last_d

    def backward(self, grad_c1s, grad_c2s, grad_last_c1, grad_last_c2, grad_last_d):
        bidir = 2 if self.bidirectional else 1
        u, c1_init, c2_init, d_init = self.saved_tensors
        c1s, c2s = self.intermediate_c1s, self.intermediate_c2s
        d = self.intermediate_d
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c1_init is None:
            c1_init = u.new(ncols).zero_()
            c2_init = u.new(ncols).zero_()
            d_init = u.new(ncols).zero_()

        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init_c1 = u.new(batch, dim*bidir)
        grad_init_c2 = u.new(batch, dim*bidir)
        grad_init_d = u.new(batch, dim*bidir)

        stream, _, bwd_func, _, bibwd_func = self.get_functions()
        FUNC = bwd_func if bidir == 1 else bibwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            d_init.contiguous().data_ptr(),
            c1s.data_ptr(),
            c2s.data_ptr(),
            d.data_ptr(),
            grad_c1s.data_ptr(),
            grad_c2s.data_ptr(),
            grad_last_c1.contiguous().data_ptr(),
            grad_last_c2.contiguous().data_ptr(),
            grad_last_d.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k-1),
            grad_u.data_ptr(),
            grad_init_c1.data_ptr(),
            grad_init_c2.data_ptr(),
            grad_init_d.data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        return grad_u, grad_init_c1, grad_init_c2, grad_init_d
