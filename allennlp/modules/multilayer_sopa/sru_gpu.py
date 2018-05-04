import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from collections import namedtuple
from pynvrtc.compiler import Program
from cupy.cuda import function
import numpy as np
from multilayer_sopa.cuda.utils  import *
from multilayer_sopa.cuda.sru import *
from multilayer_sopa.cuda.sru_bidir  import *


class SRU_Compute_GPU(Function):

    _SRU_PROG = Program((UTIL + SRU + SRU_BIDIR).encode('utf-8'), 'sru_prog.cu'.encode())
    _SRU_PTX = _SRU_PROG.compile()
    _DEVICE2FUNC = {}

    def __init__(self, d_out, k, bidirectional=False, activation_type=0):
        super(SRU_Compute_GPU, self).__init__()
        self.k = k
        self.d_out = d_out
        self.bidirectional = bidirectional
        self.activation_type = activation_type  # recurrent activation

    def compile_functions(self):
        device = torch.cuda.current_device()
        print ('SRU loaded for gpu {}'.format(device))
        mod = function.Module()
        mod.load(bytes(self._SRU_PTX.encode()))
        fwd_func = mod.get_function('sru_fwd')
        bwd_func = mod.get_function('sru_bwd')
        bifwd_func = mod.get_function('sru_bi_fwd')
        bibwd_func = mod.get_function('sru_bi_bwd')

        Stream = namedtuple('Stream', ['ptr'])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        self._DEVICE2FUNC[device] = (current_stream, fwd_func,
            bifwd_func, bwd_func, bibwd_func
        )
        return current_stream, fwd_func, bifwd_func, bwd_func, bibwd_func

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()

    def forward(self, u, init=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k - 1
        length, batch = u.size(0), u.size(1)
        d = self.d_out
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = u.new(ncols).zero_() if init is None else init
        size = (length, batch, bidir, d)
        c = u.new(*size)

        stream, fwd_func, bifwd_func, _, _ = self.get_functions()
        FUNC = fwd_func if not self.bidirectional else bifwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            init_.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(d),
            np.int32(self.k-1),
            c.data_ptr(),
            np.int32(self.activation_type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        self.save_for_backward(u, init)
        self.intermediate = c
        if self.bidirectional:
            last_hidden = torch.cat((c[-1,:,0,:], c[-1,:,1,:]), dim=1)
        else:
            last_hidden = c[-1, ...].view(batch, -1)
        return c, last_hidden

    def backward(self, grad_c, grad_last):
        bidir = 2 if self.bidirectional else 1
        u, init = self.saved_tensors
        c = self.intermediate
        length, batch = u.size(0), u.size(1)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = u.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init = u.new(batch, d*bidir)

        # For DEBUG
        #size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        #grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        stream, _, _, bwd_func, bibwd_func = self.get_functions()
        FUNC = bwd_func if not self.bidirectional else bibwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            init_.contiguous().data_ptr(),
            c.data_ptr(),
            grad_c.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(d),
            np.int32(self.k-1),
            grad_u.data_ptr(),
            grad_init.data_ptr(),
            np.int32(self.activation_type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        return grad_u, grad_init
