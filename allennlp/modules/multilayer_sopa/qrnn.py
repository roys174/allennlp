#from builtins import bytes
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from allennlp.modules.multilayer_sopa.sru_gpu import *
from allennlp.modules.multilayer_sopa.sru import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class QRNNCell(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 window_size=1,
                 dropout=0,
                 rnn_dropout=0,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 use_output_gate=False,
                 weight_norm=False,
                 layer_norm=False,
                 index=-1):
        super(QRNNCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.weight_norm = weight_norm
        self.layer_norm = layer_norm
        self.index = index
        self.activation_type = 0
        if use_tanh:
            self.activation_type = 1
        elif use_relu:
            self.activation_type = 2
        elif use_selu:
            self.activation_type = 3
        self.window_size = window_size
        self.use_output_gate = use_output_gate

        out_size = n_out*2 if bidirectional else n_out
        k = 3 if use_output_gate else 2

        self.k = k
        self.size_per_dir = n_out*k
        self.weight = nn.Parameter(torch.Tensor(
            self.window_size * n_in,
            self.size_per_dir*2 if bidirectional else self.size_per_dir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            self.size_per_dir*2 if bidirectional else self.size_per_dir
        ))
        self.init_weight()

    def init_weight(self, rescale=True):
        # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
        val_range = (3.0/self.n_in)**0.5
        self.weight.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias.data.zero_()

        # re-scale weights in case there's dropout and / or layer normalization
        w = self.weight.data.view(self.window_size*self.n_in, -1, self.n_out, self.k)
        if self.dropout>0:
            w[:,:,:,0].mul_((1-self.dropout)**0.5)
        if self.rnn_dropout>0:
            w.mul_((1-self.rnn_dropout)**0.5)
        if self.layer_norm:
            w[:,:,:,1].mul_(0.1)
            w[:,:,:,2].mul_(0.1)

        # re-parameterize when weight normalization is enabled
        if self.weight_norm:
            self.init_weight_norm()

    def init_weight_norm(self):
        weight = self.weight.data
        g = weight.norm(2, 0)
        self.gain = nn.Parameter(g)

    def apply_weight_norm(self, eps=0):
        wnorm = self.weight.norm(2, 0)#, keepdim=True)
        return self.gain.expand_as(self.weight).mul(
            self.weight / (wnorm.expand_as(self.weight) + eps)
        )

    def set_bias(self, args):
        sys.stderr.write("\nWARNING: set_bias() is deprecated. use `highway_bias` option"
            " in QRNNCell() constructor.\n"
        )
        self.highway_bias = args.highway_bias
        self.init_weight()
        #n_out = self.n_out
        #if self.bidirectional:
        #    self.bias.data[n_out*2:].zero_().add_(bias_val)
        #else:
        #    self.bias.data[n_out:].zero_().add_(bias_val)

    def calc_activation(self, x):
        if self.activation_type == 0:
            return x
        elif self.activation_type == 1:
            return x.tanh()
        elif self.activation_type == 2:
            return nn.functional.relu(x)
        else:
            assert False, 'Activation type must be 0, 1, or 2, not {}'.format(self.activation_type)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        length, batch = input.size(0), input.size(-2)
        bidir = 2 if self.bidirectional else 1
        if c0 is None:
            c0 = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2
            ).zero_())

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input
        if self.window_size == 1:
            x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
            weight = self.weight if not self.weight_norm else self.apply_weight_norm()
            u_ = x_2d.mm(weight)
            u_ = u_.view(length, batch, bidir, n_out, self.k)
        elif self.window_size == 2:
            Xm1 = []
            Xm1.append(x[:1, :, :] * 0)
            if len(x) > 1:
                Xm1.append(x[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([x, Xm1], 2)
            x_2d = source if source.dim() == 2 else source.contiguous().view(-1, n_in * 2)
            weight = self.weight if not self.weight_norm else self.apply_weight_norm()
            u_ = x_2d.mm(weight)
            u_ = u_.view(length, batch, bidir, n_out, self.k)
        else:
            assert False    # yes they only use up to 2.


        if self.use_output_gate:
            input_bias, forget_bias, output_bias = self.bias.view(self.k, bidir, self.n_out)
        else :
            input_bias, forget_bias = self.bias.view(self.k, bidir, self.n_out)
        u = Variable(u_.data.new(length, batch, bidir, n_out, 2))

        u[..., 0] = (u_[..., 0] + input_bias).tanh()
        u[..., 1] = (u_[..., 1] + forget_bias).sigmoid()
        if input.is_cuda:
            QRNN_Compute = SRU_Compute_GPU(n_out, 3, self.bidirectional)
        else:
            QRNN_Compute = SRU_Compute_CPU(n_out, 3, self.bidirectional)

        cs, c_final = QRNN_Compute(u, c0)
        gcs = self.calc_activation(cs)

        if self.use_output_gate:
            output = (u_[..., 2] + output_bias).sigmoid()
            h = output * gcs
        else:
            h = gcs
        return h.view(length, batch, -1), c_final

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))



class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=2,
                 window_size=1,
                 dropout=0,
                 rnn_dropout=0,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 use_output_gate=False,
                 weight_norm=False,
                 layer_norm=False):
        super(QRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.ln_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.use_wieght_norm = weight_norm
        self.out_size = hidden_size*2 if bidirectional else hidden_size
        if use_tanh + use_relu + use_selu > 1:
            sys.stderr.write("\nWARNING: More than one activation enabled in QRNN"
                " (tanh: {}  relu: {}  selu: {})\n".format(use_tanh, use_relu, use_selu)
            )

        for i in range(num_layers):
            l = QRNNCell(
                n_in = self.input_size if i == 0 else self.out_size,
                n_out = self.hidden_size,
                window_size=window_size,
                dropout = dropout if i+1 != num_layers else 0,
                rnn_dropout = rnn_dropout,
                bidirectional = bidirectional,
                use_tanh = use_tanh,
                use_relu = use_relu,
                use_selu = use_selu,
                use_output_gate=use_output_gate,
                weight_norm = weight_norm,
                layer_norm = layer_norm,
                index = i+1
            )
            self.rnn_lst.append(l)
            if layer_norm:
                self.ln_lst.append(LayerNorm(self.hidden_size))

    def set_bias(self,args):
        for l in self.rnn_lst:
            l.set_bias(args)

    def forward(self, input, init=None, return_hidden=True):
        input, lengths = pad_packed_sequence(input, batch_first=True)
        assert input.dim() == 3 # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if init is None:
            zeros = Variable(input.data.new(
                input.size(1), self.hidden_size * dir_
            ).zero_())
            init = [zeros for i in range(self.num_layers)]
        else:
            assert init.dim() == 3    # (depth, batch, n_out*dir_)
            init = [x.squeeze(0) for x in init.chunk(self.num_layers, 0)]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, init[i])
            prevx = self.ln_lst[i](h) if self.use_layer_norm else h
            lstc.append(c)

        prevx = pack_padded_sequence(prevx, lengths, batch_first=True)
        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx


class LayerNorm(nn.Module):
    '''
    Layer normalization module modified from:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/UtilClass.py
    '''

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x):
        if x.size(-1) == 1:
            return x
        mu = torch.mean(x, dim=-1)
        sigma = torch.std(x, dim=-1, unbiased=False)
        # HACK. PyTorch is changing behavior
        if mu.dim() == x.dim()-1:
            mu = mu.unsqueeze(mu.dim())
            sigma = sigma.unsqueeze(sigma.dim())
        output = (x - mu.expand_as(x)) / (sigma.expand_as(x) + self.eps)
        output = output.mul(self.a.expand_as(output)) \
            + self.b.expand_as(output)
        return output
