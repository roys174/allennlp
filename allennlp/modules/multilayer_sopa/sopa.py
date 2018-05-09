import sys
import math
#import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# non_cuda_compatability: comment the line below when not using cuda
from allennlp.modules.multilayer_sopa.sopa_gpu import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def SOPA_Compute_CPU(d, k, bidirectional=False):
    """CPU version of the core SOPA computation.

    Has the same interface as SOPA_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """

    def sopa_compute_cpu(u, c1_init=None, c2_init=None, d_init=None):
        bidir = 2 if bidirectional else 1
        assert u.size(-1) == k
        length, batch = u.size(0), u.size(1)
        if c1_init is None:
            assert False
            # c1_init = Variable(u.data.new(batch, bidir, d).zero_())
            # c2_init = Variable(u.data.new(batch, bidir, d).zero_())
            # d_init = Variable(u.data.new(batch, bidir, d).zero_())
        else:
            c1_init = c1_init.contiguous().view(batch, bidir, d)
            c2_init = c2_init.contiguous().view(batch, bidir, d)
            d_init = d_init.contiguous().view(batch, bidir, d)

        x_tilde1, x_tilde2, x_tilde3, forget1, forget2, selfloop \
            = u[..., 0], u[..., 1],u[..., 2],u[..., 3], u[..., 4], u[..., 5]

        c1_final, c2_final, d_final = [], [], []
        c1s = Variable(u.data.new(length, batch, bidir, d))
        c2s = Variable(u.data.new(length, batch, bidir, d))

        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c1_prev = c1_init[:, di, :]
            c2_prev = c2_init[:, di, :]
            d_prev = d_init[:, di, :]

            for t in time_seq:
                c1_t = (c1_prev - x_tilde1[t, :, di, :]) * forget1[t, :, di, :] + x_tilde1[t, :, di, :]
                tmp = x_tilde3[t, :, di, :] * d_prev
                c2_t = (c2_prev - tmp) * forget2[t, :, di, :] + tmp

                # somehow arbitrary design choice. we could revisit this gate.
                # d_t = (d_prev - x_tilde2[t, :, di, :]) * selfloop[t, :, di, :] + x_tilde2[t, :, di, :]
                # d_t = d_prev * selfloop[t, :, di, :] + x_tilde2[t, :, di, :]
                d_t = x_tilde2[t, :, di, :]
                c1_prev, c2_prev, d_prev = c1_t, c2_t, d_t
                c1s[t,:,di,:], c2s[t,:,di,:]  = c1_t, c2_t

            c1_final.append(c1_t)
            c2_final.append(c2_t)
            d_final.append(d_t)

        return c1s, c2s, \
               torch.stack(c1_final, dim=1).view(batch, -1), \
               torch.stack(c2_final, dim=1).view(batch, -1), \
               torch.stack(d_final, dim=1).view(batch, -1)

    return sopa_compute_cpu


class SOPACell(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 dropout=0.,
                 rnn_dropout=0.,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 weight_norm=False,
                 layer_norm=False,
                 highway_bias=0,
                 selfloop_bias=0,
                 index=-1,
                 coef=0.5,
                 use_highway=False,
                 use_output_gate=False,
                 window_size=1):
        super(SOPACell, self).__init__()
        assert (n_out % 2) == 0
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.bidir = 2 if self.bidirectional else 1
        self.weight_norm = weight_norm
        self.layer_norm = layer_norm
        self.index = index
        self.activation_type = 0
        self.coef = coef
        self.use_highway = use_highway
        self.use_output_gate = use_output_gate  # borrowed from qrnn
        self.highway_bias = highway_bias
        self.selfloop_bias = selfloop_bias
        self.window_size = window_size
        assert window_size <= 2    # only 1 or 2 is supported

        if use_tanh:
            self.activation_type = 1
        elif use_relu:
            self.activation_type = 2
        elif use_selu:
            self.activation_type = 3

        # i'm sticking with this for now, and may come back later
        # basic: in1, in2, in3, f1, f2, s1
        # optional: reset, output, highway affine.
        self.k = 7 if self.use_output_gate else 6
        # we do not use bias in highway affine
        self.n_bias = 7 if self.use_output_gate else 6
        if self.use_highway:
            self.k += 1
            self.n_bias += 1
            if n_in != n_out:
                self.k += 1
        self.size_per_dir = n_out*self.k

        self.weight_in = nn.Parameter(torch.Tensor(
            window_size*n_in,
            self.size_per_dir*self.bidir
        ))
        self.weight_out1 = nn.Parameter(torch.Tensor(
            self.bidir*n_out, self.bidir*n_out))
        self.weight_out2 = nn.Parameter(torch.Tensor(
            self.bidir*n_out, self.bidir*n_out))

        self.bias_in = nn.Parameter(torch.Tensor(
            n_out*self.n_bias*self.bidir
        ))
        self.bias_out = nn.Parameter(torch.Tensor(self.bidir*n_out))

        self.init_weight()

    def init_weight(self, rescale=True):
        # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
        val_range = (6.0 / (self.window_size*self.n_in + self.n_out)) ** 0.5
        self.weight_in.data.uniform_(-val_range, val_range)
        val_range = (2.0 / self.n_out) ** 0.5
        self.weight_out1.data.uniform_(-val_range, val_range)
        self.weight_out2.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias_in.data.zero_()
        self.bias_out.data.zero_()
        highway_bias, selfloop_bias, n_out = self.highway_bias, self.selfloop_bias, self.n_out
        if self.bidirectional:
            if self.use_highway:
                self.bias_in.data[6*2*n_out:7*2*n_out].zero_().add_(highway_bias)
            self.bias_in.data[5*2*n_out:6*2*n_out].zero_().add_(selfloop_bias)
        else:
            if self.use_highway:
                self.bias_in.data[6*n_out:7*n_out].zero_().add_(highway_bias)
            self.bias_in.data[5*n_out:6*n_out].zero_().add_(selfloop_bias)

        self.scale_x = 1
        if not rescale:
            return
        self.scale_x = (1 + math.exp(highway_bias) * 2) ** 0.5

        # re-scale weights in case there's dropout and / or layer normalization
        w_in = self.weight_in.data.view(self.n_in, -1, self.n_out, self.k)
        w_out1 = self.weight_out1.data
        w_out2 = self.weight_out2.data
        if self.dropout > 0:
            w_in[:, :, :, 0].mul_((1 - self.dropout) ** 0.5)
            w_in[:, :, :, 1].mul_((1 - self.dropout) ** 0.5)
            w_in[:, :, :, 2].mul_((1 - self.dropout) ** 0.5)
            w_out1.mul_((1 - self.dropout) ** 0.5)
            w_out2.mul_((1 - self.dropout) ** 0.5)
        if self.rnn_dropout > 0:
            w_in.mul_((1 - self.rnn_dropout) ** 0.5)
        if self.layer_norm:
            for i in range(3, self.k):
                w_in[:, :, :, i].mul_(0.1)

        if self.use_highway and self.n_in != self.n_out:
            w_in[:, :, :, -1].mul_(self.scale_x)
        # re-parameterize when weight normalization is enabled
        if self.weight_norm:
            self.init_weight_norm()

    def init_weight_norm(self):
        weight_in = self.weight_in.data
        g = weight_in.norm(2, 0)
        self.gain_in = nn.Parameter(g)

    def apply_weight_norm(self, eps=0):
        wnorm = self.weight_in.norm(2, 0)#, keepdim=True)
        return self.gain.expand_as(self.weight).mul(
            self.weight_in / (wnorm.expand_as(self.weight_in) + eps)
        )

    def set_bias(self, args):
        sys.stderr.write("\nWARNING: set_bias() is deprecated. use `highway_bias` option"
            " in SOPACell() constructor.\n"
        )
        self.highway_bias = args.highway_bias
        self.selfloop_bias = args.selfloop_bias
        self.init_weight()
        # n_out = self.n_out
        # if self.bidirectional:
        #     self.bias.data[n_out*2:].zero_().add_(bias_val)
        # else:
        #     self.bias.data[n_out:].zero_().add_(bias_val)


    def calc_activation(self, x):
        if self.activation_type == 0:
            return x
        elif self.activation_type == 1:
            return x.tanh()
        elif self.activation_type == 2:
            return nn.functional.relu(x)
        else:
            assert False, 'Activation type must be 0, 1, or 2, not {}'.format(self.activation_type)

    def forward(self, input, init_hidden=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        bidir = self.bidir
        length = input.size(0)
        if init_hidden is None:
            c1_init = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2).zero_())
            c2_init = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2).zero_())
            d_init = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2).zero_())
        else:
            assert (len(init_hidden) == 3)
            c1_init, c2_init, d_init = init_hidden

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((1, batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        if self.window_size == 1:
            x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        elif self.window_size == 2:
            Xm1 = []
            Xm1.append(x[:1, :, :] * 0)
            if len(x) > 1:
                Xm1.append(x[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([x, Xm1], 2)
            x_2d = source if source.dim() == 2 else source.contiguous().view(-1, n_in * 2)
        else:
            assert False    # yes they only use up to 2.

        weight_in = self.weight_in if not self.weight_norm else self.apply_weight_norm()
        u_ = x_2d.mm(weight_in)
        u_ = u_.view(length, batch, bidir, n_out, self.k)

        # some of the bias params here are never used. but i'm keeping them now
        # in case we need to bring them back in the future

        bias = self.bias_in.view(self.n_bias, bidir, n_out)
        # basic: in1, in2, in3, f1, f2, s1
        # optional: reset, output, highway affine.
        # we do not use bias in highway affine

        _, _, _, forget_bias1, forget_bias2, selfloop_bias = bias[:6, ...]
        if self.use_highway:
            reset_bias = bias[6, ...]
            reset = (u_[..., 6] + reset_bias).sigmoid()
            if self.use_output_gate:
                output_bias = bias[7, ...]
                output = (u_[..., 7] + output_bias).sigmoid()
        elif self.use_output_gate:
            output_bias = bias[6, ...]
            output = (u_[..., 6] + output_bias).sigmoid()

        u = Variable(u_.data.new(length, batch, bidir, n_out, 6))

        u[..., 0] = u_[..., 0]  # input 1
        u[..., 1] = u_[..., 1]  # input 2
        u[..., 2] = u_[..., 2]  # input 3

        u[..., 3] = (u_[..., 3] + forget_bias1).sigmoid()   # forget 1
        u[..., 4] = (u_[..., 4] + forget_bias2).sigmoid()   # forget 2
        u[..., 5] = (u_[..., 5] + selfloop_bias).sigmoid()  # selfloop

        # u.register_hook(print)
        # non_cuda_compatability: On a cuda machine
        if input.is_cuda:
            SOPA_Compute = SOPA_Compute_GPU(n_out, 6, self.bidirectional)
        else:
            SOPA_Compute = SOPA_Compute_CPU(n_out, 6, self.bidirectional)

        # non_cuda_compatability: On a non-cuda machine
        # SOPA_Compute = SOPA_Compute_CPU(n_out, 6, self.bidirectional)

        c1s, c2s, c1_final, c2_final, d_final = SOPA_Compute(u, c1_init, c2_init, d_init)

        mask_c1 = self.get_dropout_mask_((1, batch, bidir, n_out), self.dropout) \
                 if self.training and self.dropout > 0. else None
        mask_c1 = 1. if mask_c1 is None else mask_c1.expand_as(c1s)
        mask_c2 = self.get_dropout_mask_((1, batch, bidir, n_out), self.dropout) \
            if self.training and self.dropout > 0. else None
        mask_c2 = 1. if mask_c2 is None else mask_c2.expand_as(c2s)
        cs = self.bias_out + self.coef * (c1s * mask_c1).view(-1, bidir*n_out).mm(self.weight_out1) \
                + (1. - self.coef) * (c2s * mask_c2).view(-1, bidir*n_out).mm(self.weight_out2)
        
        if self.use_output_gate:
            gcs = self.calc_activation(output*cs.view(length, batch, bidir, n_out))
        else:
            gcs = self.calc_activation(cs).view(length, batch, bidir, n_out)
        # mask_h = self.get_dropout_mask_((1, batch, bidir, n_out), self.dropout) \
        #         if self.training and self.dropout > 0. else None
        # mask_h = 1. if mask_h is None else mask_h.expand_as(gcs)
        if self.use_highway:
            if self.n_in == self.n_out:
                x_prime = input.view(length, batch, bidir, n_out)
                x_prime = x_prime * self.scale_x if self.scale_x != 1 else x_prime
            else:
                x_prime = u_[..., -1]
            # h = (gcs*mask_h-x_prime)*reset+x_prime
            h = (gcs - x_prime) * reset + x_prime
        else:
            # h = gcs * mask_h
            h = gcs
        return h.view(length, batch, -1), c1_final, c2_final, d_final


    def get_dropout_mask_(self, size, p, rescale=True):
        w = self.weight_in.data
        # prob = Variable(w.new(*size).zero_() + 1 - p).cpu()
        # mask = torch.bernoulli(prob).div_(1-p) if rescale else torch.bernoulli(prob)
        # if w.is_cuda:
        #     mask = mask.cuda()
        # return mask
        if rescale:
            return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))
        else:
            return Variable(w.new(*size).bernoulli_(1-p))


'''
Constructor function:
====================
Arguments:	

- input_size: ...
- hidden_size: ...
- num_layers: ...
- dropout: vertical drop probablity
- rnn_dropout: drop probablity for computing values inside each recurrent unit, e.g., gates
- bidirectional: ...
- use_x: at most one of them could be 1. set all to zero to disable vertical activations
- weight_norm: ...
- layer_norm: ...
- highway_bias: init bias value for bias term in reset gates. useful only when using highway connections
- selfloop_bias: init bias value for bias term in selfloop gates. 
- coef: coefficients of unigram and bigram wfsas. they sum to one

forward function:
=================

Arguments:

- input (required): embeding vectors of the sequence, in [length, batch, dim] tensor
- init (None by default): init hidden states of the wfsas. useful when doing language modeling or seq2seq. 
if not specified, zero init states will be used. should you need to specify the init hidden states, 
the 'init_hidden' and 'repackage_hidden' functions in 'language_model/train_lm.py' give an example.
- return_hidden: ...

Returns:

output representations of the last layer, in [length, batch, (2*)dim] tensor.
last time step hidden states of wfsas, in 3-tuple, each of [layers, batch, (2*)dim] tensor.

'''

class SOPA(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=2,
                 dropout=0.,
                 rnn_dropout=0.,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 weight_norm=False,
                 layer_norm=False,
                 highway_bias=0,
                 selfloop_bias=0,
                 coef=.5,
                 use_highway=False,
                 use_output_gate=False,
                 window_size=1):
        super(SOPA, self).__init__()
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
        self.coef = coef
        self.use_highway = use_highway

        if use_tanh + use_relu + use_selu > 1:
            sys.stderr.write("\nWARNING: More than one activation enabled in SOPA"
                " (tanh: {}  relu: {}  selu: {})\n".format(use_tanh, use_relu, use_selu)
            )

        for i in range(num_layers):
            l = SOPACell(
                n_in=self.input_size if i == 0 else self.out_size,
                n_out=self.hidden_size,
                dropout=dropout if i+1 != num_layers else 0.,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                use_tanh=use_tanh,
                use_relu=use_relu,
                use_selu=use_selu,
                weight_norm=weight_norm,
                layer_norm=layer_norm,
                highway_bias=highway_bias,
                selfloop_bias=selfloop_bias,
                index=i+1,
                coef=coef,
                use_highway=use_highway,
                use_output_gate=use_output_gate,
                window_size=window_size
            )
            self.rnn_lst.append(l)
            if layer_norm:
                self.ln_lst.append(LayerNorm(self.hidden_size))

    def set_bias(self, args):
        for l in self.rnn_lst:
            l.set_bias(args)

    def forward(self, input, init_hidden=None, return_hidden=True):
        input, lengths = pad_packed_sequence(input, batch_first=True)
        assert input.dim() == 3 # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        batch = input.size(1)
        if init_hidden is None:
            init_hidden = [(
                Variable(input.data.new(batch, self.hidden_size * dir_).zero_()),
                Variable(input.data.new(batch, self.hidden_size * dir_).zero_()),
                Variable(input.data.new(batch, self.hidden_size * dir_).zero_()),
            ) for i in range(self.num_layers)]
            # assert False, "Please get the initial hidden states using init_hidden method."
        else:
            for c in init_hidden:
                assert c.dim() == 3
            init_hidden = [(c1.squeeze(0), c2.squeeze(0), d.squeeze(0))
                           for c1,c2,d in zip(
                        init_hidden[0].chunk(self.num_layers, 0),
                        init_hidden[1].chunk(self.num_layers, 0),
                        init_hidden[2].chunk(self.num_layers, 0)
                    )]

        prevx = input
        lstc1, lstc2, lstd = [], [], []
        for i, rnn in enumerate(self.rnn_lst):
            h, c1, c2, d = rnn(prevx, init_hidden[i])
            prevx = self.ln_lst[i](h) if self.use_layer_norm else h
            lstc1.append(c1)
            lstc2.append(c2)
            lstd.append(d)

        prevx = pack_padded_sequence(prevx, lengths, batch_first=True)
        if return_hidden:
            return prevx, (torch.stack(lstc1), torch.stack(lstc2), torch.stack(lstd))
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

