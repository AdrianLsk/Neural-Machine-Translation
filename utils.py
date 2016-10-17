from __future__ import print_function
from collections import OrderedDict

import theano.tensor as T

import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers.shape import ReshapeLayer, SliceLayer
from lasagne.layers.merge import concat, ElemwiseSumLayer
from lasagne.layers.recurrent import Gate, LSTMLayer
from lasagne.nonlinearities import tanh, linear
from lasagne.init import Orthogonal, Constant, GlorotUniform

from condGRUatt import GRULayer

input = GlorotUniform()
inner = Orthogonal(1.1)

def tanh_add(x, y):
    return tanh(T.add(x, y))


def mean_over_1_axis(x):
    return x.mean(1)


def softmax(vec, axis=-1):
    """
     The ND implementation of softmax nonlinearity applied over a specified
     axis, which is by default the second dimension.
    """
    xdev = vec - vec.max(axis, keepdims=True)
    rval = T.exp(xdev)/(T.exp(xdev).sum(axis, keepdims=True))
    return rval


def get_output_unit(input, incoming, dim_word):
    net = OrderedDict()
    # regress on combined projections
    net['output'] = DenseLayer(
        incoming, num_units=dim_word, nonlinearity=softmax, name='output'
    )

    # reshape back
    # n_batch, n_timesteps, n_features = input.input_var.shape
    n_batch = input.input_var.shape[0]
    net['reshape_output'] = ReshapeLayer(
        net.values()[-1], (n_batch, -1, n_targets),
        name='reshape_output'
    )

    return net


def get_rnn_unit(l_in, mask, rev_mask, state, rev_state, n_units, prefix,
                 grad_clip=0, context=None, attention=False):

    net = OrderedDict()
    hid = state
    rg = Gate(W_in=input, W_hid=inner, W_cell=None)
    ug = Gate(W_in=input, W_hid=inner, W_cell=None)
    hg = Gate(W_in=input, W_hid=inner, W_cell=None, nonlinearity=tanh)

    net[prefix+'gru'] = GRULayer(
        l_in, num_units=n_units, resetgate=rg, updategate=ug, hidden_update=hg,
        mask_input=mask, hid_init=hid, learn_init=False, only_return_final=False,
        grad_clipping=grad_clip, context_input=context, use_attention=attention,
        name='gru')

    if rev_mask is not None and rev_state is not None:
        net[prefix+'gru_rev'] = GRULayer(
            l_in, num_units=n_units, resetgate=rg, updategate=ug, hidden_update=hg,
            mask_input=rev_mask, hid_init=rev_state, only_return_final=False,
            learn_init=False, grad_clipping=grad_clip, context_input=context,
            backwards=True, name='gru_rev')

        net['context'] = ElemwiseSumLayer(net.values()[-2:], name='context')

    return net