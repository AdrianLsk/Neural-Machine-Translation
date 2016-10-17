from __future__ import print_function
import sys
sys.path.append('..')

from collections import OrderedDict

from lasagne.layers import InputLayer,  DenseLayer, EmbeddingLayer
from lasagne.layers.shape import DimshuffleLayer, SliceLayer, ReshapeLayer
from lasagne.layers.merge import ElemwiseMergeLayer, concat
from lasagne.layers.special import ExpressionLayer
from lasagne.layers.noise import DropoutLayer
from lasagne.nonlinearities import linear, tanh
from lasagne.init import GlorotUniform, Constant
from utils import get_rnn_unit, get_output_unit, tanh_add, mean_over_1_axis


def build_nmt_encoder_decoder(dim_word=1, n_embd=100, n_units=500, n_proj=200,
                              state=None, rev_state=None, context_type=None,
                              attention=False, drop_p=None):
    enc = OrderedDict()
    enc['input'] = InputLayer((None, None), name='input')
    enc_mask = enc['mask'] = InputLayer((None, None), name='mask')
    enc_rev_mask = enc['rev_mask'] = InputLayer((None, None), name='rev_mask')

    enc['input_emb'] = EmbeddingLayer(
        enc.values()[-1], input_size=dim_word, output_size=n_embd,
        name='input_emb'
    )

    ### ENCODER PART ###
    # rnn encoder unit
    hid_init = Constant(0.)
    hid_init_rev = Constant(0.)
    encoder_unit = get_rnn_unit(
        enc.values()[-1], enc_mask, enc_rev_mask, hid_init, hid_init_rev,
        n_units, prefix='encoder_'
    )
    enc.update(encoder_unit)

    # context layer = decoder's initial state of shape (batch_size, num_units)
    context = enc.values()[-1] # net['context']
    if context_type == 'last':
        enc['context2init'] = SliceLayer(
            context, indices=-1, axis=1, name='last_encoder_context'
        )
    elif context_type == 'mean':
        enc['context2init'] = ExpressionLayer(
            context, mean_over_1_axis, output_shape='auto',
            name='mean_encoder_context'
        )

    ### DECODER PART ###
    W_init2proj, b_init2proj = GlorotUniform(), Constant(0.)

    enc['init_state'] = DenseLayer(
        enc['context2init'], num_units=n_units, W=W_init2proj, b=b_init2proj,
        nonlinearity=tanh, name='decoder_init_state'
        )
    if state is None:
        init_state = enc['init_state']
        init_state_rev = None #if rev_state is None else init_state
        if not attention:
            # if simple attetion the context is 2D, else 3D
            context = enc['context2init']
    else:
        init_state = state
        init_state_rev = rev_state
        context = enc['context_input'] = \
            InputLayer((None, n_units), name='ctx_input')
    # (batch_size, nfeats)

    # (batch_size, valid ntsteps)
    enc['target'] = InputLayer((None, None), name='target')
    dec_mask = enc['target_mask'] = InputLayer((None, None), name='target_mask')

    enc['target_emb'] = EmbeddingLayer(
        enc.values()[-1], input_size=dim_word, output_size=n_embd,
        name='target_emb'
    )
    prevdim = n_embd
    prev2rnn = enc.values()[-1] # it's either emb or prev2rnn/noise

    decoder_unit = get_rnn_unit(
        prev2rnn, dec_mask, None, init_state, None, n_units, prefix='decoder_',
        context=context, attention=attention
    )
    enc.update(decoder_unit)

    if attention:
        ctxs = enc.values()[-1]
        ctxs_shape = ctxs.output_shape
        def get_ctx(x):
            return ctxs.ctx
        context = enc['context'] = ExpressionLayer(
            ctxs, function=get_ctx, output_shape=ctxs_shape,
            name='context'
        )

    # return all values'
    # reshape for feed-forward layer
    # 2D shapes of (batch_size * num_steps, num_units/num_feats)
    enc['rnn2proj'] = rnn2proj = ReshapeLayer(
        enc.values()[-1], (-1, n_units), name='flatten_rnn2proj'
    )

    enc['prev2proj'] = prev2proj = ReshapeLayer(
        prev2rnn, (-1, prevdim), name='flatten_prev'
    )

    if isinstance(context, ExpressionLayer):
        ctx2proj = enc['ctx2proj'] = ReshapeLayer(
            context, (-1, ctxs_shape[-1]), name='flatten_ctxs'
            )
    else:
        ctx2proj = context

    # load shared parameters
    W_rnn2proj, b_rnn2proj = GlorotUniform(), Constant(0.)
    W_prev2proj, b_prev2proj = GlorotUniform(), Constant(0.)
    W_ctx2proj, b_ctx2proj= GlorotUniform(), Constant(0.)

    # perturb rnn-to-projection by noise
    if drop_p is not None:
        rnn2proj = enc['noise_rnn2proj'] = DropoutLayer(
            rnn2proj, sigma=drop_p, name='noise_rnn2proj'
        )

        prev2proj = enc['drop_prev2proj'] = DropoutLayer(
            prev2proj, sigma=drop_p, name='drop_prev2proj'
        )

        ctx2proj = enc['noise_ctx2proj'] = DropoutLayer(
            ctx2proj, sigma=drop_p, name='noise_ctx2proj'
        )

    # project rnn
    enc['rnn_proj'] = DenseLayer(
        rnn2proj, num_units=n_proj, nonlinearity=linear, W=W_rnn2proj,
        b=b_rnn2proj, name='rnn_proj'
    )

    # project raw targets
    enc['prev_proj'] = DenseLayer(
        prev2proj, num_units=n_proj, nonlinearity=linear,
        W=W_prev2proj, b=b_prev2proj, name='prev_proj'
    )

    # project context
    enc['ctx_proj'] = DenseLayer(
        ctx2proj, num_units=n_proj, nonlinearity=linear,
        W=W_ctx2proj, b=b_ctx2proj, name='ctx_proj'
    )

    # reshape back for merging
    n_batch = enc['input'].input_var.shape[0]
    rnn2merge = enc['rnn2merge'] = ReshapeLayer(
        enc['rnn_proj'], (n_batch, -1, n_proj), name='reshaped_rnn2proj'
    )

    prev2merge = enc['prev2merge'] = ReshapeLayer(
        enc['prev_proj'], (n_batch, -1, n_proj),
        name='reshaped_prev'
    )

    if isinstance(context, ExpressionLayer):
       ctx2merge = ReshapeLayer(
           enc['ctx_proj'], (n_batch, -1, n_proj), name='reshaped_prev'
           )
    else:
        ctx2merge = enc['ctx2merge'] = DimshuffleLayer(
            enc['ctx_proj'], pattern=(0, 'x', 1), name='reshaped_context'
        )

    # combine projections into shape (batch_size, n_steps, n_proj)
    enc['proj_merge'] = ElemwiseMergeLayer(
        [rnn2merge, prev2merge, ctx2merge], merge_function=tanh_add,
        name='proj_merge'
    )

    # reshape for output regression projection
    enc['merge2proj'] = ReshapeLayer(
        enc.values()[-1], (-1, n_proj), name='flatten_proj_merge'
    )

    # perturb concatenated regressors by noise
    if drop_p is not None:
        # if noise_type == 'binary':
        enc['noise_output'] = DropoutLayer(
            enc.values()[-1], p=drop_p, name='noise_output'
        )

    # regress on combined (perturbed) projections
    out = get_output_unit(
        enc['target'], enc.values()[-1], dim_word
    )
    enc.update(out) # update graph

    return enc