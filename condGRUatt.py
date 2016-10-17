from lasagne.layers import Layer
from lasagne.layers.merge import MergeLayer
from lasagne.layers.recurrent import Gate
from lasagne import nonlinearities
from lasagne import init
import numpy as np
import theano
import theano.tensor as T

class GRULayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    Gated Recurrent Unit (GRU) Layer
    Implements the recurrent step proposed in [1]_, which computes the output
    by
    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Cho, Kyunghyun, et al: On the properties of neural
       machine translation: Encoder-decoder approaches.
       arXiv preprint arXiv:1409.1259 (2014).
    .. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
       Recurrent Neural Networks on Sequence Modeling.
       arXiv preprint arXiv:1412.3555 (2014).
    .. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    Notes
    -----
    An alternate update for the candidate hidden state is proposed in [2]_:
    .. math::
        c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\
    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 context_input=None,
                 context_mask_input=None,
                 context_pars=None,
                 use_attention=False,
                 attention_pars=None,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.context_incoming_index = -1
        self.mask_incoming_index = -1
        self.context_mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if context_input is not None:
            incomings.append(context_input)
            self.context_incoming_index = len(incomings)-1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
            if context_mask_input is not None:
                incomings.append(context_mask_input)
                self.context_mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(GRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.use_attention = use_attention
        self.ctx = None
        self.alpha = None

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        if context_input is not None:
            # allocate weights for context to gates and context to interim state
            if context_pars is None:
                # context pars
                self.W_ctx_to_updategate = \
                    self.add_param(updategate.W_hid, (num_units, num_units),
                                   name="W_ctx_to_{}".format('updategate'))
                self.W_ctx_to_resetgate = \
                    self.add_param(resetgate.W_hid, (num_units, num_units),
                                   name="W_ctx_to_{}".format('resetgate'))
                self.W_ctx_to_hidden_update = \
                    self.add_param(hidden_update.W_hid, (num_units, num_units),
                                   name="W_ctx_to_{}".format('hidden_update'))
            else:
                self.W_ctx_to_updategate = context_pars[0]
                self.W_ctx_to_resetgate = context_pars[1]
                self.W_ctx_to_hidden_update = context_pars[2]

            if use_attention:
                if attention_pars is None:
                    attention = hidden_update
                    # context attention pars
                    self.W_ctx_to_att = \
                        self.add_param(attention.W_hid, (num_units, num_units),
                                       name="W_ctx_to_att")
                    self.b_ctx_to_att = \
                        self.add_param(attention.b, (num_units,),
                                       name="b_ctx_to_att", regularizable=False)
                    # attention pars
                    self.U_att = \
                        self.add_param(attention.W_hid, (num_units, 1),
                                       name="U_att")
                    self.c_att = \
                        self.add_param(attention.b, (1,), name="c_att",
                                       regularizable=False)

                    self.W_hid_to_att = \
                        self.add_param(attention.W_hid, (num_units, num_units),
                                       name="W_hid_to_att")
                    # self.b_hid_to_att = \
                    #     self.add_param(attention.b, (num_units,),
                    #                    name="b_hid_to_att", regularizable=False)

                    # hidden to hidden 2 pars
                    self.W_hid_to_updategate2 = \
                        self.add_param(updategate.W_hid, (num_units, num_units),
                                       name="W_hid_to_{}".format('updategate2'))
                    self.b_hid_to_updategate2 = \
                        self.add_param(updategate.b, (num_units,),
                                       name="b_hid_to_{}".format('updategate2'),
                                       regularizable=False)

                    self.W_hid_to_resetgate2 = \
                        self.add_param(resetgate.W_hid, (num_units, num_units),
                                       name="W_hid_to_{}".format('resetgate2'))
                    self.b_hid_to_resetgate2 = \
                        self.add_param(resetgate.b, (num_units,),
                                       name="b_hid_to_{}".format('resetgate2'),
                                       regularizable=False)

                    self.W_hid_to_hidden_update2 = \
                        self.add_param(hidden_update.W_hid, (num_units, num_units),
                                       name="W_hid_to_{}".format('hidden_update2'))
                    self.b_hid_to_hidden_update2 = \
                        self.add_param(hidden_update.b, (num_units,),
                                       name="b_hid_to_{}".format('hidden_update2'),
                                       regularizable=False)
                else:
                    self.W_ctx_to_att = attention_pars.pop(0)
                    self.b_ctx_to_att = attention_pars.pop(0)
                    self.W_hid_to_updategate2 = attention_pars.pop(0)
                    self.b_hid_to_updategate2 = attention_pars.pop(0)
                    self.W_hid_to_resetgate2 = attention_pars.pop(0)
                    self.b_hid_to_resetgate2 = attention_pars.pop(0)
                    self.W_hid_to_hidden_update2 = attention_pars.pop(0)
                    self.b_hid_to_hidden_update2 = attention_pars.pop(0)

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        context_mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
            if self.context_mask_incoming_index > 0:
                context_mask = inputs[self.context_mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.context_incoming_index > 0:
            ctx = inputs[self.context_incoming_index]
            W_ctx_stacked = T.concatenate(
                [self.W_ctx_to_resetgate, self.W_ctx_to_updategate,
                 self.W_ctx_to_hidden_update], axis=1)

            if self.use_attention:
                W_hid_stacked2 = T.concatenate(
                    [self.W_hid_to_att, self.W_hid_to_resetgate2,
                     self.W_hid_to_updategate2, self.W_hid_to_hidden_update2],
                    axis=1)

                b_stacked2 = T.concatenate(
                    [T.zeros_like(self.b_hid_to_resetgate2),
                     self.b_hid_to_resetgate2, self.b_hid_to_updategate2,
                     self.b_hid_to_hidden_update2],
                    axis=0)

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # projection
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked
            if self.context_incoming_index > 0:
                if self.use_attention:
                    ctx = ctx.dimshuffle(1, 0, 2)
                    # (n_batch, 3*num_units) -> (n_batch, x, 3*num_units)
                    ctx_n = T.dot(ctx, self.W_ctx_to_att) + self.b_ctx_to_att
                else:
                    # (n_batch, 3*num_units) -> (x, n_batch, 3*num_units)
                    ctx = T.dot(ctx, W_ctx_stacked).dimshuffle('x', 0, 1)
                    input += ctx

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if self.context_incoming_index > 0:
                ctx = args[-1]

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
                if not self.use_attention and self.context_incoming_index > 0:
                    input_n += ctx

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update

            if self.use_attention:
                if not self.precompute_input:
                    # (n_batch, 3*num_units) -> (n_batch, x, 3*num_units)
                    ctx_n = T.dot(ctx, self.W_ctx_to_att) + self.b_ctx_to_att
                else:
                    ctx_n = args[-2]

                # attention
                # W_hid_stacked2 = [W_hid_att, U_rg, U_ug, U_hu]
                # b_stacked2 = [zeros_att, b_rg2, b_ug2, b_hu2]
                # hid_n = T.dot(hid, W_hid_att)
                hid_n = T.dot(hid, W_hid_stacked2) + b_stacked2

                ctx_n += slice_w(hid_n, 0)[None, :, :] # pctx_ + pstate_
                ctx_n = T.tanh(ctx_n)

                alpha = T.dot(ctx_n, U_att) + c_att
                # (n_steps, bs, n_units).(n_units, 1) + (1,) = (n_steps, bs, 1)
                alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
                # (n_steps, bs)
                alpha = T.exp(alpha)
                if context_mask is not None:
                    alpha = alpha * context_mask
                alpha = alpha / alpha.sum(0, keepdims=True) # (n_steps, bs)
                ctx_n = (ctx * alpha[:, :, None]).sum(0)  # current context
                # (n_steps, bs, n_units)*(n_steps, bs, 1) -(sum)> (bs, n_units)

                # hid_hid = T.dot(hid, W_hid_stacked2) + b_stacked2
                # W_ctx_stacked2 = [W_ctx_rg2, W_ctx_ug2, W_ctx_hu2] # [Wc, Wcx]
                pctx_n = T.dot(ctx_n, W_ctx_stacked)

                # Reset and update gates
                resetgate = slice_w(hid_n, 1) + slice_w(pctx_n, 0)
                updategate = slice_w(hid_n, 2) + slice_w(pctx_n, 1)
                resetgate = self.nonlinearity_resetgate(resetgate)
                updategate = self.nonlinearity_updategate(updategate)

                # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
                hidden_update_ctx = slice_w(pctx_n, 2)
                hidden_update_hid = slice_w(hid_n, 3)
                hidden_update = hidden_update_ctx + resetgate*hidden_update_hid
                if self.grad_clipping:
                    hidden_update = theano.gradient.grad_clip(
                        hidden_update, -self.grad_clipping, self.grad_clipping)
                hidden_update = self.nonlinearity_hid(hidden_update)

                # Compute (1 - u_t)h_{t - 1} + u_t c_t
                hid = (1 - updategate)*hid + updategate*hidden_update

                return hid, ctx_n, alpha.T
                # (bs, n_units), (bs, n_units), (bs, n_steps_source)
                # prepending (n_steps_target,) for the output of the scan
            else:
                return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            if self.use_attention:
                hid = step(input_n, hid_previous, *args)
            else:
                hid, ctx, alpha = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            if self.use_attention:
                return hid
            else:
                return hid, ctx, alpha

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        out_info = [hid_init]

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.context_incoming_index > 0:
            if self.use_attention and self.precompute_input:
                out_info += [None, None]
                U_att, c_att = self.U_att, self.c_att
                non_seqs += [W_ctx_stacked, W_hid_stacked2, b_stacked2, U_att,
                             c_att, ctx_n, ctx]
            else:
                non_seqs += [ctx]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=out_info,
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=out_info,
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.use_attention:
            hid_out, ctx, alpha = hid_out
            self.ctx = ctx.dimshuffle(1, 0, 2)
            self.alpha = alpha.dimshuffle(1, 0, 2)
            # (bs, n_steps_target, n_steps_source)

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out