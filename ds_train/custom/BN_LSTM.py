from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from tensorflow.python.ops.rnn_cell_impl import _BIAS_VARIABLE_NAME, _WEIGHTS_VARIABLE_NAME, LayerRNNCell, LSTMStateTuple


class BNLSTMCell(LayerRNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    http://www.bioinf.jku.at/publications/older/2604.pdf
  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.

  The recurrent batch normalization is based on:

    https://arxiv.org/pdf/1603.09025.pdf

  """

    def __init__(self,
                 num_units,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 proj_clip=None,
                 num_unit_shards=None,
                 num_proj_shards=None,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 normalize_in_to_hidden=False,
                 normalize_in_together=True,
                 normalize_cell=False,
                 normalize_config=None,
                 name=None):
        """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      normalize_in_to_hidden: If True, inputs and state will be normalized.
      normalize_in_together: Only has an effect if normalize_in_to_hidden is True.
        If True, both inputs and state will be normalized together, with only
        one beta and gamma shared between the two. If False, each will receive
        their own beta and gamma, but this will result in the inputs and state
        being multiplied with the weights separately, instead of together.
      normalize_cell: If True, cell will be normalized.
      norm_config: Dictionary to pass as parameters to layers.batch_normalization.
        If None, then the default batch_normalization configuration is used,
        except that no beta is used, and the gamma initializer is a constant
        initializer set to 0.1, as per the referenced paper
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      When restoring from CudnnLSTM-trained checkpoints, use
      `CudnnCompatibleLSTMCell` instead.
    """
        super(BNLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._normalize_in_to_hidden = normalize_in_to_hidden
        self._normalize_in_together = normalize_in_to_hidden and normalize_in_together
        self._normalize_cell = normalize_cell
        self._normalize_config = normalize_config

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj)
                                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units)
                                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s" %
                inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (partitioned_variables.fixed_size_partitioner(
            self._num_unit_shards)
                             if self._num_unit_shards is not None else None)

        if self._normalize_in_to_hidden or self._normalize_cell:
            if self._normalize_config is None:
                #Default normalization configuration
                #See https://arxiv.org/pdf/1603.09025.pdf for reason for gamma_initializer
                self._normalize_config = {
                    'center':
                    False,
                    'scale':
                    True,
                    'gamma_initializer':
                    init_ops.constant_initializer(0.1, dtype=self.dtype)
                }
            else:
                self._normalize_config['center'] = False

        if not self._normalize_in_to_hidden or self._normalize_in_together:
            self._kernel = self.add_variable(
                _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + h_depth, 4 * self._num_units],
                initializer=self._initializer,
                partitioner=maybe_partitioner)
            if self._normalize_in_to_hidden:
                self._bn = BatchNormalization(**self._normalize_config)
        else:
            self._kernel_m = self.add_variable(
                "i_scope/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth, 4 * self._num_units],
                initializer=self._initializer,
                partitioner=maybe_partitioner)
            with vs.variable_scope(None, "i_scope"):
                self._bn_i = BatchNormalization(**self._normalize_config)

            self._kernel_m = self.add_variable(
                "m_scope/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[h_depth, 4 * self._num_units],
                initializer=self._initializer,
                partitioner=maybe_partitioner)
            with vs.variable_scope(None, "m_scope"):
                self._bn_m = BatchNormalization(**self._normalize_config)

        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        if self._normalize_cell:
            self._normalize_config_cell = self._normalize_config
            self._normalize_config_cell['center'] = True
            self._bn_c = BatchNormalization(**self._normalize_config_cell)

        if self._use_peepholes:
            self._w_f_diag = self.add_variable(
                "w_f_diag",
                shape=[self._num_units],
                initializer=self._initializer)
            self._w_i_diag = self.add_variable(
                "w_i_diag",
                shape=[self._num_units],
                initializer=self._initializer)
            self._w_o_diag = self.add_variable(
                "w_o_diag",
                shape=[self._num_units],
                initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                partitioned_variables.fixed_size_partitioner(
                    self._num_proj_shards)
                if self._num_proj_shards is not None else None)
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)

        self.built = True

    def call(self, inputs, state, training=False):
        """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, `[batch, num_units].
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
      training: if batch normalization is activated, then this parameter will
        be passed to it when called
    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units],
                                     [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        if not self._normalize_in_to_hidden or self._normalize_in_together:
            lstm_matrix = math_ops.matmul(
                array_ops.concat([inputs, m_prev], 1), self._kernel)
            if self._normalize_in_to_hidden:
                lstm_matrix = self._bn(lstm_matrix, training=training)
        else:
            op_i = math_ops.matmul(inputs, self._kernel_i)
            op_m = math_ops.matmul(m_prev, self._kernel_m)
            lstm_matrix = self._bn_i(op_i, training=training)
            lstm_matrix += self._bn_m(op_m, training=training)

        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) *
                 c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev +
                 sigmoid(i) * self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type

        if not self._normalize_cell:
            c_new = c
        else:
            c_new = self._bn_c(c, training=training)

        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c_new) * self._activation(c_new)
        else:
            m = sigmoid(o) * self._activation(c_new)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip,
                                           self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m)
                     if self._state_is_tuple else array_ops.concat([c, m], 1))
        return m, new_state
