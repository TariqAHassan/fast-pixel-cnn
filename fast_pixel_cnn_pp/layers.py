"""

    Layers
    ~~~~~~

"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from fast_pixel_cnn_pp.nn import get_name, get_vars_maybe_avg, int_shape, concat_elu, get_var_maybe_avg


@add_arg_scope
def dense(x, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    """ Fully connected layer """
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', [int(x.get_shape()[1]), num_units],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(x, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [1, num_units]) * (x_init - tf.reshape(m_init, [1, num_units]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            # V,g,b = get_vars_maybe_avg(['V','g','b'], ema)
            V = tf.get_variable('V', [int(x.get_shape()[1]), num_units], tf.float32)
            g = tf.get_variable('g', [num_units], tf.float32)
            b = tf.get_variable('b', [num_units], tf.float32)
            if ema is not None:
                V, g, b = ema.average(V), ema.average(g), ema.average(b)
            # tf.assert_variables_initialized([V,g,b])

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', nonlinearity=None,
           init_scale=1., counters={}, init=False, ema=None, **kwargs):
    """Convolutional layer """
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [int(x.get_shape()[-1]), num_filters], tf.float32,
                                tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(x, V_norm, [1] + stride + [1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (
                x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME',
             nonlinearity=None, init_scale=1., counters={},
             init=False, ema=None, **kwargs):
    """ Transposed convolutional layer """
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0], xs[2] * stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + filter_size[0] - 1, xs[2] * stride[1] + filter_size[1] - 1,
                        num_filters]
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [num_filters, int(x.get_shape()[-1])],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 3])
            x_init = tf.nn.conv2d_transpose(x, V_norm, target_shape, [1] + stride + [1], padding=pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (
                x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

            # calculate convolutional layer output
            x = tf.nn.conv2d_transpose(x, W, target_shape, [1] + stride + [1], padding=pad)
            x = tf.nn.bias_add(x, b)

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def nin(x, num_units, **kwargs):
    """ A network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])


# -------------------------------------------------------------------------------------------------
# Meta-layer consisting of multiple base layers
# -------------------------------------------------------------------------------------------------


@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu,
                 conv=conv2d, init=False, counters={}, ema=None,
                 dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_filters, counters=counters, init=init)
    if a is not None:  # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters, counters=counters, init=init)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    c2 = conv(c1, num_filters * 2, init_scale=0.1, counters=counters, init=init)

    # add projection of h vector if included: conditional generation
    if h is not None:
        with tf.variable_scope(get_name('conditional_weights', counters)):
            hw = get_var_maybe_avg('hw', ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if init:
            hw = hw.initialized_value()
        c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])

    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    return x + c3
