from builtins import range
from builtins import object

import tensorflow as tf
import numpy as np

class TensorParameter(object):
    """Custom wrapper around a Tensor."""
    def __init__(self, name, shape, dtype=tf.float32, old_ema_rate=False):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.old_ema_rate = old_ema_rate
        self.keep_old = old_ema_rate is not False

        self.internal_tensor = tf.get_variable(name, shape, dtype, initializer=tf.initializers.zeros())
        self.params_list = [self.internal_tensor]
        self.trainable_params_list = [self.internal_tensor]
        if self.keep_old:
            self.old_internal_tensor = tf.get_variable(name+"_old", shape, dtype, trainable=False)
            self.old_init_ops = [tf.assign(self.old_internal_tensor, self.internal_tensor)]
            self.old_ema_ops = [tf.assign(self.old_internal_tensor, (1. - self.old_ema_rate) * self.old_internal_tensor + self.old_ema_rate * self.internal_tensor)]
            self.params_list.append(self.old_internal_tensor)

    def __call__(self, use_old=False):
        if use_old:
            assert self.keep_old
            return self.old_internal_tensor
        else:
            return self.internal_tensor

class FeedForwardNet(object):
    """Custom feed-forward network layers."""
    def __init__(self, name, in_size, out_shape, layers=1, hidden_size=32, internal_nonlinearity=None, final_nonlinearity=None, old_ema_rate=False):
        self.name = name
        self.in_size = in_size
        self.out_shape = out_shape
        self.out_size = int(np.prod(out_shape))
        self.layers = layers
        self.hidden_dim = hidden_size
        self.internal_nonlinearity = tf.nn.relu if internal_nonlinearity is None else internal_nonlinearity
        self.final_nonlinearity = (lambda x:x) if final_nonlinearity is None else final_nonlinearity
        self.old_ema_rate = old_ema_rate
        self.keep_old = old_ema_rate is not False

        self.weights = [None] * layers
        self.biases = [None] * layers
        if self.keep_old:
            self.old_weights = [None] * layers
            self.old_biases = [None] * layers
            self.old_init_ops = []
            self.old_ema_ops = []

        self.params_list = []
        self.trainable_params_list = []

        with tf.variable_scope(name):
            # main trainable parameters of ensemble
            for layer_i in range(self.layers):
                in_size = self.hidden_dim
                out_size = self.hidden_dim
                if layer_i == 0: in_size = self.in_size
                if layer_i == self.layers - 1: out_size = self.out_size
                self.weights[layer_i] = tf.get_variable("weights%d" % layer_i, [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
                self.biases[layer_i] = tf.get_variable("bias%d" % layer_i, [1, out_size], initializer=tf.initializers.zeros())
                self.params_list += [self.weights[layer_i], self.biases[layer_i]]
                self.trainable_params_list += [self.weights[layer_i], self.biases[layer_i]]

            # old backups of the parameters, as estimated via EMA
            if self.keep_old:
                for layer_i in range(self.layers):
                    in_size = self.hidden_dim
                    out_size = self.hidden_dim
                    if layer_i == 0: in_size = self.in_size
                    if layer_i == self.layers - 1: out_size = self.out_size
                    self.old_weights[layer_i] = tf.get_variable("old_weights%d" % layer_i, [in_size, out_size], trainable=False)
                    self.old_biases[layer_i] = tf.get_variable("old_bias%d" % layer_i, [1, out_size], trainable=False)
                    self.params_list += [self.old_weights[layer_i], self.old_biases[layer_i]]
                    self.old_init_ops.append(tf.assign(self.old_weights[layer_i], self.weights[layer_i]))
                    self.old_init_ops.append(tf.assign(self.old_biases[layer_i], self.biases[layer_i]))
                    self.old_ema_ops.append(tf.assign(self.old_weights[layer_i], (1. - self.old_ema_rate) * self.old_weights[layer_i] + self.old_ema_rate * self.weights[layer_i]))
                    self.old_ema_ops.append(tf.assign(self.old_biases[layer_i], (1. - self.old_ema_rate) * self.old_biases[layer_i] + self.old_ema_rate * self.biases[layer_i]))

    def __call__(self, x, use_old=False, stop_params_gradient=False):
        # decide whether to use old weights
        if use_old:
            assert self.keep_old
            weights = [tf.stop_gradient(weight) for weight in self.old_weights]
            biases = [tf.stop_gradient(bias) for bias in self.old_biases]
        elif stop_params_gradient:
            weights = [tf.stop_gradient(weight) for weight in self.weights]
            biases = [tf.stop_gradient(bias) for bias in self.biases]
        else:
            weights = self.weights
            biases = self.biases

        # reshape input
        batch_shape = tf.shape(x)[:-1]
        assert x.shape.as_list()[-1] == self.in_size
        x = tf.reshape(x, [-1, self.in_size])

        # main network
        h = x
        for layer_i in range(self.layers):
            if layer_i + 1 < self.layers:
                h = self.internal_nonlinearity(tf.matmul(h, weights[layer_i]) + biases[layer_i])
            else:
                h = tf.matmul(h, weights[layer_i]) + biases[layer_i]

        # reshape to original batching
        h = tf.reshape(h, tf.concat([batch_shape, tf.constant(self.out_shape, dtype=tf.int32)], axis=0))
        h = self.final_nonlinearity(h)

        return h

    def l2(self):
        return tf.add_n([tf.reduce_sum(tf.square(param)) for param in self.trainable_params_list])

    def clipping(self):
        return [tf.assign(param, tf.clip_by_value(param, -1., 1.)) for param in self.trainable_params_list]

class ConvNet(object):
    """Custom convolutional network layers."""
    def __init__(self, name, in_shape, out_shape, layers, internal_nonlinearity=None, final_nonlinearity=None, old_ema_rate=False):
        self.name = name
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.out_size = int(np.prod(out_shape))
        self.n_layers = len(layers)
        self.layers = layers
        self.internal_nonlinearity = tf.nn.leaky_relu if internal_nonlinearity is None else internal_nonlinearity
        self.final_nonlinearity = (lambda x:x) if final_nonlinearity is None else final_nonlinearity
        self.old_ema_rate = old_ema_rate
        self.keep_old = old_ema_rate is not False

        self.weights = [None] * self.n_layers
        self.biases = [None] * self.n_layers
        if self.keep_old:
            self.old_weights = [None] * self.n_layers
            self.old_biases = [None] * self.n_layers
            self.old_init_ops = []
            self.old_ema_ops = []

        self.params_list = []
        self.trainable_params_list = []

        width, height, depth = self.in_shape
        with tf.variable_scope(name):
            # main trainable parameters of ensemble
            for layer_i in range(self.n_layers):
                in_depth = depth
                filter_shape, stride, out_depth = self.layers[layer_i]
                self.weights[layer_i] = tf.get_variable("weights%d" % layer_i, filter_shape + [in_depth, out_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
                self.biases[layer_i] = tf.get_variable("bias%d" % layer_i, [1, 1, 1, out_depth], initializer=tf.initializers.zeros())
                self.params_list += [self.weights[layer_i], self.biases[layer_i]]
                self.trainable_params_list += [self.weights[layer_i], self.biases[layer_i]]
                width /= stride[0]
                height /= stride[1]
                depth = out_depth
            self.final_weight = tf.get_variable("final_weight", [width * height * depth, self.out_size], initializer=tf.contrib.layers.xavier_initializer())
            self.final_bias = tf.get_variable("final_bias", [1, self.out_size], initializer=tf.initializers.zeros())
            self.params_list += [self.final_weight, self.final_bias]
            self.trainable_params_list += [self.final_weight, self.final_bias]

            # old backups of the parameters, as estimated via EMA
            if self.keep_old:
                width, height, depth = self.in_shape
                for layer_i in range(self.n_layers):
                    in_depth = depth
                    filter_shape, stride, out_depth = self.layers[layer_i]
                    self.old_weights[layer_i] = tf.get_variable("old_weights%d" % layer_i, filter_shape + [in_depth, out_depth], trainable=False)
                    self.old_biases[layer_i] = tf.get_variable("old_bias%d" % layer_i, [1, 1, 1, out_depth], trainable=False)
                    self.params_list += [self.old_weights[layer_i], self.old_biases[layer_i]]
                    self.old_init_ops.append(tf.assign(self.old_weights[layer_i], self.weights[layer_i]))
                    self.old_init_ops.append(tf.assign(self.old_biases[layer_i], self.biases[layer_i]))
                    self.old_ema_ops.append(tf.assign(self.old_weights[layer_i], (1. - self.old_ema_rate) * self.old_weights[layer_i] + self.old_ema_rate * self.weights[layer_i]))
                    self.old_ema_ops.append(tf.assign(self.old_biases[layer_i], (1. - self.old_ema_rate) * self.old_biases[layer_i] + self.old_ema_rate * self.biases[layer_i]))
                    width /= stride[0]
                    height /= stride[1]
                    depth = out_depth
                self.old_final_weight = tf.get_variable("old_final_weight", [width * height * depth, self.out_size], trainable=False)
                self.old_final_bias = tf.get_variable("old_final_bias", [1, self.out_size], trainable=False)
                self.params_list += [self.old_final_weight, self.old_final_bias]
                self.old_init_ops.append(tf.assign(self.old_final_weight, self.final_weight))
                self.old_init_ops.append(tf.assign(self.old_final_bias, self.final_bias))
                self.old_ema_ops.append(tf.assign(self.old_final_weight, (1. - self.old_ema_rate) * self.old_final_weight + self.old_ema_rate * self.final_weight))
                self.old_ema_ops.append(tf.assign(self.old_final_bias, (1. - self.old_ema_rate) * self.old_final_bias + self.old_ema_rate * self.final_bias))

    def __call__(self, x, use_old=False, stop_params_gradient=False, add_x_channel_dim=True):
        if add_x_channel_dim: x = tf.expand_dims(x, -1)

        # decide whether to use old weights
        if use_old:
            assert self.keep_old
            weights = [tf.stop_gradient(weight) for weight in self.old_weights]
            biases = [tf.stop_gradient(bias) for bias in self.old_biases]
            final_weight = tf.stop_gradient(self.old_final_weight)
            final_bias = tf.stop_gradient(self.old_final_bias)
        elif stop_params_gradient:
            weights = [tf.stop_gradient(weight) for weight in self.weights]
            biases = [tf.stop_gradient(bias) for bias in self.biases]
            final_weight = tf.stop_gradient(self.final_weight)
            final_bias = tf.stop_gradient(self.final_bias)
        else:
            weights = self.weights
            biases = self.biases
            final_weight = self.final_weight
            final_bias = self.final_bias

        # reshape input
        batch_shape = tf.shape(x)[:-3]
        assert x.shape.as_list()[-3:] == self.in_shape
        x = tf.reshape(x, [-1] + list(self.in_shape))

        # main network
        h = x
        width, height, depth = self.in_shape
        for layer_i in range(self.n_layers):
            filter_shape, stride, out_depth = self.layers[layer_i]
            h = self.internal_nonlinearity(tf.nn.convolution(h, weights[layer_i], 'SAME', strides=stride) + biases[layer_i])
            width /= stride[0]
            height /= stride[1]
            depth = out_depth
        h = tf.reshape(h, [-1, width * height * depth])
        h = tf.matmul(h, final_weight) + final_bias

        # reshape to original batching
        h = tf.reshape(h, tf.concat([batch_shape, tf.constant(self.out_shape, dtype=tf.int32)], axis=0))
        h = self.final_nonlinearity(h)

        return h

    def l2(self):
        return tf.add_n([tf.reduce_sum(tf.square(param)) for param in self.trainable_params_list])

    def clipping(self):
        return [tf.assign(param, tf.clip_by_value(param, -1., 1.)) for param in self.trainable_params_list]

class DeConvNet(object):
    """Custom convolutional network layers."""
    def __init__(self, name, in_size, out_shape, layers, internal_nonlinearity=None, final_nonlinearity=None, old_ema_rate=False):
        self.name = name
        self.in_size = in_size
        self.out_shape = out_shape
        self.out_size = int(np.prod(out_shape))
        self.n_layers = len(layers)
        self.layers = layers
        self.internal_nonlinearity = tf.nn.leaky_relu if internal_nonlinearity is None else internal_nonlinearity
        self.final_nonlinearity = (lambda x:x) if final_nonlinearity is None else final_nonlinearity
        self.old_ema_rate = old_ema_rate
        self.keep_old = old_ema_rate is not False

        self.weights = [None] * self.n_layers
        self.biases = [None] * self.n_layers
        if self.keep_old:
            self.old_weights = [None] * self.n_layers
            self.old_biases = [None] * self.n_layers
            self.old_init_ops = []
            self.old_ema_ops = []

        self.params_list = []
        self.trainable_params_list = []

        width = self.out_shape[0] / np.prod([stride[0] for _, stride, _ in self.layers])
        height = self.out_shape[1] / np.prod([stride[1] for _, stride, _ in self.layers])
        depth = self.layers[0][2]
        with tf.variable_scope(name):
            # main trainable parameters of ensemble
            self.initial_weight = tf.get_variable("initial_weight", [self.in_size, width * height * depth], initializer=tf.contrib.layers.xavier_initializer())
            self.initial_bias = tf.get_variable("initial_bias", [1, width * height * depth], initializer=tf.initializers.zeros())
            self.params_list += [self.initial_weight, self.initial_bias]
            self.trainable_params_list += [self.initial_weight, self.initial_bias]
            for layer_i in range(self.n_layers):
                filter_shape, stride, in_depth = self.layers[layer_i]
                out_depth = self.layers[layer_i+1][2] if layer_i+1 < self.n_layers else self.out_shape[2]
                self.weights[layer_i] = tf.get_variable("weights%d" % layer_i, filter_shape + [out_depth, in_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
                self.biases[layer_i] = tf.get_variable("bias%d" % layer_i, [1, 1, 1, out_depth], initializer=tf.initializers.zeros())
                self.params_list += [self.weights[layer_i], self.biases[layer_i]]
                self.trainable_params_list += [self.weights[layer_i], self.biases[layer_i]]
                width *= stride[0]
                height *= stride[1]

            # old backups of the parameters, as estimated via EMA
            if self.keep_old:
                width = self.out_shape[0] / np.prod([stride[0] for _, stride, _ in self.layers])
                height = self.out_shape[1] / np.prod([stride[1] for _, stride, _ in self.layers])
                depth = self.layers[0][2]
                self.old_initial_weight = tf.get_variable("old_initial_weight", [self.in_size, width * height * depth], trainable=False)
                self.old_initial_bias = tf.get_variable("old_initial_bias", [1, width * height * depth], trainable=False)
                self.params_list += [self.old_initial_weight, self.old_initial_bias]
                self.old_init_ops.append(tf.assign(self.old_initial_weight, self.initial_weight))
                self.old_init_ops.append(tf.assign(self.old_initial_bias, self.initial_bias))
                self.old_ema_ops.append(tf.assign(self.old_initial_weight, (1. - self.old_ema_rate) * self.old_initial_weight + self.old_ema_rate * self.initial_weight))
                self.old_ema_ops.append(tf.assign(self.old_initial_bias, (1. - self.old_ema_rate) * self.old_initial_bias + self.old_ema_rate * self.initial_bias))
                for layer_i in range(self.n_layers):
                    filter_shape, stride, in_depth = self.layers[layer_i]
                    out_depth = self.layers[layer_i + 1][2] if layer_i + 1 < self.n_layers else self.out_shape[2]
                    self.old_weights[layer_i] = tf.get_variable("old_weights%d" % layer_i, filter_shape + [out_depth, in_depth], trainable=False)
                    self.old_biases[layer_i] = tf.get_variable("old_bias%d" % layer_i, [1, 1, 1, out_depth], trainable=False)
                    self.params_list += [self.old_weights[layer_i], self.old_biases[layer_i]]
                    self.old_init_ops.append(tf.assign(self.old_weights[layer_i], self.weights[layer_i]))
                    self.old_init_ops.append(tf.assign(self.old_biases[layer_i], self.biases[layer_i]))
                    self.old_ema_ops.append(tf.assign(self.old_weights[layer_i], (1. - self.old_ema_rate) * self.old_weights[layer_i] + self.old_ema_rate * self.weights[layer_i]))
                    self.old_ema_ops.append(tf.assign(self.old_biases[layer_i], (1. - self.old_ema_rate) * self.old_biases[layer_i] + self.old_ema_rate * self.biases[layer_i]))
                    width *= stride[0]
                    height *= stride[1]

    def __call__(self, x, use_old=False, stop_params_gradient=False, add_x_channel_dim=True):
        if add_x_channel_dim: x = tf.expand_dims(x, -1)

        # decide whether to use old weights
        if use_old:
            assert self.keep_old
            weights = [tf.stop_gradient(weight) for weight in self.old_weights]
            biases = [tf.stop_gradient(bias) for bias in self.old_biases]
            initial_weight = tf.stop_gradient(self.old_initial_weight)
            initial_bias = tf.stop_gradient(self.old_initial_bias)
        elif stop_params_gradient:
            weights = [tf.stop_gradient(weight) for weight in self.weights]
            biases = [tf.stop_gradient(bias) for bias in self.biases]
            initial_weight = tf.stop_gradient(self.initial_weight)
            initial_bias = tf.stop_gradient(self.initial_bias)
        else:
            weights = self.weights
            biases = self.biases
            initial_weight = self.initial_weight
            initial_bias = self.initial_bias

        # reshape input
        batch_shape = tf.shape(x)[:-1]
        assert x.shape.as_list()[-1] == self.in_size
        x = tf.reshape(x, [-1] + [self.in_size])

        # main network
        h = x
        width = self.out_shape[0] / np.prod([stride[0] for _, stride, _ in self.layers])
        height = self.out_shape[1] / np.prod([stride[1] for _, stride, _ in self.layers])
        depth = self.layers[0][2]
        h = tf.matmul(h, initial_weight) + initial_bias
        h = tf.reshape(h, [-1, width, height, depth])
        for layer_i in range(self.n_layers):
            filter_shape, stride, in_depth = self.layers[layer_i]
            out_depth = self.layers[layer_i + 1][2] if layer_i + 1 < self.n_layers else self.out_shape[2]
            h = self.internal_nonlinearity(tf.nn.conv2d_transpose(h, weights[layer_i], output_shape=(tf.shape(h)[0], width*stride[0], height*stride[1], out_depth), strides=[1]+stride+[1]) + biases[layer_i])
            width *= stride[0]
            height *= stride[1]

        # reshape to original batching
        h = tf.reshape(h, tf.concat([batch_shape, tf.constant(self.out_shape, dtype=tf.int32)], axis=0))
        h = self.final_nonlinearity(h)

        return h

    def l2(self):
        return tf.add_n([tf.reduce_sum(tf.square(param)) for param in self.trainable_params_list])

    def clipping(self):
        return [tf.assign(param, tf.clip_by_value(param, -1., 1.)) for param in self.trainable_params_list]