import tensorflow as tf


def flatten(x):
    batch, _, _, channels = [i.value for i in x.get_shape()]
    return tf.reshape(x, shape=[batch, -1, channels])


def convLayer(net, num_filters, filter_size, name, relu, strides=1, bias=False, pad='REFLECT', Norm='INSTANCE',
              training=True):
    weights_init = convInit(net, num_filters, filter_size, name=name)
    if 'SPECTRAL' in Norm:
        weights_init = spectral(weights_init, name, training)
    strides_shape = [1, strides, strides, 1]
    if pad == 'REFLECT':
        net = tf.pad(net, [[0, 0], [filter_size // 2, filter_size // 2],
                           [filter_size // 2, filter_size // 2], [0, 0]], mode="REFLECT")
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='VALID')
    elif pad == 'SAME':
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    elif pad == 'VALID':
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='VALID')
    if bias:
        net = net + tf.get_variable(name + '_bias', [num_filters])
    if 'NOT' not in Norm:
        if 'INSTANCE' in Norm:
            net = instanceNorm(net, name=name)
        elif 'BATCH' in Norm:
            net = batchNorm(net, training, name=name)
    if relu is 'RELU':
        net = tf.nn.relu(net)
    elif isinstance(relu, float):
        net = tf.nn.leaky_relu(net, relu)
    return net


def residualBlock(net, filter_num, filter_size, name, pad, relu='RELU', Norm='INSTANCE', training=True):
    tmp = convLayer(net, filter_num, filter_size, name, 1, pad=pad, Norm=Norm, training=training)
    return net + convLayer(tmp, filter_num, filter_size, name + '_1', strides=1, relu=relu, pad=pad, Norm=Norm,
                           training=training)


def newResidualBlock(net, filter_num, filter_size, name, pad, training=True):
    x = batchNorm(net, training=training, name=name + '_bn')
    x = tf.nn.leaky_relu(x)
    x = convLayer(x, filter_num, filter_size, pad=pad, strides=1, Norm='SPECTRAL,BATCH', training=training,
                  name=name + '_deconv1', relu=.2, bias=False)
    x = convLayer(x, filter_num, filter_size, pad=pad, Norm='SPECTRAL,NOT', name=name + '_conv1', relu=False,
                  training=training, bias=True)
    return x + net


def residualBlockUp(net, filter_num, filter_size, name, pad, training=True):
    x = batchNorm(net, training=training, name=name + '_bn')
    x = tf.nn.leaky_relu(x)
    x = transposeConv(x, filter_num, filter_size, pad=pad, strides=2, Norm='SPECTRAL,BATCH', training=training,
                      name=name + '_deconv1', relu=.2, bias=False)
    x = convLayer(x, filter_num, filter_size, pad=pad, Norm='SPECTRAL,NOT', name=name + '_conv1', relu=False,
                  training=training, bias=True)
    s = transposeConv(net, filter_num, filter_size, strides=2, Norm='SPECTRAL,NOT', pad=pad, name=name + '_deconv2',
                      relu=False, training=training, bias=True)
    return x + s


def residualBlockDown(net, filter_num, filter_size, name, pad, training=True):
    x = batchNorm(net, training=training, name=name)
    x = tf.nn.leaky_relu(x)
    x = convLayer(x, filter_num, filter_size, strides=2, pad=pad, Norm='SPECTRAL,BATCH', training=training,
                  name=name + '_conv1', relu=.2, bias=False)
    x = convLayer(x, filter_num, filter_size, pad=pad, Norm='SPECTRAL,NOT', name=name + '_conv2', relu=False,
                  training=training, bias=True)
    s = convLayer(net, filter_num, filter_size, strides=2, pad=pad, Norm='SPECTRAL,NOT', name=name + '_conv3',
                  relu=False, training=training, bias=True)
    return x + s


def instanceNorm(net, name, training=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.get_variable(initializer=tf.zeros(var_shape), name=name + "_shift")
    scale = tf.get_variable(shape=var_shape, initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32),
                            name=name + "_scale")
    epsilon = 1e-9
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift


def batchNorm(x, training, name, decay=0.9):
    # batch, rows, cols, channels = [i.value for i in x.get_shape()]
    # size = [channels]
    # scale = tf.get_variable(initializer=tf.ones(size), name=name + 'scale')
    # shift = tf.get_variable(initializer=tf.ones(size), name=name + 'shift')
    # pop_mean = tf.get_variable(initializer=tf.zeros(size), trainable=False, name=name + 'pop_mean')
    # pop_var = tf.get_variable(initializer=tf.ones(size), trainable=False, name=name + 'pop_var')
    # epsilon = 1e-3
    #
    # batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    # train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    # train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
    #
    # def batch_statistics():
    #     with tf.control_dependencies([train_mean, train_var]):
    #         return tf.nn.batch_normalization(x, batch_mean, batch_var, shift, scale, epsilon, name='batchNorm')
    #
    # def population_statistics():
    #     return tf.nn.batch_normalization(x, pop_mean, pop_var, shift, scale, epsilon, name='batchNorm')
    #
    # if training:
    #     return batch_statistics()
    # else:
    #     return population_statistics()
    return tf.layers.batch_normalization(x,
                                         momentum=decay,
                                         epsilon=1e-05,
                                         training=training,
                                         name=name)


def spectral(weight, name, is_training, iter=1):
    _, _, _, out_channels = [i.value for i in weight.get_shape()]
    w = tf.reshape(weight, [-1, out_channels])

    u = tf.get_variable(name + "u", [1, out_channels], initializer=tf.truncated_normal_initializer(),
                        trainable=False)  # [1, output_filters]

    u_norm = u
    v_norm = None
    for i in range(iter):
        v_ = tf.matmul(u_norm, w, transpose_b=True)  # [1, N]
        v_norm = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_norm, w)  # [1, output_filters]
        u_norm = tf.nn.l2_normalize(u_)

    sigma = tf.matmul(tf.matmul(v_norm, w), u_norm, transpose_b=True)  # [1,1]
    w_norm = w / sigma

    with tf.control_dependencies([tf.cond(tf.cast(is_training, tf.bool),
                                          true_fn=lambda: u.assign(u_norm), false_fn=lambda: u.assign(u))]):
        w_norm = tf.reshape(w_norm, [i.value for i in weight.get_shape()])

    return w_norm


def convInit(net, out_channels, filter_size, name, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.get_variable(initializer=tf.truncated_normal(weights_shape, stddev=0.02),
                                   dtype=tf.float32, name=name)
    return weights_init


def transposeConv(net, num_filters, filter_size, strides, name, relu, pad="VALID", Norm='INSTANCE', training=True,
                  bias=True):
    weights_init = convInit(net, num_filters, filter_size, transpose=True, name=name)
    if 'SPECTRAL' in Norm:
        weights_init = spectral(weights_init, name, training)
    batch_size, rows, cols, _ = [i.value for i in net.get_shape()]
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])
    if pad == 'SAME':
        new_shape = [batch_size, rows * strides, cols * strides, num_filters]

    else:
        new_shape = [batch_size, rows * strides + max(filter_size - strides, 0),
                     cols * strides + max(filter_size - strides, 0), num_filters]
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, new_shape, strides_shape, padding=pad)
    if bias:
        net = net + tf.get_variable(name + '_bias', [num_filters])
    if 'NOT' not in Norm:
        if 'INSTANCE' in Norm:
            net = instanceNorm(net, name=name)
        elif 'BATCH' in Norm:
            net = batchNorm(net, training, name=name)
    if relu is 'RELU':
        net = tf.nn.relu(net)
    elif isinstance(relu, float):
        net = tf.nn.leaky_relu(net, relu)
    return net


def resizeConv2D(net, num_filters, filter_size, name, relu, strides=1, bias=False, pad='VALID', Norm='INSTANCE',
                 training=True):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''

    height = net.get_shape()[1].value
    width = net.get_shape()[2].value
    new_height = int(height * strides)
    new_width = int(width * strides)

    net_resized = tf.image.resize_images(net, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return convLayer(net_resized, num_filters=num_filters, filter_size=filter_size, name=name, strides=1,
                     bias=bias, relu=relu, pad=pad, Norm=Norm, training=training)


def attention(net, num_filters, is_training, name):
    f = convLayer(net, num_filters // 8, filter_size=1, strides=1, bias=True, pad='VALID', training=is_training,
                  name=name + '_f', Norm='SPECTRAL,NOT', relu=False)
    g = convLayer(net, num_filters // 8, filter_size=1, strides=1, bias=True, pad='VALID', training=is_training,
                  name=name + '_g', Norm='SPECTRAL,NOT', relu=False)
    h = convLayer(net, num_filters, filter_size=1, strides=1, bias=True, pad='VALID', training=is_training,
                  name=name + '_h', Norm='SPECTRAL,NOT', relu=False)

    f_flatten = flatten(f)
    g_flatten = flatten(g)

    s = tf.matmul(g_flatten, f_flatten, transpose_b=True)  # [bs, N, N]
    beta = tf.nn.softmax(s, axis=-1)  # attention map

    o = tf.matmul(beta, flatten(h))  # [bs, N, N]*[bs, N, c]->[bs, N, c]
    gamma = tf.get_variable(name + "gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=[i.value for i in net.get_shape()])  # [bs, h, w, c]
    net = gamma * o + net

    return net
