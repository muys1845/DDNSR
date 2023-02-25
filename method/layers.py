from tensorflow import keras
import tensorflow as tf


def vgg_feature_model(i: int, j: int, weights='imagenet'):
    """ Feature extractor from a pretrained VGG19 network. """
    vgg = keras.applications.VGG19(include_top=False, weights=weights)
    vgg.trainable = False
    conv_layer = vgg.get_layer('block{}_conv{}'.format(i, j))
    conv_layer.activation = None
    return keras.Model(inputs=vgg.input, outputs=conv_layer.output)


def conv(filters, kernel_size=3, strides=1, activation=None, use_bias=True):
    """ Basic convolution layer. """
    return keras.layers.Conv2D(filters, kernel_size, strides, padding='same', activation=activation, use_bias=use_bias)


class ResBlock(keras.layers.Layer):
    """ Basic residual block. """

    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.c1 = conv(filters, activation=tf.nn.leaky_relu)
        self.c2 = conv(filters)
        self.config = {'filters': filters}

    def call(self, inputs):
        dy = self.c1(inputs)
        return inputs + self.c2(dy)

    def get_config(self):
        return self.config


class DenseBlock(keras.layers.Layer):
    """ Basic dense block. """

    def __init__(self, filters_growing=32, filters_out=64):
        super(DenseBlock, self).__init__()
        self.conv = [conv(filters_growing, activation=tf.nn.leaky_relu) for _ in range(4)]
        self.out = conv(filters_out)
        self.config = {'filters_growing': filters_growing, 'filters_out': filters_out}

    def call(self, inputs):
        y = inputs
        temp = [y]
        for layer in self.conv:
            temp.append(layer(y))
            y = tf.concat(temp, axis=3)
        return self.out(y)

    def get_config(self):
        return self.config


class RRDB(keras.layers.Layer):
    """ Basic RRDB block. """

    def __init__(self, filters_growing=32, filters_out=64, blocks=3, beta=0.2):
        super(RRDB, self).__init__()
        self.blocks = [DenseBlock(filters_growing, filters_out) for _ in range(blocks)]
        self.beta = beta
        self.config = {'filters_growing': filters_growing, 'filters_out': filters_out, 'blocks': blocks, 'beta': beta}

    def call(self, inputs):
        y = inputs
        for block in self.blocks:
            y += block(y) * self.beta
        return inputs + y * self.beta

    def get_config(self):
        return self.config


def upsample(imgs, scale: int = 4, method='bicubic'):
    """ Bicubic interpolation. """
    b, h, w, c = imgs.shape
    return tf.image.resize(imgs, (h * scale, w * scale), method=method)


def downsample(imgs, scale: int = 4, method='bicubic'):
    """ Bicubic down-sample. """
    b, h, w, c = imgs.shape
    return tf.image.resize(imgs, (h // scale, w // scale), method=method)


def upscale(x, scale: int = 4):
    """ Box upscaling (also called nearest neighbors). """
    b, h, w = tf.shape(x)[:-1]
    c = x.shape[-1]
    x = tf.reshape(x, [b, h, 1, w, 1, c])
    x = tf.tile(x, [1, 1, scale, 1, scale, 1])
    x = tf.reshape(x, [b, h * scale, w * scale, c])
    return x


def downscale(x, scale: int = 4):
    """ Average pool. """
    return tf.nn.avg_pool(x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')


custom_objects = {'ResBlock': ResBlock, 'DenseBlock': DenseBlock, 'RRDB': RRDB, 'leaky_relu': tf.nn.leaky_relu}
