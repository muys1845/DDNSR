from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Activation
from tensorflow import keras
from method import layers
import tensorflow as tf


def degradation_network(filters=64, downscale_layers=2, blocks=8, input_type='dual'):
    """ Our proposed dual-input degradation network. """
    x1, x2 = keras.Input(shape=(None, None, 3)), keras.Input(shape=(None, None, 3))
    y = layers.conv(filters)(x1)
    for i in range(1, downscale_layers + 1):
        for kernel_size, strides in zip([3, 4], [1, 2]):
            y = layers.conv(filters << i, kernel_size, strides, activation=tf.nn.leaky_relu)(y)
    y = layers.conv(filters)(y)

    assert input_type in ['dual', 'single']
    if input_type == 'dual':
        noise = x2
    else:
        noise = tf.zeros_like(x2)

    for i in range(4):
        noise = layers.conv(filters << i, activation=tf.nn.leaky_relu)(noise)

    dy = layers.conv(filters)(tf.concat([y, noise], axis=3))
    for _ in range(blocks):
        dy = layers.ResBlock(filters)(dy)
    y += layers.conv(filters)(dy)

    y = layers.conv(3)(y)
    return keras.Model([x1, x2], y)


def reconstruction_network(filters=64, blocks=23, upscale_layers=2):
    """ The RRDB-net. """
    x = keras.Input(shape=(None, None, 3))
    dy = y = layers.conv(filters)(x)
    for _ in range(blocks):
        dy = layers.RRDB()(dy)
    y += layers.conv(filters)(dy)
    for _ in range(upscale_layers):
        y = layers.upscale(y, 2)
        y = layers.conv(filters, activation=tf.nn.leaky_relu)(y)
    y = layers.conv(filters, activation=tf.nn.leaky_relu)(y)
    y = layers.conv(3)(y)
    return keras.Model(x, y)


def discriminator_vggstyle(filters, downscale_layers):
    """ VGG style discriminator, used for track1. """
    model = keras.Sequential([layers.conv(filters, activation=tf.nn.leaky_relu)])
    for i in range(downscale_layers):
        for kernel_size, strides in zip([3, 4], [1, 2]):
            if i != 0 or kernel_size != 3:
                model.add(layers.conv(filters << i, kernel_size, strides, use_bias=False))
                model.add(BatchNormalization())
                model.add(Activation(tf.nn.leaky_relu))
    model.add(Flatten())
    model.add(Dense(100, activation=tf.nn.leaky_relu))
    model.add(Dense(1))
    return model


def discriminator_patchgan():
    """ PatchGAN discriminator, used for track2. """
    model = keras.Sequential([layers.conv(64, 4, strides=2, activation=tf.nn.leaky_relu)])
    for filters, strides in zip([128, 256, 512], [2, 2, 1]):
        model.add(layers.conv(filters, 4, strides=strides, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation(tf.nn.leaky_relu))
    model.add(layers.conv(1, 4))
    return model
