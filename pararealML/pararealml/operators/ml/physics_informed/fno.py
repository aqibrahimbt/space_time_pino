# import numpy as np
# import tensorflow as tf
# from pararealml import *
# from pararealml.operators.fdm import *
# from pararealml.operators.ml.physics_informed import *


# import pickle
# from keras import layers


# class SpectralConv2d(layers.Layer):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(SpectralConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = self.add_weight(shape=(in_channels, out_channels, self.modes1, self.modes2, 2),
#                                         initializer=tf.initializers.RandomNormal(0.0, self.scale),
#                                         trainable=True)
#         self.weights2 = self.add_weight(shape=(in_channels, out_channels, self.modes1, self.modes2, 2),
#                                         initializer=tf.initializers.RandomNormal(0.0, self.scale),
#                                         trainable=True)

#     def call(self, x):
#         batchsize = tf.shape(x)[0]
#         x_ft = tf.signal.rfft2d(x)
#         out_ft = tf.zeros(tf.concat([[batchsize, self.out_channels], tf.shape(x_ft)[2:]], axis=0), dtype=tf.complex64)
#         out_ft = tf.tensor_scatter_nd_update(
#             out_ft, 
#             tf.constant([[i, j] for i in range(min(self.modes1, tf.shape(out_ft)[2])) 
#                                for j in range(min(self.modes2, tf.shape(out_ft)[3]))]), 
#             tf.einsum("bixy,ioxy->boxy", 
#                       tf.cast(x_ft[:, :, :self.modes1, :self.modes2], dtype=tf.complex64),
#                       tf.complex(self.weights1[:, :, :, :, 0], self.weights1[:, :, :, :, 1]))
#         )
#         x = tf.signal.irfft2d(out_ft, fft_length=tf.shape(x)[1:3])
#         return x

# class FNO2D(tf.keras.Model):
#     def __init__(self, modes1, modes2, width):
#         super(FNO2D, self).__init__()
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.width = width
        
#         self.fc0 = layers.Dense(self.width)
#         self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.w0 = layers.Dense(self.width)
#         self.w1 = layers.Dense(self.width)
#         self.w2 = layers.Dense(self.width)
#         self.w3 = layers.Dense(self.width)
#         self.fc1 = layers.Dense(128)
#         self.fc2 = layers.Dense(1)

#         self.transpose_layer = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))

#     def call(self, inputs):
#         # Assuming inputs shape is (batch_size, n_points, 3)
#         # where 3 represents (x, y, t) coordinates
#         x = self.fc0(inputs)
#         x = self.transpose_layer(x)  # (batch, width, n_points)
#         x = tf.reshape(x, [-1, self.width, 100, 100])  # Assuming 100x100 grid
        
#         x1 = self.conv0(x)
#         x2 = self.w0(tf.transpose(x, perm=[0, 2, 3, 1]))
#         x = x1 + tf.transpose(x2, perm=[0, 3, 1, 2])
#         x = tf.nn.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(tf.transpose(x, perm=[0, 2, 3, 1]))
#         x = x1 + tf.transpose(x2, perm=[0, 3, 1, 2])
#         x = tf.nn.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(tf.transpose(x, perm=[0, 2, 3, 1]))
#         x = x1 + tf.transpose(x2, perm=[0, 3, 1, 2])
#         x = tf.nn.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(tf.transpose(x, perm=[0, 2, 3, 1]))
#         x = x1 + tf.transpose(x2, perm=[0, 3, 1, 2])

#         x = tf.transpose(x, perm=[0, 2, 3, 1])  # (batch, 100, 100, width)
#         x = self.fc1(x)
#         x = tf.nn.gelu(x)
#         x = self.fc2(x)
        
#         return tf.reshape(x, [-1, 10000, 1])  # Reshape back to (batch, n_points, 1)

import tensorflow as tf

class SpectralConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = self.add_weight(shape=(in_channels, out_channels, modes1, modes2),
                                        initializer=tf.initializers.RandomNormal(0, self.scale),
                                        trainable=True, name='weights1')
        self.weights2 = self.add_weight(shape=(in_channels, out_channels, modes1, modes2),
                                        initializer=tf.initializers.RandomNormal(0, self.scale),
                                        trainable=True, name='weights2')

    def call(self, x):
        batch_size = tf.shape(x)[0]
        size1, size2 = tf.shape(x)[1], tf.shape(x)[2]

        x_ft = tf.signal.fft2d(tf.cast(x, tf.complex64))
        
        # Compute first part
        out1 = tf.einsum("bixy,ioxy->boxy", 
                         x_ft[:, :self.modes1, :self.modes2, :], 
                         tf.cast(self.weights1, tf.complex64))
        
        # Compute second part
        out2 = tf.einsum("bixy,ioxy->boxy", 
                         x_ft[:, -self.modes1:, :self.modes2, :], 
                         tf.cast(self.weights2, tf.complex64))
        
        # Combine results
        out_ft = tf.concat([out1, tf.zeros((batch_size, size1 - 2*self.modes1, self.modes2, self.out_channels), dtype=tf.complex64), out2], axis=1)
        
        return tf.math.real(tf.signal.ifft2d(out_ft))

class FNO2d(tf.keras.Model):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = tf.keras.layers.Dense(self.width)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1)

        self.conv_layers = [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(4)]
        self.w_layers = [tf.keras.layers.Conv2D(self.width, (1, 1), activation='relu') for _ in range(4)]

    def call(self, x):
        x = self.fc0(x)
        x = tf.reshape(x, (-1, self.width, 1, 1))

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = tf.nn.relu(x)

        x = tf.reshape(x, (-1, self.width))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class FNO(tf.keras.Model):
    def __init__(
        self,
        modes1: int = 4,
        modes2: int = 4,
        width: int = 64,
        branch_net_input_size: int = 13,
    ):
        super(FNO, self).__init__()
        self._fno = FNO2d(modes1, modes2, width)
        self._branch_net_input_size = branch_net_input_size

    @property
    def branch_net_input_size(self) -> int:
        return self._branch_net_input_size

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = None,
        mask: tf.Tensor = None,
    ) -> tf.Tensor:
        return self._fno(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], 1])