"""
    class for simulating the response of an LTI system
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.linalg import dft
from scipy.signal import cheby1


class LTILayer(layers.Layer):

    def __init__(self, T, M, **kwargs):
        """
        class constructor

        T  : Total duration of the input signal
        M  : Number of time steps
        """
        super(LTILayer, self).__init__(**kwargs)

        # define filter coefficients
        num, den = cheby1(4, 0.1, 2 * np.pi * 20, analog=True)

        # define frequency vector
        f = np.reshape(np.fft.fftfreq(M) * M / T, (1, M))

        # evaluate the dft matrix
        F = dft(M, 'sqrtn')

        # evaluate the numerator and denominator at each frequency
        H_num = np.concatenate([(1j * 2 * np.pi * f) ** s for s in range(len(num) - 1, -1, -1)], axis=0)
        H_den = np.concatenate([(1j * 2 * np.pi * f) ** s for s in range(len(den) - 1, -1, -1)], axis=0)

        # evaluate the frequency response
        H = np.diag((num @ H_num) / (den @ H_den))

        # evaluate the full transformation and convert to a tensor of correct shape
        self.L = tf.constant(np.reshape(F.conj().T @ H @ F, (1, 1, M, M)), dtype=tf.complex64)

    def call(self, inputs):
        """
        Method to evaluate the ouput of the layer which represents the response of the system to the input
        """

        # convert variables to complex
        x = tf.cast(tf.transpose(inputs, perm=[0, 2, 1, 3]), tf.complex64)

        # repeat the transformation matrix
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, 1, 1], dtype=np.int32))], 0)
        L = tf.tile(self.L, temp_shape)

        # apply the transformation
        y = tf.transpose(tf.math.real(tf.matmul(L, x)), perm=[0, 2, 1, 3])

        return y
