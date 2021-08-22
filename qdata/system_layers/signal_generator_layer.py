"""
This class defines a custom tensorflow layer that generates a sequence of control pulse parameters
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class SignalGenerator(layers.Layer):

    def __init__(self, T, M, n_max, waveform="Gaussian", **kwargs):
        """
        class constructor

        T             : Total time of evolution
        M             : Number of discrete time steps
        n_max         : Maximum number of control pulses in the sequence
        waveform      : Waveform shape can either be "Gaussian", "Square", or "Zero"
        """
        # we must call thus function for any tensorflow custom layer
        super(SigGen, self).__init__(**kwargs)

        # store the parameters
        self.n_max = n_max
        self.T = T
        self.M = M
        self.time_range = tf.constant(np.reshape([(0.5 * T / M) + (j * T / M) for j in range(M)], (1, M, 1, 1)),
                                      dtype=tf.float32)

        if waveform == "Gaussian":
            self.call = self.call_Gaussian
        elif waveform == "Square":
            self.call = self.call_Square
        else:
            self.call = self.call_0

        # define the constant parmaters to shift the pulses correctly
        self.pulse_width = (0.5 * self.T / self.n_max)

        self.a_matrix = np.ones((self.n_max, self.n_max))
        self.a_matrix[np.triu_indices(self.n_max, 1)] = 0
        self.a_matrix = tf.constant(np.reshape(self.a_matrix, (1, self.n_max, self.n_max)), dtype=tf.float32)

        self.b_matrix = np.reshape([idx + 0.5 for idx in range(self.n_max)], (1, self.n_max, 1)) * self.pulse_width
        self.b_matrix = tf.constant(self.b_matrix, dtype=tf.float32)

    def call_Square(self, inputs, training=False):
        """
        Method to generate square pulses

        """
        # generate randomly the signal parameters
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, 1], dtype=np.int32))], 0)
        a_matrix = tf.tile(self.a_matrix, temp_shape)
        b_matrix = tf.tile(self.b_matrix, temp_shape)

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.n_max, 1], dtype=np.int32))], 0)
        amplitude = 100 * tf.random.uniform(shape=temp_shape, minval=-1, maxval=1, dtype=tf.float32)
        position = 0.5 * self.pulse_width + tf.random.uniform(shape=temp_shape, dtype=tf.float32) * (
                    ((self.T - self.n_max * self.pulse_width) / (self.n_max + 1)) - 0.5 * self.pulse_width)
        position = tf.matmul(a_matrix, position) + b_matrix
        std = self.pulse_width * tf.ones(temp_shape, dtype=tf.float32)

        # combine the parameters into one tensor
        signal_parameters = tf.concat([amplitude, position, std], -1)

        # construct the signal
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, 1, 1], dtype=np.int32))], 0)
        time_range = tf.tile(self.time_range, temp_shape)
        tau = [tf.reshape(tf.matmul(position[:, idx, :], tf.ones([1, self.M])), (tf.shape(time_range))) for idx in
               range(self.n_max)]
        A = [tf.reshape(tf.matmul(amplitude[:, idx, :], tf.ones([1, self.M])), (tf.shape(time_range))) for idx in
             range(self.n_max)]
        sigma = [tf.reshape(tf.matmul(std[:, idx, :], tf.ones([1, self.M])), (tf.shape(time_range))) for idx in
                 range(self.n_max)]
        signal = [tf.multiply(A[idx], tf.cast(tf.logical_and(tf.greater(time_range, tau[idx] - 0.5 * sigma[idx]),
                                                             tf.less(time_range, tau[idx] + 0.5 * sigma[idx])),
                                              tf.float32)) for idx in range(self.n_max)]
        signal = tf.add_n(signal)

        return signal_parameters, signal

    def call_Gaussian(self, inputs, training=False):
        """
        Method to generate Gaussian pulses

        """

        # generate randomly the signal parameters
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, 1], dtype=np.int32))], 0)
        a_matrix = tf.tile(self.a_matrix, temp_shape)
        b_matrix = tf.tile(self.b_matrix, temp_shape)

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.n_max, 1], dtype=np.int32))], 0)
        amplitude = 100 * tf.random.uniform(shape=temp_shape, minval=-1, maxval=1, dtype=tf.float32)
        position = 0.5 * self.pulse_width + tf.random.uniform(shape=temp_shape, dtype=tf.float32) * (
                    ((self.T - self.n_max * self.pulse_width) / (self.n_max + 1)) - 0.5 * self.pulse_width)
        position = tf.matmul(a_matrix, position) + b_matrix
        std = self.pulse_width * tf.ones(temp_shape, dtype=tf.float32) / 6

        # combine the parameters into one tensor
        signal_parameters = tf.concat([amplitude, position, std], -1)

        # construct the signal
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, 1, 1], dtype=np.int32))], 0)
        time_range = tf.tile(self.time_range, temp_shape)
        tau = [tf.reshape(tf.matmul(position[:, idx, :], tf.ones([1, self.M])), (tf.shape(time_range))) for idx in
               range(self.n_max)]
        A = [tf.reshape(tf.matmul(amplitude[:, idx, :], tf.ones([1, self.M])), (tf.shape(time_range))) for idx in
             range(self.n_max)]
        sigma = [tf.reshape(tf.matmul(std[:, idx, :], tf.ones([1, self.M])), (tf.shape(time_range))) for idx in
                 range(self.n_max)]
        signal = [tf.multiply(A[idx], tf.exp(-0.5 * tf.square(tf.divide(time_range - tau[idx], sigma[idx])))) for idx in
                  range(self.n_max)]
        signal = tf.add_n(signal)

        return signal_parameters, signal

    def call_0(self, inputs, training=False):
        """
        Method to generate the zero pulse sequence [for free evolution analysis]
        """

        # construct zero signal
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.M, 1, 1], dtype=np.int32))], 0)
        signal = tf.zeros(temp_shape, dtype=tf.float32)
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.n_max, 3], dtype=np.int32))], 0)
        signal_parameters = tf.zeros(temp_shape, dtype=tf.float32)

        return signal_parameters, signal
