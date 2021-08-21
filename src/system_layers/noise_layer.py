"""
class for generating time-domain realizations of noise
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class NoiseLayer(layers.Layer):

    def __init__(self, T, M, K, profile, **kwargs):
        """
        class constructor

        T      : Total duration of the input signal
        M      : Number of time steps
        K      : Number of realizations
        profile: Type of noise

        """
        super(NoiseLayer, self).__init__(**kwargs)

        # store class parameters
        self.T = T
        self.M = M
        self.K = K

        # define a vector of discriteized frequencies
        f = np.fft.fftfreq(M) * M / T

        # define time step
        Ts = T / M

        # check the noise type, initialize required variables and define the correct "call" method
        if profile == 0:  # No noise
            self.call = self.call_0
        elif profile == 1:  # PSD of 1/f + a bump
            alpha = 1
            S_Z = 1 * np.array(
                [(1 / (fq + 1) ** alpha) * (fq <= 15) + (1 / 16) * (fq > 15) + np.exp(-((fq - 30) ** 2) / 50) / 2 for fq
                 in f[f >= 0]])
            self.P_temp = tf.constant(np.tile(np.reshape(np.sqrt(S_Z * M / Ts), (1, 1, self.M // 2)), (1, self.K, 1)),
                                      dtype=tf.complex64)
            self.call = self.call_1
        elif profile == 2:  # Colored Gaussian Stationary Noise
            self.g = 0.1
            self.color = tf.ones([self.M // 4, 1, 1], dtype=tf.float32)
            self.call = self.call_2
        elif profile == 3:  # Colored Gaussian Non-stationary Noise
            time_range = [(0.5 * T / M) + (j * T / M) for j in range(M)]
            self.g = 0.2
            self.color = tf.ones([self.M // 4, 1, 1], dtype=tf.float32)
            self.non_stationary = tf.constant(
                np.reshape(1 - (np.abs(np.array(time_range) - 0.5 * T) * 2), (1, M, 1, 1)), dtype=tf.float32)
            self.call = self.call_3
        elif profile == 4:  # Colored Non-Gaussian Non-stationary Noise
            time_range = [(0.5 * T / M) + (j * T / M) for j in range(M)]
            self.g = 0.01
            self.color = tf.ones([self.M // 4, 1, 1], dtype=tf.float32)
            self.non_stationary = tf.constant(
                np.reshape(1 - (np.abs(np.array(time_range) - 0.5 * T) * 2), (1, M, 1, 1)), dtype=tf.float32)
            self.call = self.call_4
        elif profile == 5:  # PSD of 1/f
            alpha = 1
            S_Z = 1 * np.array([(1 / (fq + 1) ** alpha) for fq in f[f >= 0]])
            self.P_temp = tf.constant(np.tile(np.reshape(np.sqrt(S_Z * M / Ts), (1, 1, self.M // 2)), (1, self.K, 1)),
                                      dtype=tf.complex64)
            self.call = self.call_1
        elif profile == 6:  # correlated noise
            self.g = 0.3
            self.call = self.call_6

    def call_0(self, inputs, training=False):  # No noise
        """
        Method to generate type 0 noise

        """
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.M, self.K, 1], dtype=np.int32))], 0)
        return tf.zeros(temp_shape, dtype=tf.float32)

    def call_1(self, inputs, training=False):  # PSD of 1/f + a bump
        """
        Method to generate type 1 and type 5 noise

        """
        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, 1], dtype=np.int32))], 0)
        P_temp = tf.tile(self.P_temp, temp_shape)

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M // 2], dtype=np.int32))], 0)
        P_temp = tf.multiply(P_temp, tf.exp(
            2 * np.pi * 1j * tf.cast(tf.random.uniform(temp_shape, dtype=tf.float32), dtype=tf.complex64)))

        noise = tf.math.real(
            tf.signal.ifft(tf.concat([P_temp, tf.reverse(tf.math.conj(P_temp), axis=tf.constant([2]))], axis=2)))
        noise = tf.transpose(tf.expand_dims(noise, axis=-1), perm=[0, 2, 1, 3])

        return noise

    def call_2(self, inputs, training=False):  # Colored Gaussian Stationary Noise
        """
        Method to generate type 2 noise

        """
        temp_shape = tf.concat(
            [self.K * tf.shape(inputs)[0:1], tf.constant(np.array([self.M + (self.M // 4) - 1, 1], dtype=np.int32))], 0)
        noise = self.g * tf.nn.convolution(input=tf.random.normal(temp_shape), filters=self.color, padding="VALID")

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M, 1], dtype=np.int32))], 0)
        noise = tf.transpose(tf.reshape(tf.transpose(noise, perm=[0, 2, 1]), temp_shape), perm=[0, 2, 1, 3])

        return noise

    def call_3(self, inputs, training=False):  # Colored Gaussian Non-stationary Noise
        """
        Method to generate type 3 noise

        """
        temp_shape = tf.concat(
            [self.K * tf.shape(inputs)[0:1], tf.constant(np.array([self.M + (self.M // 4) - 1, 1], dtype=np.int32))], 0)
        noise = self.g * tf.nn.convolution(input=tf.random.normal(temp_shape), filters=self.color, padding="VALID")

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M, 1], dtype=np.int32))], 0)
        noise = tf.transpose(tf.reshape(tf.transpose(noise, perm=[0, 2, 1]), temp_shape), perm=[0, 2, 1, 3])

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, self.K, 1], dtype=np.int32))], 0)
        non_stationary = tf.tile(self.non_stationary, temp_shape)

        return tf.multiply(noise, non_stationary)

    def call_4(self, inputs, training=False):  # Colored Gaussian Non-stationary Noise
        """
        Method to generate type 4 noise

        """

        temp_shape = tf.concat(
            [self.K * tf.shape(inputs)[0:1], tf.constant(np.array([self.M + (self.M // 4) - 1, 1], dtype=np.int32))], 0)
        noise = tf.nn.convolution(input=tf.random.normal(temp_shape), filters=self.color, padding="VALID")

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([self.K, self.M, 1], dtype=np.int32))], 0)
        noise = tf.transpose(tf.reshape(tf.transpose(noise, perm=[0, 2, 1]), temp_shape), perm=[0, 2, 1, 3])

        temp_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant(np.array([1, self.K, 1], dtype=np.int32))], 0)
        non_stationary = tf.tile(self.non_stationary, temp_shape)

        return tf.square(tf.multiply(noise, non_stationary)) * self.g

    def call_6(self, inputs, training=False):  # correlated noise
        """
        Method to generate type 6 noise

        """
        return self.g * (tf.square(inputs))
