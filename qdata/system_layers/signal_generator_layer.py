"""Generator of Control pulse parameter
"""

from numpy import (
    ones,
    reshape,
    triu_indices
)
from tensorflow import (
    constant,
    float32,
    Tensor
)
from tensorflow.keras import layers


class SignalGenerator(layers.Layer):
    """Generates a sequence a sequence of control pulse parameters

    :param total_time: Total time of evolution
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence
    :param waveform: Waveform shape can either be "Gaussian", "Square", or "Zero".
    defaults to Gaussian
    """
    def __init__(self,
                 total_time,
                 num_time_steps,
                 max_control_pulse,
                 waveform="Gaussian", **kwargs):

        # store the parameters
        self.n_max = max_control_pulse
        self.T = total_time
        self.M = num_time_steps
        self.time_range = constant(
            reshape([(0.5 * total_time / num_time_steps) +
                     (j * total_time / num_time_steps) for j in range(num_time_steps)],
                    (1, num_time_steps, 1, 1)), dtype=float32)

        if waveform == "Gaussian":
            self.call = self.call_Gaussian
        elif waveform == "Square":
            self.call = self.call_Square
        else:
            self.call = self.call_0

        # define the constant parmaters to shift the pulses correctly
        self.pulse_width = (0.5 * self.T / self.n_max)

        self.a_matrix = ones((self.n_max, self.n_max))
        self.a_matrix[triu_indices(self.n_max, 1)] = 0
        self.a_matrix = constant(
            reshape(self.a_matrix, (1, self.n_max, self.n_max)), dtype=float32)

        self.b_matrix = reshape(
            [idx + 0.5 for idx in range(self.n_max)],
            (1, self.n_max, 1)) * self.pulse_width
        self.b_matrix = constant(self.b_matrix, dtype=float32)

        # we must call thus function for any tensorflow custom layer
        super().__init__(**kwargs)

    def call_Square(self, inputs: Tensor):
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
