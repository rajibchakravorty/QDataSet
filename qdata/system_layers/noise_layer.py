"""
class for generating time-domain realizations of noise
"""

from numpy import ( # pylint: disable=redefined-builtin
    abs,
    array,
    exp,
    int32,
    pi,
    reshape,
    sqrt,
    tile,
)
from numpy.fft import fftfreq

from tensorflow import (
    cast,
    complex64,
    concat,
    constant,
    exp as tfexp,
    expand_dims,
    float32,
    math,
    multiply,
    nn,
    ones,
    random,
    signal,
    square,
    shape,
    reverse,
    tile as tftile,
    transpose,
    zeros,
)
from tensorflow.keras import layers


class NoiseLayer(layers.Layer):
    """Noise Layer definition

        total_duration      : Total duration of the input signal
        num_time_steps      : Number of time steps
        num_realization      : Number of realizations
        profile: Type of noise

   """

    def __init__(
            self, total_duration, num_time_steps, num_realization, profile, **kwargs):

        super().__init__(**kwargs)

        # store class parameters
        self.total_duration = total_duration
        self.num_time_steps = num_time_steps
        self.num_realization = num_realization

        # define a vector of discriteized frequencies
        f = fftfreq(num_time_steps) * num_time_steps / total_duration

        # define time step
        time_step = total_duration / num_time_steps

        # check the noise type, initialize required variables and define the correct "call" method
        if profile == 0:  # No noise
            self.call = self.call_0
        elif profile == 1:  # PSD of 1/f + a bump
            alpha = 1
            S_Z = 1 * array(
                [(1 / (fq + 1) ** alpha) *
                 (fq <= 15) + (1 / 16) *
                 (fq > 15) + exp(-((fq - 30) ** 2) / 50) / 2 for fq
                 in f[f >= 0]])
            self.P_temp = constant(
                tile(
                    reshape(
                        sqrt(S_Z * num_time_steps / time_step),
                        (1, 1, self.num_time_steps // 2)),
                    (1, self.num_realization, 1)),
                dtype=complex64)
            self.call = self.call_1

        elif profile == 2:  # Colored Gaussian Stationary Noise
            self.g = 0.1
            self.color = ones([self.num_time_steps // 4, 1, 1], dtype=float32)
            self.call = self.call_2
        elif profile == 3:  # Colored Gaussian Non-stationary Noise
            time_range = [
                (0.5 * total_duration / num_time_steps) +
                (j * total_duration / num_time_steps) for j in range(num_time_steps)]
            self.g = 0.2
            self.color = ones(
                [self.num_time_steps // 4, 1, 1], dtype=float32)
            self.non_stationary = constant(
                reshape(
                    1 - (abs(array(time_range) - 0.5 * total_duration) * 2),
                    (1, num_time_steps, 1, 1)), dtype=float32)
            self.call = self.call_3
        elif profile == 4:  # Colored Non-Gaussian Non-stationary Noise
            time_range = [
                (0.5 * total_duration / num_time_steps) +
                (j * total_duration / num_time_steps) for j in range(num_time_steps)]
            self.g = 0.01
            self.color = ones([self.num_time_steps // 4, 1, 1], dtype=float32)
            self.non_stationary = constant(
                reshape(
                    1 - (abs(array(time_range) - 0.5 * total_duration) * 2),
                    (1, num_time_steps, 1, 1)), dtype=float32)
            self.call = self.call_4
        elif profile == 5:  # PSD of 1/f
            alpha = 1
            S_Z = 1 * array([(1 / (fq + 1) ** alpha) for fq in f[f >= 0]])
            self.P_temp = constant(
                tile(reshape(sqrt(S_Z * num_time_steps / time_step),
                             (1, 1, self.num_time_steps // 2)), (1, self.num_realization, 1)),
                dtype=complex64)
            self.call = self.call_1
        elif profile == 6:  # correlated noise
            self.g = 0.3
            self.call = self.call_6

    def call_0(self, inputs, training=False):  # No noise
        """
        Method to generate type 0 noise

        """
        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([self.num_time_steps, self.num_realization, 1], dtype=int32))], 0)
        return zeros(temp_shape, dtype=float32)

    def call_1(self, inputs, training=False):  # PSD of 1/f + a bump
        """
        Method to generate type 1 and type 5 noise

        """
        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([1, 1], dtype=int32))], 0)
        P_temp = tftile(self.P_temp, temp_shape)

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([self.num_realization, self.num_time_steps // 2], dtype=int32))], 0)
        P_temp = multiply(P_temp, tfexp(
            2 * pi * 1j * cast(
                random.uniform(temp_shape, dtype=float32), complex64)))

        noise = math.real(
            signal.ifft(concat(
                [P_temp, reverse(math.conj(P_temp), axis=constant([2]))], 2)))
        noise = transpose(expand_dims(noise, axis=-1), perm=[0, 2, 1, 3])

        return noise

    def call_2(self, inputs, training=False):  # Colored Gaussian Stationary Noise
        """
        Method to generate type 2 noise

        """
        temp_shape = concat(
            [self.num_realization * shape(inputs)[0:1],
             constant(array([self.num_time_steps + (self.num_time_steps // 4) - 1, 1],
                            dtype=int32))], 0)
        noise = self.g * nn.convolution(
            input=random.normal(temp_shape), filters=self.color, padding="VALID")

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([self.num_realization, self.num_time_steps, 1], dtype=int32))], 0)
        noise = transpose(
            reshape(transpose(noise, perm=[0, 2, 1]), temp_shape), perm=[0, 2, 1, 3])

        return noise

    def call_3(self, inputs, training=False):  # Colored Gaussian Non-stationary Noise
        """
        Method to generate type 3 noise

        """
        temp_shape = concat(
            [self.num_realization * shape(inputs)[0:1],
             constant(
                 array(
                     [self.num_time_steps + (self.num_time_steps // 4) - 1, 1],
                     dtype=int32))], 0)
        noise = self.g * nn.convolution(
            input=random.normal(temp_shape), filters=self.color, padding="VALID")

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([self.num_realization, self.num_time_steps, 1], dtype=int32))], 0)
        noise = transpose(
            reshape(transpose(noise, perm=[0, 2, 1]), temp_shape), perm=[0, 2, 1, 3])

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([1, self.num_realization, 1], dtype=int32))], 0)
        non_stationary = tftile(self.non_stationary, temp_shape)

        return multiply(noise, non_stationary)

    def call_4(self, inputs, training=False):  # Colored Gaussian Non-stationary Noise
        """
        Method to generate type 4 noise

        """

        temp_shape = concat(
            [self.num_realization * shape(inputs)[0:1],
             constant(
                 array(
                     [self.num_time_steps + (self.num_time_steps // 4) - 1, 1], dtype=int32))], 0)
        noise = nn.convolution(
            input=random.normal(temp_shape), filters=self.color, padding="VALID")

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([self.num_realization, self.num_time_steps, 1], dtype=int32))], 0)
        noise = transpose(
            reshape(transpose(noise, perm=[0, 2, 1]), temp_shape), perm=[0, 2, 1, 3])

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([1, self.num_realization, 1], dtype=int32))], 0)
        non_stationary = tile(self.non_stationary, temp_shape)

        return square(multiply(noise, non_stationary)) * self.g

    def call_6(self, inputs, training=False):  # correlated noise
        """
        Method to generate type 6 noise

        """
        return self.g * (square(inputs))
