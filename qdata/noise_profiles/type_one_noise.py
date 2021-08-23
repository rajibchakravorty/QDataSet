"""
class for generating time-domain realizations of noise
"""

from numpy import (
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
    random,
    signal,
    shape,
    reverse,
    tile as tftile,
    transpose,
)


class TypeOneNoiseProfile():
    """Noise Layer definition

        total_duration      : Total duration of the input signal
        num_time_steps      : Number of time steps
        num_realization      : Number of realizations
   """

    def __init__(
            self, total_duration, num_time_steps, num_realization, **kwargs):

        # store class parameters
        self.total_duration = total_duration
        self.num_time_steps = num_time_steps
        self.num_realization = num_realization

        # define a vector of discreteized frequencies
        frequencies = fftfreq(num_time_steps) * num_time_steps / total_duration

        # define time step
        time_step = total_duration / num_time_steps

        alpha = 1
        s_zvalues = 1 * array(
            [(1 / (fq + 1) ** alpha) *
             (fq <= 15) + (1 / 16) *
             (fq > 15) + exp(-((fq - 30) ** 2) / 50) / 2
             for fq in frequencies[frequencies >= 0]])
        self.p_temporary = constant(
                tile(
                    reshape(
                        sqrt(s_zvalues * num_time_steps / time_step),
                        (1, 1, self.num_time_steps // 2)),
                    (1, self.num_realization, 1)),
                dtype=complex64)
        super().__init__(**kwargs)

    def call(self, inputs):  # PSD of 1/f + a bump
        """
        Method to generate type 1 and type 5 noise

        """
        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([1, 1], dtype=int32))], 0)
        p_temporary = tftile(self.p_temporary, temp_shape)

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([self.num_realization, self.num_time_steps // 2], dtype=int32))], 0)
        p_temporary = multiply(p_temporary, tfexp(
            2 * pi * 1j * cast(
                random.uniform(temp_shape, dtype=float32), complex64)))

        noise = math.real(
            signal.ifft(concat(
                [p_temporary, reverse(math.conj(p_temporary), axis=constant([2]))], 2)))
        noise = transpose(expand_dims(noise, axis=-1), perm=[0, 2, 1, 3])

        return noise
