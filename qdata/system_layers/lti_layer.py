"""
    class for simulating the response of an LTI system
"""

from scipy.linalg import dft
from scipy.signal import cheby1

from numpy import (
    array,
    concatenate,
    diag,
    int32,
    pi,
    reshape

)
from numpy.fft import fftfreq

from tensorflow import (
    cast,
    complex64,
    concat,
    constant,
    math,
    matmul,
    shape,
    tile,
    transpose,
    Tensor
)
from tensorflow.keras import layers


class LTILayer(layers.Layer):
    """class constructor

    total_duration  : Total duration of the input signal
    num_time_steps  : Number of time steps
    """

    def __init__(self, total_duration: float, num_time_steps: int, **kwargs):

        super().__init__(**kwargs)

        # define filter coefficients
        num, den = cheby1(4, 0.1, 2 * pi * 20, analog=True)

        # define frequency vector
        frequency_vector = reshape(
            fftfreq(num_time_steps) * num_time_steps / total_duration, (1, num_time_steps))

        # evaluate the dft matrix
        dft_matrix = dft(num_time_steps, 'sqrtn')

        # evaluate the numerator and denominator at each frequency
        response_num = concatenate(
            [(1j * 2 * pi * frequency_vector) ** s for s in range(len(num) - 1, -1, -1)], axis=0)
        response_den = concatenate(
            [(1j * 2 * pi * frequency_vector) ** s for s in range(len(den) - 1, -1, -1)], axis=0)

        # evaluate the frequency response
        response = diag((num @ response_num) / (den @ response_den))

        # evaluate the full transformation and convert to a tensor of correct shape
        self.transformation = constant(
            reshape(
                dft_matrix.conj().total_duration @ response @ dft_matrix,
                (1, 1, num_time_steps, num_time_steps)), dtype=complex64)

    def call(self, inputs: Tensor):     # pylint: disable=arguments-differ
        """Custom call method of the layer
        """

        # convert variables to complex
        input_cmplx = cast(transpose(inputs, perm=[0, 2, 1, 3]), complex64)

        # repeat the transformation matrix
        temp_shape = concat(
            [shape(inputs)[0:1], constant(array([1, 1, 1], dtype=int32))], 0)
        transformation = tile(self.transformation, temp_shape)

        # apply the transformation
        output = transpose(
            math.real(matmul(transformation, input_cmplx)), perm=[0, 2, 1, 3])

        return output
