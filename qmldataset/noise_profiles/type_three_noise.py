"""
class for generating type 3 noise
"""

from numpy import (  # pylint: disable=redefined-builtin
    abs,
    array,
    int32,
    reshape
)

from tensorflow import (
    concat,
    constant,
    float32,
    multiply,
    nn,
    ones,
    random,
    shape,
    tile as tftile,
    transpose
)

from .base_noise_profile import BaseNoiseProfile

class TypeThreeNoiseProfile(BaseNoiseProfile):
    """Type three Noise definition

    :param total_duration: Total duration of the input signal
    :param num_time_steps: Number of time steps
    :param num_realization: Number of realizations
    :param factor: Multiplication factor

   """

    def __init__(
            self,
            total_duration: float,
            num_time_steps: int,
            num_realization: int,
            factor: float = 0.2):

        super().__init__(
            total_duration=total_duration,
            num_time_steps=num_time_steps,
            num_realization=num_realization
        )
        self.factor = factor

        time_range = [
            (0.5 * total_duration / num_time_steps) +
            (j * total_duration / num_time_steps) for j in range(num_time_steps)]
        self.color = ones(
            [self.num_time_steps // 4, 1, 1], dtype=float32)
        self.non_stationary = constant(
            reshape(
                1 - (abs(array(time_range) - 0.5 * total_duration) * 2),
                (1, num_time_steps, 1, 1)), dtype=float32)

    def call(self, inputs):  # Colored Gaussian Non-stationary Noise
        """
        Method to generate type 3 noise
        """
        temp_shape = concat(
            [self.num_realization * shape(inputs)[0:1],
             constant(
                 array(
                     [self.num_time_steps + (self.num_time_steps // 4) - 1, 1],
                     dtype=int32))], 0)
        noise = self.factor * nn.convolution(
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
