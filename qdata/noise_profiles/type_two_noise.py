"""
class for generating type 2 noise
"""

from numpy import (
    array,
    int32,
    reshape
)

from tensorflow import (
    concat,
    constant,
    float32,
    nn,
    ones,
    random,
    shape,
    transpose
)

from .base_noise_profile import BaseNoiseProfile


class TypeTwoNoiseProfile(BaseNoiseProfile):
    """Type one Noise definition

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
            factor: float = 0.1):

        super().__init__(
            total_duration=total_duration,
            num_time_steps=num_time_steps,
            num_realization=num_realization
        )
        self.factor = factor

        self.color = ones([self.num_time_steps // 4, 1, 1], dtype=float32)

    def call(self, inputs):  # Colored Gaussian Stationary Noise
        """
        Method to generate type 2 noise

        """
        temp_shape = concat(
            [self.num_realization * shape(inputs)[0:1],
             constant(array([self.num_time_steps + (self.num_time_steps // 4) - 1, 1],
                            dtype=int32))], 0)
        noise = self.factor * nn.convolution(
            input=random.normal(temp_shape), filters=self.color, padding="VALID")

        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(array([self.num_realization, self.num_time_steps, 1], dtype=int32))], 0)
        noise = transpose(
            reshape(transpose(noise, perm=[0, 2, 1]), temp_shape), perm=[0, 2, 1, 3])

        return noise
