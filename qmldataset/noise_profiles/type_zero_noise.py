"""Defines a profile of zero noise
"""

from numpy import (
    array,
    int32,
)

from tensorflow import (
    concat,
    constant,
    float32,
    shape,
    zeros
)

from .base_noise_profile import BaseNoiseProfile


class TypeZeroNoiseProfile(BaseNoiseProfile):
    """Type Zero Noise definition

    :param total_duration      : Total duration of the input signal
    :param num_time_steps      : Number of time steps
    :param num_realization      : Number of realizations

   """

    def __init__(
            self,
            total_duration: float,
            num_time_steps: int,
            num_realization: int):

        super().__init__(
            total_duration=total_duration,
            num_time_steps=num_time_steps,
            num_realization=num_realization
        )

    def call(self, inputs):  # No noise
        """
        Method to generate type 0 noise
        """
        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(
                 array([self.num_time_steps, self.num_realization, 1], dtype=int32))], 0)
        return zeros(temp_shape, dtype=float32)
