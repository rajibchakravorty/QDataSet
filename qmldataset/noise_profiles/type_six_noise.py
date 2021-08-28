"""
class for generating type 6 noise
"""

from tensorflow import (
    square,
)

from .base_noise_profile import BaseNoiseProfile


class TypeSixNoiseProfile(BaseNoiseProfile):
    """Type six Noise definition

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
            factor: float = 0.3):

        super().__init__(
            total_duration=total_duration,
            num_time_steps=num_time_steps,
            num_realization=num_realization
        )
        self.factor = factor

    def call(self, inputs):  # correlated noise
        """
        Method to generate type 6 noise

        """
        return self.factor * (square(inputs))
