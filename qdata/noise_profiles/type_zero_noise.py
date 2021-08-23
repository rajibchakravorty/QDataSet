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


class TypeZeroNoiseProfile():
    """Noise Layer definition

        total_duration      : Total duration of the input signal
        num_time_steps      : Number of time steps
        num_realization      : Number of realizations
        profile: Type of noise

   """

    def __init__(
            self, total_duration, num_time_steps, num_realization, **kwargs):

        # store class parameters
        self.total_duration = total_duration
        self.num_time_steps = num_time_steps
        self.num_realization = num_realization

        super().__init__(**kwargs)

    def call(self, inputs):  # No noise
        """
        Method to generate type 0 noise
        """
        temp_shape = concat(
            [shape(inputs)[0:1],
             constant(
                 array([self.num_time_steps, self.num_realization, 1], dtype=int32))], 0)
        return zeros(temp_shape, dtype=float32)
