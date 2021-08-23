"""
class for generating type 6 noise
"""

from tensorflow import (
    square,
)


class TypeSixNoiseProfile():
    """Noise Layer definition

        total_duration      : Total duration of the input signal
        num_time_steps      : Number of time steps
        num_realization      : Number of realizations

   """

    def __init__(
            self,
            total_duration,
            num_time_steps,
            num_realization,
            **kwargs):

        self.factor = kwargs.get('factor', 0.3)
        if 'factor' in kwargs:
            del kwargs['factor']

        # store class parameters
        self.total_duration = total_duration
        self.num_time_steps = num_time_steps
        self.num_realization = num_realization

        super().__init__(**kwargs)

    def call(self, inputs):  # correlated noise
        """
        Method to generate type 6 noise

        """
        return self.factor * (square(inputs))
