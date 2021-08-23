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


class TypeTwoNoiseProfile():
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

        self.factor = kwargs.get('factor', 0.1)
        if 'factor' in kwargs:
            del kwargs['factor']

        # store class parameters
        self.total_duration = total_duration
        self.num_time_steps = num_time_steps
        self.num_realization = num_realization

        self.color = ones([self.num_time_steps // 4, 1, 1], dtype=float32)

        super().__init__(**kwargs)

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
