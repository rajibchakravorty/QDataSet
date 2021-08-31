"""
class for generating time-domain realizations of noise
"""

from tensorflow import Tensor
from tensorflow.keras import layers

from ..noise_profiles import create_noise_profile


class NoiseLayer(layers.Layer):
    """Noise Layer definition

        total_duration      : Total duration of the input signal
        num_time_steps      : Number of time steps
        num_realization      : Number of realizations
        profile: Type of noise; One of 'Type 0', 'Type 1', 'Type 2',
        'Type 3', 'Type 4', 'Type 5', 'Type 6'

   """

    def __init__(
            self,
            total_duration: float,
            num_time_steps: int,
            num_realization: int,
            profile: str,
            **kwargs):

        super().__init__(**kwargs)

        self.noise = create_noise_profile(
            profile,
            total_duration,
            num_time_steps,
            num_realization
        )

    def call(self, inputs: Tensor):  # pylint: disable=arguments-differ
        """Custom call method of the layer
        """

        return self.noise.call(inputs)
