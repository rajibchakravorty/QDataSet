"""Generator of Control pulse parameter
"""

from tensorflow import Tensor
from tensorflow.keras import layers

from ..control_pulse import create_control_pulses


class SignalGenerator(layers.Layer):
    """Generates a sequence a sequence of control pulse parameters

    :param total_time: Total time of evolution
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence
    :param waveform: Waveform shape can either be "Gaussian", "Square", or "Zero".
    defaults to Gaussian
    """
    def __init__(self,
                 total_time,
                 num_time_steps,
                 max_control_pulse,
                 waveform="Gaussian", **kwargs):

        # store the parameters
        self.max_control_pulse = max_control_pulse
        self.total_time = total_time
        self.num_time_steps = num_time_steps

        self.waveform = waveform

        # we must call thus function for any tensorflow custom layer
        super().__init__(**kwargs)

    def call(self, inputs: Tensor):     # pylint: disable=arguments-differ
        """Custom call method for the layer
        """

        return create_control_pulses(
            max_control_pulse=self.max_control_pulse,
            total_time=self.total_time,
            num_time_steps=self.num_time_steps,
            inputs=inputs,
            waveform=self.waveform
        )
