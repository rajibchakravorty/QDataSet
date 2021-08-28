"""Base noise profile class
"""


class BaseNoiseProfile():
    """Base Noise Layer definition

    :param total_duration : Total duration of the input signal
    :param num_time_steps : Number of time steps
    :param num_realization : Number of realizations

   """

    def __init__(
            self,
            total_duration: float,
            num_time_steps: float,
            num_realization: float):

        # store class parameters
        self.total_duration = total_duration
        self.num_time_steps = num_time_steps
        self.num_realization = num_realization

    def call(self, inputs):  # PSD of 1/f + a bump
        """
        Method to generate type 1 and type 5 noise
        """
        pass    # pylint: disable=unnecessary-pass
