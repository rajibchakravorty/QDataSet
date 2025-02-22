"""Method to create the required noise profile
"""

from .base_noise_profile import BaseNoiseProfile
from .type_zero_noise import TypeZeroNoiseProfile
from .type_one_noise import TypeOneNoiseProfile
from .type_two_noise import TypeTwoNoiseProfile
from .type_three_noise import TypeThreeNoiseProfile
from .type_four_noise import TypeFourNoiseProfile
from .type_five_noise import TypeFiveNoiseProfile
from .type_six_noise import TypeSixNoiseProfile


def create_noise_profile(
        noise_profile_type: str,
        total_duration: float,
        num_time_steps: float,
        num_realization: float) -> BaseNoiseProfile:
    """Creates a noise profile of specified type

    :param noise_profile_type: One of 'Type 0',
    'Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6'

    :param total_duration: Total duration of the input signal
    :param num_time_steps: Number of time steps
    :param num_realization: Number of realizations

    :returns: A noise object of specified profile type
    """

    noise_profile = None

    if noise_profile_type.lower() == 'type 0':
        noise_profile = TypeZeroNoiseProfile(
            total_duration, num_time_steps, num_realization)

    if noise_profile_type.lower() == 'type 1':
        noise_profile = TypeOneNoiseProfile(
            total_duration, num_time_steps, num_realization)

    if noise_profile_type.lower() == 'type 2':
        noise_profile = TypeTwoNoiseProfile(
            total_duration, num_time_steps, num_realization)

    if noise_profile_type.lower() == 'type 3':
        noise_profile = TypeThreeNoiseProfile(
            total_duration, num_time_steps, num_realization)

    if noise_profile_type.lower() == 'type 4':
        noise_profile = TypeFourNoiseProfile(
            total_duration, num_time_steps, num_realization)

    if noise_profile_type.lower() == 'type 5':
        noise_profile = TypeFiveNoiseProfile(
            total_duration, num_time_steps, num_realization)

    if noise_profile_type.lower() == 'type 6':
        noise_profile = TypeSixNoiseProfile(
            total_duration, num_time_steps, num_realization)

    return noise_profile
