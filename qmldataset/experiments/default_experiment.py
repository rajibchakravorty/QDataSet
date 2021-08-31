"""Create one of the default experiments, save experiment results
"""

from . import config_G1q_X
from .custom_experiment import run_custom_experiment


def run_default_experiment(
        experiment_config: str,
        num_examples: int,
        batch_size: int,
        output_location: str
):
    """
    :param experiment_config: An experiment configuration; must be one of
    ['G_1q_X']
    :param num_examples: Number of experiments to create
    :param batch_size: Size of each batch
    :param output_location: The absolute path of the temporary location to save intermediate
    results
    """
    if experiment_config == "G_1q_X":
        configuration = config_G1q_X

    run_custom_experiment(
        evolution_time=configuration.evolution_time,
        num_time_steps=configuration.num_time_steps,
        dimension=configuration.dimension,
        dynamic_operators=configuration.dynamic_operators,
        static_operators=configuration.static_operators,
        noise_operators=configuration.noise_operators,
        measurement_operators=configuration.measurement_operators,
        initial_states=configuration.initial_states,
        num_realizations=configuration.num_realizations,
        pulse_shape=configuration.pulse_shape,
        num_pulses=configuration.num_pulses,
        noise_profile=configuration.noise_profile,
        experiment_name=configuration.name,
        num_examples=num_examples,
        batch_size=batch_size,
        output_location=output_location
    )
