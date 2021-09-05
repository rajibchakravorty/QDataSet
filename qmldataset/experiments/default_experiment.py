"""Create one of the default experiments, save experiment results
"""

from .custom_experiment import run_custom_experiment
from .default_config_names import default_configs


def run_default_experiment(
        experiment_config: str,
        pulse_shape: str,
        num_examples: int,
        batch_size: int,
        output_location: str
):
    """
    :param experiment_config: An experiment configuration; must be one of
    ['G_1q_X', 'G_1q_XZ_N1']
    :param pulse_shape: Shape of the pulse; One of 'Gaussian', 'Square, 'Zero'
    :param num_examples: Number of experiments to create
    :param batch_size: Size of each batch
    :param output_location: The absolute path of the temporary location to save intermediate
    results

    :raises ValueError: If experiment config is not one of the default ones.
    """
    if experiment_config not in default_configs:
        raise ValueError('{} is not knows configuration'.format(experiment_config))

    configuration = default_configs[experiment_config]
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
        pulse_shape=pulse_shape,
        num_pulses=configuration.num_pulses,
        noise_profile=configuration.noise_profile,
        experiment_name=configuration.name,
        num_examples=num_examples,
        batch_size=batch_size,
        output_location=output_location
    )
