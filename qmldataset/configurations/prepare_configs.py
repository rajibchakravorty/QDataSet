"""Create simulator parameter configuration for default experiments
"""

from typing import Dict, Any, List

from numpy import array
from .default_config_names import default_configs


def get_default_configuration(
        config_name: str,
) -> Dict[str, Any]:
    """Creates the configuration parameters of default experiments
    :param config_name: An experiment configuration; must be one of
    ['1q_X', '1q_X_N1Z', '1q_X_N2Z', '1q_X_N3Z',
     '1q_X_N4Z', '1q_XY', '1q_XY_N1X_N5Z', '1q_XY_N1X_N6Z',
     '1q_XY_N3X_N6Z', '2q_IX_XI_XX', '2q_IX_XI_XX_N1N5IZ_N1N5ZI',
     '2q_IX_XI_XX_N1N6IZ_N1N6ZI', '2q_IX_XI_N1N6IZ_N1N6ZI']

    :returns: A dictionary containing the parameters of the chosen configuration
    :raises ValueError: If experiment config is not one of the default ones.
    """
    if config_name not in default_configs:
        raise ValueError(
            '{} is not known configuration. Please supply one of {}'.format(
                config_name, list(default_configs.keys())))

    configuration = default_configs[config_name]

    return get_custom_config(
        configuration.evolution_time,
        configuration.num_time_steps,
        configuration.dimension,
        configuration.dynamic_operators,
        configuration.static_operators,
        configuration.noise_operators,
        configuration.measurement_operators,
        configuration.initial_states,
        configuration.num_pulses,
        configuration.noise_profile
    )


def get_custom_config(
        evolution_time: float,
        num_time_steps: int,
        dimension: int,
        dynamic_operators: List[array],
        static_operators: List[array],
        noise_operators: List[array],
        measurement_operators: List[array],
        initial_states: List[array],
        num_pulses: int,
        noise_profile: List[str]
) -> Dict[str, Any]:
    """Creates the configuration of a custom experiment

    :param evolution_time : Evolution time
    :param num_time_steps: Number of time steps
    :param dimension: Dimension of the system
    :param dynamic_operators: A list of arrays that represent the
    terms of the control Hamiltonian (that depend on pulses)
    :param static_operators: A list of arrays that represent
    the terms of the drifting Hamiltonian (that are constant)
    :param noise_operators: A list of arrays that represent
    the terms of the classical noise Hamiltonians
    :param measurement_operators: A list of arrays that represent
    measurement operators
    :param initial_states: A list of arrays representing initial states
    :param num_pulses: Number of pulses per control sequence: defaults to 5
    :param noise_profile : The type of noise, a value chosen from
    ['Type 0','Type 1','Type 2','Type 4','Type 5','Type 6'];
    defaults to ['Type 0']

    :returns: A dictionary containing the parameters of the custom configuration
    :raises ValueError: If experiment config is not one of the default ones.
    """

    return {
        "evolution_time": evolution_time,
        "num_time_steps": num_time_steps,
        "dimension": dimension,
        "dynamic_operators": dynamic_operators,
        "static_operators": static_operators,
        "noise_operators": noise_operators,
        "measurement_operators": measurement_operators,
        "initial_states": initial_states,
        "num_pulses": num_pulses,
        "noise_profile": noise_profile
    }
