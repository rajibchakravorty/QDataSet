"""
Module to generate the dataset and store it
based on the simulation parameters passed as a dictionary
"""

from typing import Dict, Any, List

from numpy import array

from ..system_layers.quantum_ml_simulator import QuantumMLSimulator

from ..configurations.prepare_configs import get_default_configuration, get_custom_config


def create_simulator(
        simulation_parameters: Dict[str, Any],
        pulse_shape: str,
        distortion: bool
) -> QuantumMLSimulator:
    """Creates a simulator with specified parameters

    :param simulation_parameters: A dictionary with simulation parameters
    :param pulse_shape: Shape of the pulse; must be one of `Gaussian`, `Square` or `Zero`
    :param distortion: True if distortion to be added, False otherwise

    :returns: The simulator created from given parameters
    """
    return QuantumMLSimulator(
        simulation_parameters["evolution_time"],
        simulation_parameters["num_time_steps"],
        simulation_parameters["dynamic_operators"],
        simulation_parameters["static_operators"],
        simulation_parameters["noise_operators"],
        simulation_parameters["measurement_operators"],
        simulation_parameters["initial_states"],
        simulation_parameters["num_realizations"],
        pulse_shape,
        simulation_parameters["num_pulses"],
        distortion,
        simulation_parameters["noise_profile"])


def create_default_simulator(
        experiment_name: str,
        distortion: bool,
        pulse_shape: str,

) -> QuantumMLSimulator:
    """Create a Quantum ML simulator from a known experiment name

    :param experiment_name: A known experiment name; must be one of
    ['1q_X', '1q_X_N1Z', '1q_X_N2Z', '1q_X_N3Z',
     '1q_X_N4Z', '1q_XY', '1q_XY_N1X_N5Z', '1q_XY_N1X_N6Z',
     '1q_XY_N3X_N6Z', '2q_IX_XI_XX', '2q_IX_XI_XX_N1N5IZ_N1N5ZI',
     '2q_IX_XI_XX_N1N6IZ_N1N6ZI', '2q_IX_XI_N1N6IZ_N1N6ZI']
     :param distortion: Indicating if distortion would be added or not
     :param pulse_shape: Shape of the pulse; must one one of "Gaussian",
     "Square" or "Zero"

     :returns: A QML simulator prepared out of chosen configuration

     :raises ValueError: If the pulse_shape if one of the allowed shapes.
    """

    if pulse_shape not in ["Gaussian", "Square", "Zero"]:
        raise ValueError(
            "Pulse Shape is not known. Expected one of {}, found {}".format(
                ["Gaussian", "Square", "Zero"], pulse_shape
            )
        )
    simulation_parameters = get_default_configuration(experiment_name)

    return create_simulator(
        simulation_parameters=simulation_parameters,
        pulse_shape=pulse_shape,
        distortion=distortion
    )


def create_custom_simulator(
        evolution_time: float,
        num_time_steps: int,
        dimension: int,
        dynamic_operators: List[array],
        static_operators: List[array],
        noise_operators: List[array],
        measurement_operators: List[array],
        initial_states: List[array],
        num_realizations: int,
        num_pulses: int,
        noise_profile: List[str],
        distortion: bool,
        pulse_shape: str
) -> QuantumMLSimulator:

    """
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
    :param num_realizations: Number of noise realizations; defaults to 1
    :param num_pulses: Number of pulses per control sequence: defaults to 5
    :param noise_profile : The type of noise, a value chosen from
    ['Type 0','Type 1','Type 2','Type 4','Type 5','Type 6'];
    defaults to ['Type 0']
    :param distortion: Indicating if distortion would be added or not
    :param pulse_shape: Shape of the pulse; must one one of "Gaussian",
     "Square" or "Zero"

    :returns: A QML simulator created out of chosen parameter

    :raises ValueError: If the pulse_shape if one of the allowed shapes.
    """

    if pulse_shape not in ["Gaussian", "Square", "Zero"]:
        raise ValueError(
            "Pulse Shape is not known. Expected one of {}, found {}".format(
                ["Gaussian", "Square", "Zero"], pulse_shape
            )
        )
    simulation_parameters = get_custom_config(
        evolution_time=evolution_time,
        num_time_steps=num_time_steps,
        dimension=dimension,
        dynamic_operators=dynamic_operators,
        static_operators=static_operators,
        noise_operators=noise_operators,
        measurement_operators=measurement_operators,
        initial_states=initial_states,
        num_realizations=num_realizations,
        num_pulses=num_pulses,
        noise_profile=noise_profile
    )

    return create_simulator(
        simulation_parameters=simulation_parameters,
        pulse_shape=pulse_shape,
        distortion=distortion
    )
