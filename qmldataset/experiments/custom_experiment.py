"""Create a experiment with custom simulation parameters
"""

from typing import List
from numpy import array

from .. utilities.check_noise import check_noise
from ..utilities.simulate import simulate


def run_custom_experiment(
        evolution_time: float,
        num_time_steps: int,
        dimension: int,
        dynamic_operators: List[array],
        static_operators: List[array],
        noise_operators: List[array],
        measurement_operators: List[array],
        initial_states: List[array],
        num_realizations: int,
        pulse_shape: str,
        num_pulses: int,
        noise_profile: str,
        experiment_name: str,
        num_examples: int,
        batch_size: int,
        output_location: str,
):
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
    :param pulse_shape: The type of waveform [either "Zero", "Square", or "Gaussian"];
    defaults to Gaussian
    :param num_pulses: Number of pulses per control sequence: defaults to 5
    :param noise_profile : The type of noise, a value chosen from
    ['Type 0','Type 1','Type 2','Type 4','Type 5','Type 6'];
    defaults to 'Type 0'
    :param experiment_name: A descriptive name of the simulation (used to create pickle file
    with results
    :param num_examples: Number of experiments to create
    :param batch_size: Size of each batch
    :param output_location: The absolute path of the temporary location to save intermediate
    results
    """

    simulator_parameters = {
        "evolution_time": evolution_time,
        "num_time_steps": num_time_steps,
        "dynamic_operators": dynamic_operators,
        "static_operators": static_operators,
        "noise_operators": noise_operators,
        "measurement_operators": measurement_operators,
        "initial_states": initial_states,
        "num_realizations": num_realizations,
        "num_pulses": num_pulses,
        "pulse_shape": pulse_shape,
        "noise_profile": noise_profile
    }

    check_noise(simulator_parameters, dimension)
    simulate(
        simulation_parameters=simulator_parameters,
        simulation_name=experiment_name,
        num_examples=num_examples,
        batch_size=batch_size,
        output_location=output_location
    )
