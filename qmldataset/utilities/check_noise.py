"""Checking noise
"""

from typing import Dict, Any
from numpy import average, eye, zeros
from numpy.linalg import norm

from ..system_layers import QuantumTFSimulator


def check_noise(simulation_parameters: Dict[str, Any], dimension: int):
    """
    This function calculates the coherence measurements to check the noise behaviour,
    based on the simulation parameters passed as a dictionary

    :param simulation_parameters: Simulation Parameters
    :param dimension: Dimension of the system
    """
    simulator = QuantumTFSimulator(
        simulation_parameters["evolution_time"],
        simulation_parameters["num_time_steps"],
        simulation_parameters["dynamic_operators"],
        simulation_parameters["static_operators"],
        simulation_parameters["noise_operators"],
        simulation_parameters["measurement_operators"],
        simulation_parameters["initial_states"],
        simulation_parameters["num_realizations"],
        simulation_parameters["pulse_shape"],
        simulation_parameters["num_pulses"],
        False,
        simulation_parameters["noise_profile"])
    # 3) Run the simulator and collect the results
    print("Running the simulation\n")
    simulation_results = simulator.simulate(zeros((1,)), batch_size=1)
    expectations = simulation_results[9]
    obs_vector = simulation_results[10:]
    obs_vector = [average(V, axis=1) for V in obs_vector]
    print("Analyzing results\n")
    print("Measurement are:")
    print(average(expectations, axis=1))
    print("The Vo operators are:")
    print(obs_vector)
    print("The distance measures are:")
    print([norm(vector[0, :] - eye(dimension), 2) for vector in obs_vector])
