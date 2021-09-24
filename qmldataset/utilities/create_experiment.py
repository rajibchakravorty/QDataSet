"""Create a single experiment with number of noise realizations
based on a simulator
"""
from typing import Dict, Any
from numpy import average, zeros

from ..system_layers.quantum_ml_simulator import QuantumMLSimulator


def run_experiment(
        simulator: QuantumMLSimulator
) -> Dict[str, Any]:
    """Run an experiment once and collect the simulation result

    :param simulator: Simulator object
    :returns: A dictionary containing the result of the experiment
    """

    simulation_results = simulator.simulate(
        zeros((1, 1)),
        batch_size=1)

    pulse_parameters, pulses, distorted_pulses, noise = simulation_results[0:4]
    hamitonian_0, \
    hamiltonian_1, \
    unitary_0, \
    _, unitary_i, \
    expectations = simulation_results[4:10]
    vector_obs = simulation_results[10:]

    result = {
        "sim_parameters": simulator.get_simulation_parameters(),
        "pulse_parameters": pulse_parameters[0, :],
        "time_range": simulator.time_range,
        "pulses": pulses[0, :],
        "distorted_pulses": distorted_pulses[0, :],
        "noise": noise[0, :],
        "H0": hamitonian_0[0, :],
        "H1": hamiltonian_1[0, :],
        "U0": unitary_0[0, :],
        "UI": unitary_i[0, :],
        "vo": [V[0, :] for V in vector_obs],
        "average_vo": [
            average(V[0, :], axis=1) for V in vector_obs],
        "expectations": expectations[0, :],
        "average_expectation": [average(expectations[0, :], axis=1)]
    }

    return result
