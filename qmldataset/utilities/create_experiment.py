"""Create a single experiment with number of noise realizations
based on a simulator
"""
from typing import Dict, Any, List
from numpy import average, zeros

from ..system_layers.quantum_ml_simulator import QuantumMLSimulator


def run_experiment(
        simulator: QuantumMLSimulator,
        num_examples: int = 1,
) -> List[Dict[str, Any]]:
    """Run the simulation and save the result

    :param num_examples: Number of configurations to create
    :param simulator: Simulator object

    :returns: A list of dictionary containing the result of the experiment; length of the
    list is equal to num_examples
    """

    simulation_results = simulator.simulate(
        zeros((num_examples, 1)),
        batch_size=num_examples)

    pulse_parameters, pulses, distorted_pulses, noise = simulation_results[0:4]
    hamitonian_0, \
    hamiltonian_1, \
    unitary_0, \
    _, unitary_i, \
    expectations = simulation_results[4:10]
    vector_obs = simulation_results[10:]

    experiment_result = []
    for idx_ex in range(num_examples):
        result = {
            "sim_parameters": simulator.get_simulation_parameters(),
            "pulse_parameters": pulse_parameters[idx_ex:idx_ex + 1, :],
            "time_range": simulator.time_range,
            "pulses": pulses[idx_ex:idx_ex + 1, :],
            "distorted_pulses": distorted_pulses[idx_ex:idx_ex + 1, :],
            "noise": noise[idx_ex:idx_ex + 1, :],
            "H0": hamitonian_0[idx_ex:idx_ex + 1, :],
            "H1": hamiltonian_1[idx_ex:idx_ex + 1, :],
            "U0": unitary_0[idx_ex:idx_ex + 1, :],
            "UI": unitary_i[idx_ex:idx_ex + 1, :],
            "vo": [V[idx_ex:idx_ex + 1, :] for V in vector_obs],
            "average_vo": [
                average(V[idx_ex:idx_ex + 1, :], axis=1) for V in vector_obs],
            "expectations": expectations[idx_ex:idx_ex + 1, :],
            "average_expectation": average(expectations[idx_ex:idx_ex + 1, :], axis=1)
        }
        experiment_result.append(result)

    return experiment_result
