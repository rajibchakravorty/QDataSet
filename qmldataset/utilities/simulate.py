"""
Module to generate the dataset and store it
based on the simulation parameters passed as a dictionary
"""

from typing import Dict, Any
from os import remove
from os.path import join
import time
import zipfile
import pickle

import numpy as np

from ..system_layers import QuantumTFSimulator


def create_simulation(
        simulation_parameters: Dict[str, Any],
        distortion: bool
) -> QuantumTFSimulator:
    """Creates a simulator with specified parameters

    :param simulation_parameters: A dictionary with simulation parameters
    :param distortion: True if distortion to be added, False otherwise

    :returns: The simulator created from given parameters
    """
    return QuantumTFSimulator(
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
        distortion,
        simulation_parameters["noise_profile"])


def save_simulation_result(
        num_examples : int,
        batch_size: int,
        distortion: bool,
        simulator: QuantumTFSimulator,
        simulation_name: str,
        zip_file: str,
        temp_location: str
):
    """Run the simulation and save the result

    :param num_examples: Number of experiments to create
    :param batch_size: Size of each batch
    :param distortion: True if the simulator is for distorted pulse, False otherwise
    :param simulator: Simulator object
    :param simulation_name: A descriptive name of the simulation (used to create pickle file
    with results
    :param zip_file: The absolute path of the zip file to save the results
    :param temp_location: The absolute path of the temporary location to save intermediate
    results
    """
    for idx_batch in range(num_examples // batch_size):
        ###########################################################################
        print("Simulating and storing batch %d\n" % idx_batch)
        start = time.time()
        simulation_results = simulator.simulate(
            np.zeros((batch_size, 1)),
            batch_size=batch_size)

        elapsed_time = time.time() - start
        pulse_parameters, pulses, distorted_pulses, noise = simulation_results[0:4]
        hamitonian_0, \
        hamiltonian_1, \
        unitary_0, \
        _, unitary_i, \
        expectations = simulation_results[4:10]
        vector_obs = simulation_results[10:]
        ###########################################################################
        # 4) Save the results in an external file and zip everything
        for idx_ex in range(batch_size):
            results = {"sim_parameters": simulator.get_simulation_parameters(),
                       "elapsed_time": elapsed_time,
                       "pulse_parameters": pulse_parameters[idx_ex:idx_ex + 1, :],
                       "time_range": simulator.time_range,
                       "pulses": pulses[idx_ex:idx_ex + 1, :],
                       "distorted_pulses": (
                           pulses[idx_ex:idx_ex + 1, :] if not distortion
                           else distorted_pulses[idx_ex:idx_ex+1, :]),
                       "expectations": np.average(expectations[idx_ex:idx_ex + 1, :], axis=1),
                       "Vo_operator": [
                           np.average(V[idx_ex:idx_ex + 1, :], axis=1) for V in vector_obs],
                       "noise": noise[idx_ex:idx_ex + 1, :],
                       "H0": hamitonian_0[idx_ex:idx_ex + 1, :],
                       "H1": hamiltonian_1[idx_ex:idx_ex + 1, :],
                       "U0": unitary_0[idx_ex:idx_ex + 1, :],
                       "UI": unitary_i[idx_ex:idx_ex + 1, :],
                       "Vo": [V[idx_ex:idx_ex + 1, :] for V in vector_obs],
                       "Eo": expectations[idx_ex:idx_ex + 1, :]
                       }
            # open a pickle file
            result_file_name = join(
                temp_location, '{}_ex_{}'.format(
                    simulation_name,
                    idx_ex + idx_batch * batch_size))
            with open(result_file_name, 'wb') as result_file:
                # save the results
                pickle.dump(results, result_file, -1)
            # add the file to zip folder
            zip_file.write(result_file_name)
            # remove the pickle file
            remove(result_file_name)


def simulate(
        simulation_parameters: Dict[str, Any],
        simulation_name: str,
        num_examples: int,
        batch_size: int,
        output_location: str
):
    """Main Api method to run and save simulation outcome

    :param simulation_parameters: Parameters to create a simulator from
    :param simulation_name: A descriptive name for the simulation
    :param num_examples: Number of experiments to create
    :param batch_size: Size of each batch
    :param output_location: The absolute path of the folder where the resulting zip files
    will be saved;for each call there will be 2 files creates - one without distortion and
    the other with distortion of pulses
    """

    simulator = create_simulation(simulation_parameters, False)
    # 2) Run the simulator for pulses without distortions and collect/save the results
    print("Running simulator for pulses without distortion")
    zipfile_name = join(output_location, simulation_name, '.zip')
    with zipfile.ZipFile(
            zipfile_name,
            mode='w',
            compression=zipfile.ZIP_DEFLATED) as simulation_without_distortion:

        save_simulation_result(
            num_examples,
            batch_size,
            False,
            simulator,
            simulation_name,
            simulation_without_distortion,
            output_location
        )
    print("Pulses without distortion saved in {}".format(zipfile_name))

    print("Running the simulation for pulses with distortion")
    simulator = create_simulation(simulation_parameters, True)
    zipfile_name = join(output_location, simulation_name, '_distortion.zip')
    with zipfile.ZipFile(
            zipfile_name,
            mode='w',
            compression=zipfile.ZIP_DEFLATED) as simulation_with_distortion:
        save_simulation_result(
            num_examples,
            batch_size,
            True,
            simulator,
            simulation_name,
            simulation_with_distortion,
            output_location
        )
