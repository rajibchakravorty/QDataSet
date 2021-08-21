"""
This function generates the dataset and stores it based on the simulation parameters passed as a dictionary
"""

import os
import time
import zipfile
import pickle

import numpy as np

from ..system_layers import QuantumTFSimulator


def simulate(sim_parameters):

    simulator = QuantumTFSimulator(
        sim_parameters["T"], sim_parameters["M"], sim_parameters["dynamic_operators"],
        sim_parameters["static_operators"], sim_parameters["noise_operators"],
        sim_parameters["measurement_operators"],
        sim_parameters["initial_states"], sim_parameters["K"], sim_parameters["pulse_shape"],
        sim_parameters["num_pulses"], False, sim_parameters["noise_profile"])

    fzip = zipfile.ZipFile("%s.zip" % sim_parameters["name"], mode='w', compression=zipfile.ZIP_DEFLATED)

    # 2) Run the simulator for pulses without distortions and collect the results
    print("Running the simulation for pulses without distortion\n")
    for idx_batch in range(sim_parameters["num_ex"] // sim_parameters["batch_size"]):
        ###########################################################################
        print("Simulating and storing batch %d\n" % idx_batch)
        start = time.time()
        simulation_results = simulator.simulate(np.zeros((sim_parameters["batch_size"], 1)),
                                                batch_size=sim_parameters["batch_size"])
        sim_parameters["elapsed_time"] = time.time() - start
        pulse_parameters, pulses, distorted_pulses, noise = simulation_results[0:4]
        H0, H1, U0, Uc, UI, expectations = simulation_results[4:10]
        Vo = simulation_results[10:]
        ###########################################################################
        # 4) Save the results in an external file and zip everything
        for idx_ex in range(sim_parameters["batch_size"]):
            Results = {"sim_parameters": sim_parameters,
                       "pulse_parameters": pulse_parameters[idx_ex:idx_ex + 1, :],
                       "time_range": simulator.time_range,
                       "pulses": pulses[idx_ex:idx_ex + 1, :],
                       "distorted_pulses": pulses[idx_ex:idx_ex + 1, :],
                       "expectations": np.average(expectations[idx_ex:idx_ex + 1, :], axis=1),
                       "Vo_operator": [np.average(V[idx_ex:idx_ex + 1, :], axis=1) for V in Vo],
                       "noise": noise[idx_ex:idx_ex + 1, :],
                       "H0": H0[idx_ex:idx_ex + 1, :],
                       "H1": H1[idx_ex:idx_ex + 1, :],
                       "U0": U0[idx_ex:idx_ex + 1, :],
                       "UI": UI[idx_ex:idx_ex + 1, :],
                       "Vo": [V[idx_ex:idx_ex + 1, :] for V in Vo],
                       "Eo": expectations[idx_ex:idx_ex + 1, :]
                       }
            # open a pickle file
            fname = "%s_ex_%d" % (sim_parameters["name"], idx_ex + idx_batch * sim_parameters["batch_size"])
            f = open(fname, 'wb')
            # save the results
            pickle.dump(Results, f, -1)
            # close the pickle file
            f.close()
            # add the file to zip folder
            fzip.write(fname)
            # remove the pickle file
            os.remove(fname)
    ###########################################################################
    # close the zip file
    fzip.close()
    os.system('cp %s.zip /projects/QuantumDS/%s.zip' % (sim_parameters["name"], sim_parameters["name"]))
    ###########################################################################
    # 3) Run the simulator for pulses with distortions and collect the results
    print("Running the simulation for pulses with distortion\n")
    simulator = QuantumTFSimulator(
        sim_parameters["T"], sim_parameters["M"], sim_parameters["dynamic_operators"],
        sim_parameters["static_operators"], sim_parameters["noise_operators"],
        sim_parameters["measurement_operators"],
        sim_parameters["initial_states"], sim_parameters["K"], sim_parameters["pulse_shape"],
        sim_parameters["num_pulses"], True, sim_parameters["noise_profile"])

    fzip = zipfile.ZipFile("%s_D.zip" % sim_parameters["name"], mode='w', compression=zipfile.ZIP_DEFLATED)

    for idx_batch in range(sim_parameters["num_ex"] // sim_parameters["batch_size"]):
        ###########################################################################
        print("Simulating and storing batch %d\n" % idx_batch)
        start = time.time()
        simulation_results = simulator.simulate(np.zeros((sim_parameters["batch_size"], 1)),
                                                batch_size=sim_parameters["batch_size"])
        sim_parameters["elapsed_time"] = time.time() - start
        pulse_parameters, pulses, distorted_pulses, noise = simulation_results[0:4]
        H0, H1, U0, Uc, UI, expectations = simulation_results[4:10]
        Vo = simulation_results[10:]
        ###########################################################################
        # 4) Save the results in an external file and zip everything
        for idx_ex in range(sim_parameters["batch_size"]):
            Results = {"sim_parameters": sim_parameters,
                       "pulse_parameters": pulse_parameters[idx_ex:idx_ex + 1, :],
                       "time_range": simulator.time_range,
                       "pulses": pulses[idx_ex:idx_ex + 1, :],
                       "distorted_pulses": distorted_pulses[idx_ex:idx_ex + 1, :],
                       "expectations": np.average(expectations[idx_ex:idx_ex + 1, :], axis=1),
                       "Vo_operator": [np.average(V[idx_ex:idx_ex + 1, :], axis=1) for V in Vo],
                       "noise": noise[idx_ex:idx_ex + 1, :],
                       "H0": H0[idx_ex:idx_ex + 1, :],
                       "H1": H1[idx_ex:idx_ex + 1, :],
                       "U0": U0[idx_ex:idx_ex + 1, :],
                       "UI": UI[idx_ex:idx_ex + 1, :],
                       "Vo": [V[idx_ex:idx_ex + 1, :] for V in Vo],
                       "Eo": expectations[idx_ex:idx_ex + 1, :]
                       }
            # open a pickle file
            fname = "%s_D_ex_%d" % (sim_parameters["name"], idx_ex + idx_batch * sim_parameters["batch_size"])
            f = open(fname, 'wb')
            # save the results
            pickle.dump(Results, f, -1)
            # close the pickle file
            f.close()
            # add the file to zip folder
            fzip.write(fname)
            # remove the pickle file
            os.remove(fname)
    ###########################################################################
    # close the zip file
    fzip.close()
    os.system('cp %s_D.zip /projects/QuantumDS/%s_D.zip' % (sim_parameters["name"], sim_parameters["name"]))
