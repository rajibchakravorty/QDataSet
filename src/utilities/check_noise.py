"""
This function calculates the coherence measurements to check the noise behaviour, based on the simulation parameters passed as a dictionary
"""

from numpy import average, eye
from numpy.linalg import norm

from ..system_layers import QuantumTFSimulator


def check_noise(sim_parameters):

    simulator = QuantumTFSimulator(
        sim_parameters["T"], sim_parameters["M"], sim_parameters["dynamic_operators"],
        sim_parameters["static_operators"], sim_parameters["noise_operators"],
        sim_parameters["measurement_operators"],
        sim_parameters["initial_states"], sim_parameters["K"], "Zero",
        sim_parameters["num_pulses"], False, sim_parameters["noise_profile"])
    # 3) Run the simulator and collect the results
    print("Running the simulation\n")
    simulation_results = simulator.simulate(np.zeros((1,)), batch_size=1)
    H0, H1, U0, Uc, UI, expectations = simulation_results[4:10]
    Vo = simulation_results[10:]
    Vo = [average(V, axis=1) for V in Vo]
    print("Analyzing results\n")
    print("Measurement are:")
    print(average(expectations, axis=1))
    print("The Vo operators are:")
    print(Vo)
    print("The distance measures are:")
    print([norm(V[0, :] - eye(sim_parameters["dim"]), 2) for V in Vo])
