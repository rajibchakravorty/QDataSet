"""Sample use of qmldataset package
"""
from itertools import product
from numpy import array, kron
from qmldataset import pauli_operators, create_custom_simulator, run_experiment


def create_experiment():
    """In this sample usage we will create a custom experiment
    """
    dimension = 4
    evolution_time = 1
    num_time_steps = 1024
    omega = [10, 12]
    dynamic_operators = [
        kron(pauli_operators[1], pauli_operators[0]),
        kron(pauli_operators[0], pauli_operators[1]),
        kron(pauli_operators[1], pauli_operators[1])]
    static_operators = [
        omega[0] * kron(pauli_operators[3], pauli_operators[0]),
        omega[1] * kron(pauli_operators[0], pauli_operators[3])]
    noise_operators = [
        kron(pauli_operators[3], pauli_operators[0]),
        kron(pauli_operators[0], pauli_operators[3])]
    measurement_operators = [
        kron(meas_op_one, meas_op_two) for meas_op_one, meas_op_two in list(
            product(pauli_operators, pauli_operators))][1:]
    initial_states_1q = [
        array([[0.5, 0.5], [0.5, 0.5]]), array([[0.5, -0.5], [-0.5, 0.5]]),
        array([[0.5, -0.5j], [0.5j, 0.5]]), array([[0.5, 0.5j], [-0.5j, 0.5]]),
        array([[1, 0], [0, 0]]), array([[0, 0], [0, 1]])
    ]
    initial_states = [
        kron(init_state_one, init_state_two) for init_state_one, init_state_two in list(
            product(initial_states_1q, initial_states_1q))]
    num_realizations = 500
    num_pulses = 5
    noise_profile = ['Type 1', 'Type 5']
    pulse_shape = "Square"
    distortion = True

    simulator = create_custom_simulator(
        evolution_time=evolution_time,
        num_time_steps=num_time_steps,
        dimension=dimension,
        dynamic_operators=dynamic_operators,
        static_operators=static_operators,
        noise_operators=noise_operators,
        measurement_operators=measurement_operators,
        initial_states=initial_states,
        num_realizations=num_realizations,
        pulse_shape=pulse_shape,
        num_pulses=num_pulses,
        noise_profile=noise_profile,
        distortion=distortion
    )

    # run and gather of one experiment result
    experiment_result = run_experiment(
        simulator=simulator
    )

    for param in experiment_result:
        print("-- {} --".format(param))
        print("-- {} --".format(experiment_result[param]))


if __name__ == '__main__':
    create_experiment()
