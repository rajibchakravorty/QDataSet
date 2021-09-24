"""Sample use of qmldataset package
"""
from itertools import product
from numpy import array, kron
from qmldataset import pauli_operators, run_custom_experiment

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
    experiment_name = 'custom_2q_IX_XI_XX_N1N5Z_N1N5Z'
    num_examples = 100
    batch_size = 4
    output_location = '/home/rchakrav/progs/qmldataset_result'
    pulse_shape = "Square"

    run_custom_experiment(
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
        experiment_name=experiment_name,
        num_examples=num_examples,
        batch_size=batch_size,
        output_location=output_location
    )


if __name__ == '__main__':
    create_experiment()
