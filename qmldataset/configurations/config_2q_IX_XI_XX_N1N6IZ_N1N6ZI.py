# pylint: disable=invalid-name
"""Config for 2q_IX_XI_XX_N1N6IZ_N1N6ZI experiment - 2-qubit, Control on X Axis for
both qubits and X Axis interacting control, Type 1 and Type 6 Noise on Z axis for both qubits
"""
from itertools import product
from numpy import array, kron
from ..utilities.constants import pauli_operators

name = "2q_IX_XI_XX_N1N5IZ_N1N6ZI"


dimension = 4
evolution_time = 1
num_time_steps = 1024
omega = [10, 12]
dynamic_operators = [
    kron(pauli_operators[1], pauli_operators[0]),
    kron(pauli_operators[0], pauli_operators[1]),
    kron(pauli_operators[1], pauli_operators[1])]
static_operators = [
    omega[0]*kron(pauli_operators[3], pauli_operators[0]),
    omega[1]*kron(pauli_operators[0], pauli_operators[3])]
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
num_realizations = 2000
num_pulses = 5
noise_profile = ['Type 1', 'Type 6']
