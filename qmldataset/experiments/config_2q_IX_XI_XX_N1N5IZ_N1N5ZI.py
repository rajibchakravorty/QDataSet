# pylint: disable=invalid-name
"""Config for 2q_IX_XI_XX_N1N5IZ_N1N5ZI experiment - 2-qubit, Control on X Axis for
both qubits and X Axis interacting control, Type 1 and Type 5 Noise on Z axis for both qubits
"""
from itertools import product
import numpy as np
from ..utilities.constants import pauli_operators

name = "2q_IX_XI_XX_N1N5IZ_N1N5ZI"


dimension = 4
evolution_time = 1
num_time_steps = 1024
omega = [10, 12]
dynamic_operators = [
    np.kron(pauli_operators[1], pauli_operators[0]),
    np.kron(pauli_operators[0], pauli_operators[1]),
    np.kron(pauli_operators[1], pauli_operators[1])]
static_operators = [
    omega[0]*np.kron(pauli_operators[3], pauli_operators[0]),
    omega[1]*np.kron(pauli_operators[0], pauli_operators[3])]
noise_operators = [
    np.kron(pauli_operators[3], pauli_operators[0]),
    np.kron(pauli_operators[0], pauli_operators[3])]
measurement_operators = [
    np.kron(meas_op_one, meas_op_two) for meas_op_one, meas_op_two in list(
        product(pauli_operators, pauli_operators))][1:]
initial_states_1q = [
    np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]),
    np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
    np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])
]
initial_states = [
    np.kron(init_state_one, init_state_two) for init_state_one, init_state_two in list(
        product(initial_states_1q, initial_states_1q))]
num_realizations = 2000
num_pulses = 5
noise_profile = ['Type 1', 'Type 5']
