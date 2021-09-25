# pylint: disable=invalid-name
"""
Configuration for experiment 1q_XY - 1-qubit, Control on X and Y -Axes, No Noise
"""
from numpy import array
from ..utilities.constants import pauli_operators

name = "1q_XY"

dimension = 2
evolution_time = 1
num_time_steps = 1024
omega = 12
dynamic_operators = [0.5*pauli_operators[1],
                     0.5*pauli_operators[2]]
static_operators = [0.5*pauli_operators[3]*omega]
noise_operators = [0.5*pauli_operators[1],
                   0.5*pauli_operators[3]]
initial_states = [
    array([[0.5, 0.5], [0.5, 0.5]]), array([[0.5, -0.5], [-0.5, 0.5]]),
    array([[0.5, -0.5j], [0.5j, 0.5]]), array([[0.5, 0.5j], [-0.5j, 0.5]]),
    array([[1, 0], [0, 0]]), array([[0, 0], [0, 1]])
]
measurement_operators = pauli_operators[1:]
num_realizations = 2000
num_pulses = 5
noise_profile = ['Type 0', 'Type 0']
