# pylint: disable=invalid-name
"""
Configuration for experiment G_1q_X
"""
import numpy as np
from ..utilities import pauli_operators

name = "G_1q_X"

dimension = 2   # dimension of the system
omega = 12  # qubit energy gap
evolution_time = 1
num_time_steps = 1024
dynamic_operators = [0.5*pauli_operators[1]]
static_operators = [0.5*pauli_operators[3]*omega]
noise_operators = [0.5*pauli_operators[3]]
measurement_operators = pauli_operators[1:]
initial_states = [
    np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]),
    np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
    np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])
]
num_realizations = 2000
pulse_shape = "Gaussian"
num_pulses = 5
noise_profile = 'Type 0'
