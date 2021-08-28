"""
Defining constants
"""

from numpy import eye, array

pauli_operators = [
    eye(2),
    array([[0., 1.], [1., 0.]]),
    array([[0., -1j], [1j, 0.]]),
    array([[1., 0.], [0., -1.]])]
