"""Quantum Evolution
"""
from tensorflow import (
    complex64,
    eye,
    shape,
    Tensor
)
from tensorflow.keras import layers
from .quantum_step_layer import QuantumCell


class QuantumEvolution(layers.RNN):
    """Custom layer that takes Hamiltonian as input, and
    produces the time-ordered evolution unitary as output

    :param time_step: time step for each propagator
    """
    def __init__(self, time_step: float, **kwargs):

        # use the custom-defined QuantumCell as base class for the nodes
        cell = QuantumCell(time_step)

        # we must call thus function for any tensorflow custom layer
        super().__init__(cell, **kwargs)

    def call(self, inputs: Tensor):     # pylint: disable=arguments-differ
        """Custom call method of the layer
        """

        # define identity matrix with correct dimensions to be used as initial propagtor
        dimensions = shape(inputs)
        identity = eye(
            dimensions[-1], batch_shape=[dimensions[0], dimensions[2]], dtype=complex64)

        return super().call(inputs, initial_state=[identity])
