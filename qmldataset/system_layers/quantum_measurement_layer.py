"""Quantum Measurement Layer
"""
from numpy import (
    array,
    int32
)
from tensorflow import (
    complex64,
    concat,
    constant,
    expand_dims,
    linalg,
    math,
    matmul,
    shape,
    tile,
    Tensor
)

from tensorflow.keras import layers


class QuantumMeasurement(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the unitary as input,
    and generates the measurement outcome probability as output

    :param initial_state : The initial density matrix of the state before evolution.
    :param measurement_operator: The measurement operator

    """
    def __init__(self,
                 initial_state: array,
                 measurement_operator: array, **kwargs):

        self.initial_state = constant(initial_state, dtype=complex64)
        self.measurement_operator = constant(measurement_operator, dtype=complex64)

        # we must call thus function for any tensorflow custom layer
        super().__init__(**kwargs)

    def call(self, inputs: Tensor):     # pylint: disable=arguments-differ
        """Custom call method of the layer
        """

        # extract the different inputs of this layer which are the Vo and Uc
        examples, propagator = inputs

        # construct a tensor in the form of a row vector whose
        # elements are [d1,1,1,1], where d1 corresponds to the
        # number of configurations of the input
        temp_shape = concat(
            [shape(examples)[0:1], constant(array([1, 1, 1], dtype=int32))], 0)

        # add an extra dimensions for the initial state and measurement
        # tensors to represent batch and realization
        initial_state = expand_dims(expand_dims(self.initial_state, 0), 0)
        measurement_operator = expand_dims(expand_dims(self.measurement_operator, 0), 0)

        # repeat the initial state and measurement tensors along the batch dimensions
        initial_state = tile(initial_state, temp_shape)
        measurement_operator = tile(measurement_operator, temp_shape)

        # evolve the initial state using the propagator provided as input
        final_state = matmul(matmul(propagator, initial_state),
                             propagator, adjoint_b=True)

        # tile along the realization axis
        temp_shape = concat([
            constant(array([1, ], dtype=int32)),
            shape(examples)[1:2],
            constant(array([1, 1], dtype=int32))], 0)

        final_state = tile(final_state, temp_shape)
        measurement_operator = tile(measurement_operator, temp_shape)

        # calculate the probability of the outcome
        expectation = linalg.trace(
            matmul(matmul(examples, final_state), measurement_operator))

        return expand_dims(math.real(expectation), -1)
