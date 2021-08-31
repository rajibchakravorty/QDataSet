"""
This class defines a custom tensorflow layer that takes the Hamiltonian parameters
as input, and generates the Hamiltonain matrix as an output at each time step for
each example in the batch
"""

from numpy import (
    array,
    int32
)
from tensorflow import (
    add_n,
    cast,
    complex64,
    concat,
    constant,
    expand_dims,
    multiply,
    shape,
    tile,
    Tensor
    )
from tensorflow.keras import layers


class HamiltonianConstruction(layers.Layer):
    """Class constructor

    dynamic_operators: a list of all operators that have time-varying coefficients
    static_operators : a list of all operators that have constant coefficients
    """

    def __init__(self, dynamic_operators: array, static_operators: array, **kwargs):
        self.dynamic_operators = [constant(op, dtype=complex64) for op in dynamic_operators]
        self.static_operators = [constant(op, dtype=complex64) for op in static_operators]
        self.dim = dynamic_operators[0].shape[-1]

        # this has to be called for any tensorflow custom layer
        super().__init__(**kwargs)

    def call(self, inputs: Tensor):     # pylint: disable=arguments-differ
        """Custom call method of the layer
        """

        hamiltonians = []
        # loop over the strengths of all dynamic operators

        for idx_op, dynamic_op in enumerate(self.dynamic_operators):
            # select the particular strength of the operator
            hamiltonian = cast(inputs[:, :, :, idx_op:idx_op + 1], complex64)

            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3, 1,1],
            # where d1, d2, and d3 correspond to the number of experiments,
            # number of time steps of the input, and number of realizations
            temp_shape = concat(
                [shape(inputs)[0:3], constant(array([1, 1], dtype=int32))], 0)

            # add two extra dimensions for batch, time, and realization
            operator = expand_dims(dynamic_op, 0)
            operator = expand_dims(operator, 0)
            operator = expand_dims(operator, 0)

            # repeat the pauli operators along the batch and time dimensions
            operator = tile(operator, temp_shape)

            # repeat the pulse waveform to as dxd matrix
            temp_shape = constant(array([1, 1, 1, self.dim, self.dim], dtype=int32))
            hamiltonian = expand_dims(hamiltonian, -1)
            hamiltonian = tile(hamiltonian, temp_shape)

            # Now multiply each operator with its corresponding strength element-wise and
            # add to the list of Hamiltonians
            hamiltonians.append(multiply(operator, hamiltonian))

        # loop over the strengths of all static operators
        for static_op in self.static_operators:
            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3,1,1],
            # where d1, d2, and d2 correspond to the number of experiments, number of time steps
            # of the input, and number of realizations
            temp_shape = concat(
                [shape(inputs)[0:3], constant(array([1, 1], dtype=int32))], 0)

            # add two extra dimensions for batch and time
            operator = expand_dims(static_op, 0)
            operator = expand_dims(operator, 0)
            operator = expand_dims(operator, 0)

            # repeat the pauli operators along the batch and time dimensions
            operator = tile(operator, temp_shape)

            # Now add to the list of Hamiltonians
            hamiltonians.append(operator)

        # now add all components together
        hamiltonians = add_n(hamiltonians)

        return hamiltonians
