"""
Module to construct the Vo operator using the interaction picture definition
"""

from numpy import(
    array,
    int32
)
from tensorflow import (
    complex64,
    concat,
    constant,
    expand_dims,
    matmul,
    shape,
    tile,
    Tensor
)
from tensorflow.keras import layers


class VoLayer(layers.Layer):
    """Vo Operator Layer
    :param observable: The observable to be measured
    """

    def __init__(self, observable: array, **kwargs):

        self.observable = constant(observable, dtype=complex64)

        # this has to be called for any tensorflow custom layer
        super().__init__(**kwargs)

    def call(self, inputs: Tensor):     # pylint: disable=arguments-differ
        """Custom call method of the layer
        """
        # retrieve the two inputs: Uc and UI
        unitary_i, unitary_c = inputs

        unitary_tilde = matmul(unitary_c, matmul(unitary_i, unitary_c, adjoint_b=True))

        # expand the observable operator along batch and realizations axis
        observable = expand_dims(expand_dims(self.observable, 0), 0)

        temp_shape = concat([shape(unitary_c)[0:2], constant(array([1, 1], dtype=int32))], 0)
        observable = tile(observable, temp_shape)

        # return Vo operator
        return matmul(
            observable,
            matmul(matmul(unitary_tilde, observable, adjoint_a=True), unitary_tilde))
