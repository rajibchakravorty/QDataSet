"""One step in Quantum processing
"""

from tensorflow import (
    complex64,
    constant,
    linalg,
    matmul,
    Tensor
)
from tensorflow.keras import layers


class QuantumCell(layers.Layer):
    """Define one quantum cell.
    :param time_step: time step for each propagator
    """
    def __init__(self, time_step: float, **kwargs):

        # here we define the time-step including the imaginary unit,
        # so we can later use it directly with the expm function
        self.time_step = constant(time_step * -1j, dtype=complex64)

        # we must define this parameter for RNN cells
        self.state_size = [1]

        # we must call thus function for any tensorflow custom layer
        super().__init__(**kwargs)

    def call(self, inputs: Tensor, states: Tensor):     # pylint: disable=arguments-differ
        """Custom call for the layer

        :param inputs: The tensor representing the input to the layer.
        :param states: The tensor representing the state of the cell.
        """

        previous_output = states[0]

        # evaluate -i*H*delta_T
        hamiltonian = inputs * self.time_step

        # evaluate U = expm(-i*hamiltomian*time_step)
        unitary = linalg.expm(hamiltonian)

        # accumulate unitaries to the rest of the propagators
        new_output = matmul(unitary, previous_output)

        return new_output, [new_output]
