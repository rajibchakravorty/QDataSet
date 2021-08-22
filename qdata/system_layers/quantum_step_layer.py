"""
This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces one step forward propagator
"""

import tensorflow as tf
from tensorflow.keras import layers


class QuantumCell(layers.Layer):

    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
        delta_T: time step for each propagator
        """

        # here we define the time-step including the imaginary unit, so we can later use it directly with the expm function
        self.delta_T = tf.constant(delta_T * -1j, dtype=tf.complex64)

        # we must define this parameter for RNN cells
        self.state_size = [1]

        # we must call thus function for any tensorflow custom layer
        super(QuantumCell, self).__init__(**kwargs)

    def call(self, inputs, states):
        """
        This method must be defined for any custom layer, it is where the calculations are done.

        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        states: The tensor representing the state of the cell. This is passed automatically by tensorflow.
        """

        previous_output = states[0]

        # evaluate -i*H*delta_T
        Hamiltonian = inputs * self.delta_T

        # evaluate U = expm(-i*H*delta_T)
        U = tf.linalg.expm(Hamiltonian)

        # accuamalte U to to the rest of the propagators
        new_output = tf.matmul(U, previous_output)

        return new_output, [new_output]
