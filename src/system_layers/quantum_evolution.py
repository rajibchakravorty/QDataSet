"""
This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces the time-ordered evolution unitary as output
"""
import tensorflow as tf
from tensorflow.keras import layers

from .quantum_step_layer import QuantumCell

class QuantumEvolution(layers.RNN):

    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.

        delta_T: time step for each propagator
        """

        # use the custom-defined QuantumCell as base class for the nodes
        cell = QuantumCell(delta_T)

        # we must call thus function for any tensorflow custom layer
        super(QuantumEvolution, self).__init__(cell, **kwargs)

    def call(self, inputs):
        """
        This method must be defined for any custom layer, it is where the calculations are done.

        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        """

        # define identity matrix with correct dimensions to be used as initial propagtor
        dimensions = tf.shape(inputs)
        I = tf.eye(dimensions[-1], batch_shape=[dimensions[0], dimensions[2]], dtype=tf.complex64)

        return super(QuantumEvolution, self).call(inputs, initial_state=[I])
