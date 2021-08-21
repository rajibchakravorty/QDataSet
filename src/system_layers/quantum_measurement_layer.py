"""
This class defines a custom tensorflow layer that takes the unitary as input,
and generates the measurement outcome probability as output
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class QuantumMeasurement(layers.Layer):

    def __init__(self, initial_state, measurement_operator, **kwargs):
        """
        Class constructor

        initial_state       : The inital density matrix of the state before evolution.
        Measurement_operator: The measurement operator
        """
        self.initial_state = tf.constant(initial_state, dtype=tf.complex64)
        self.measurement_operator = tf.constant(measurement_operator, dtype=tf.complex64)

        # we must call thus function for any tensorflow custom layer
        super(QuantumMeasurement, self).__init__(**kwargs)

    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.

        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow.
        """

        # extract the different inputs of this layer which are the Vo and Uc
        Vo, Uc = x

        # construct a tensor in the form of a row vector whose elements are [d1,1,1,1], where d1 corresponds to the
        # number of examples of the input
        temp_shape = tf.concat([tf.shape(Vo)[0:1], tf.constant(np.array([1, 1, 1], dtype=np.int32))], 0)

        # add an extra dimensions for the initial state and measurement tensors to represent batch and realization
        initial_state = tf.expand_dims(tf.expand_dims(self.initial_state, 0), 0)
        measurement_operator = tf.expand_dims(tf.expand_dims(self.measurement_operator, 0), 0)

        # repeat the initial state and measurment tensors along the batch dimensions
        initial_state = tf.tile(initial_state, temp_shape)
        measurement_operator = tf.tile(measurement_operator, temp_shape)

        # evolve the initial state using the propagator provided as input
        final_state = tf.matmul(tf.matmul(Uc, initial_state), Uc, adjoint_b=True)

        # tile along the realization axis
        temp_shape = tf.concat([tf.constant(np.array([1, ], dtype=np.int32)), tf.shape(Vo)[1:2],
                                tf.constant(np.array([1, 1], dtype=np.int32))], 0)
        final_state = tf.tile(final_state, temp_shape)
        measurement_operator = tf.tile(measurement_operator, temp_shape)

        # calculate the probability of the outcome
        expectation = tf.linalg.trace(tf.matmul(tf.matmul(Vo, final_state), measurement_operator))

        return tf.expand_dims(tf.math.real(expectation), -1)
