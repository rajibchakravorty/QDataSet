"""
This class defines a custom tensorflow layer that constructs the Vo operator using the interaction picture definition
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class VoLayer(layers.Layer):

    def __init__(self, O, **kwargs):
        """
        Class constructor

        O: The observable to be measured
        """
        # this has to be called for any tensorflow custom layer
        super(VoLayer, self).__init__(**kwargs)

        self.O = tf.constant(O, dtype=tf.complex64)

    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.

        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow.
        """

        # retrieve the two inputs: Uc and UI
        UI, Uc = x

        UI_tilde = tf.matmul(Uc, tf.matmul(UI, Uc, adjoint_b=True))

        # expand the observable operator along batch and realizations axis
        O = tf.expand_dims(self.O, 0)
        O = tf.expand_dims(O, 0)

        temp_shape = tf.concat([tf.shape(Uc)[0:2], tf.constant(np.array([1, 1], dtype=np.int32))], 0)
        O = tf.tile(O, temp_shape)

        # Construct Vo operator
        VO = tf.matmul(O, tf.matmul(tf.matmul(UI_tilde, O, adjoint_a=True), UI_tilde))

        return VO
