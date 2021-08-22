"""
This is the main class that defines machine learning model of the qubit.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from .signal_generator_layer import SignalGenerator
from .lti_layer import LTILayer
from .noise_layer import NoiseLayer
from .quantum_evolution import QuantumEvolution
from .vo_layer import VoLayer
from .hamiltonian_construction_layer import HamiltonianConstruction
from .quantum_measurement_layer import QuantumMeasurement


class QuantumTFSimulator():

    def __init__(self, T, M, dynamic_operators, static_operators, noise_operators, measurement_operators,
                 initial_states, K=1, waveform="Gaussian", num_pulses=5, distortion=False, noise_profile=0):
        """
        Class constructor.

        T                : Evolution time
        M                : Number of time steps
        dynamic_operators: A list of arrays that represent the terms of the control Hamiltonian (that depend on pulses)
        static_operators : A list of arrays that represent the terms of the drifting Hamiltonian (that are constant)
        noise_operators  : A list of arrays that represent the terms of the classical noise Hamiltonians
        K                : Number of noise realizations
        waveform         : The type of waveform [either "Zero", "Square", or "Gaussian"]
        num_pulses       : Number of pulses per control sequence
        distortion       : True for simulating distortions, False for no distortions
        noise_profile    : The type of noise, a value chosen from [0,1,2,4,5,6]
        """

        delta_T = T / M
        self.time_range = [(0.5 * T / M) + (j * T / M) for j in range(M)]

        # define a dummy input layer needed to generate the control pulses and noise
        dummy_input = layers.Input(shape=(1,))

        # define the custom tensorflow layer that generates the control pulses for each direction and concatente if neccesary
        if len(dynamic_operators) > 1:
            pulses = [SignalGenerator(T, M, num_pulses, waveform)(dummy_input) for _ in dynamic_operators]
            pulse_parameters = layers.Concatenate(axis=-1)([p[0] for p in pulses])
            pulse_time_domain = layers.Concatenate(axis=-1)([p[1] for p in pulses])
        else:
            pulse_parameters, pulse_time_domain = SignalGenerator(T, M, num_pulses, waveform)(dummy_input)

        if distortion == True:
            distorted_pulse_time_domain = LTILayer(T, M)(pulse_time_domain)
        else:
            distorted_pulse_time_domain = pulse_time_domain

        # define the custom tensorflow layer that generates the noise realizations in time-domain and concatente if neccesary
        if len(noise_operators) > 1:
            noise = []
            for profile in noise_profile:
                if profile != 6:  # uncorrelated along different directions
                    noise.append(NoiseLayer(T, M, K, profile)(dummy_input))
                else:  # correlated with the prevu=ious direction
                    noise.append(NoiseLayer(T, M, K, profile)(noise[-1]))
            noise_time_domain = layers.Concatenate(axis=-1)(noise)
        else:
            noise_time_domain = NoiseLayer(T, M, K, noise_profile[0])(dummy_input)

            # define the custom tensorflow layer that constructs the H0 part of the Hamiltonian from parameters at each time step
        H0 = HamiltonianConstruction(dynamic_operators=dynamic_operators, static_operators=static_operators, name="H0")(
            distorted_pulse_time_domain)

        # define the custom tensorflow layer that constructs the H1 part of the Hamiltonian from parameters at each time step
        H1 = HamiltonianConstruction(dynamic_operators=noise_operators, static_operators=[], name="H1")(
            noise_time_domain)

        # define the custom tensorflow layer that constructs the time-ordered evolution of H0
        U0 = QuantumEvolution(delta_T, return_sequences=True, name="U0")(H0)

        # define Uc which is U0(T)
        Uc = layers.Lambda(lambda u0: u0[:, -1, :, :, :], name="Uc")(U0)

        # define custom tensorflow layer to calculate HI
        U0_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1, 1, K, 1, 1], dtype=tf.int32)))(U0)
        HI = layers.Lambda(lambda x: tf.matmul(tf.matmul(x[0], x[1], adjoint_a=True), x[0]), name="HI")([U0_ext, H1])

        # define the custom defined tensorflow layer that constructs the time-ordered evolution of HI
        UI = QuantumEvolution(delta_T, return_sequences=False, name="UI")(HI)

        # construct the Vo operators
        Uc_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1, K, 1, 1], dtype=tf.int32)))(Uc)
        Vo = [VoLayer(O, name="V%d" % idx_O)([UI, Uc_ext]) for idx_O, O in enumerate(measurement_operators)]

        # add the custom defined tensorflow layer that calculates the measurement outcomes
        expectations = [
            [QuantumMeasurement(rho, X, name="rho%dM%d" % (idx_rho, idx_X))([Vo[idx_X], Uc]) for idx_X, X in
             enumerate(measurement_operators)]
            for idx_rho, rho in enumerate(initial_states)]

        # concatenate all the measurement outcomes
        expectations = layers.Concatenate(axis=-1)(sum(expectations, []))

        # define now the tensorflow model
        self.model = Model(inputs=dummy_input,
                           outputs=[pulse_parameters, pulse_time_domain, distorted_pulse_time_domain, noise_time_domain,
                                    H0, H1, U0, Uc, UI, expectations] + Vo)

        # print a summary of the model showing the system_layers and their connections
        self.model.summary()

    def simulate(self, simulator_inputs, batch_size=1):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.

        simulator inputs: A dummy numpy array of shape (number of examples to simulate, 1)

        batch_size:  The number of examples to process at each batch, chosen according to available memory

        returns a list of arrays representing H0,H1,U0,U0(T),VO,expectations respectively
        """
        return self.model.predict(simulator_inputs, verbose=1, batch_size=batch_size)
