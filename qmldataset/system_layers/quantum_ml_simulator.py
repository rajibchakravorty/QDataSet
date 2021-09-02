"""
This is the main class that defines machine learning model of the qubit.
"""

from typing import List, Dict, Any

from numpy import array

from tensorflow import (
    constant,
    int32,
    matmul,
    tile,
    Tensor
)

from tensorflow.keras import layers, Model

from .hamiltonian_construction_layer import HamiltonianConstruction
from .lti_layer import LTILayer
from .noise_layer import NoiseLayer
from .quantum_evolution import QuantumEvolution
from .quantum_measurement_layer import QuantumMeasurement
from .signal_generator_layer import SignalGenerator
from .vo_layer import VoLayer


class QuantumTFSimulator():
    """Main simulator class.

    :param evolution_time : Evolution time
    :param num_time_steps: Number of time steps
    :param dynamic_operators: A list of arrays that represent the
    terms of the control Hamiltonian (that depend on pulses)
    :param static_operators: A list of arrays that represent
    the terms of the drifting Hamiltonian (that are constant)
    :param noise_operators: A list of arrays that represent
    the terms of the classical noise Hamiltonians
    :param measurement_operators: A list of arrays that represent
    measurement operators
    :param initial_states: A list of arrays representing initial states
    :param num_realizations: Number of noise realizations; defaults to 1
    :param pulse_shape: The type of waveform [either "Zero", "Square", or "Gaussian"];
    defaults to Gaussian
    :param num_pulses: Number of pulses per control sequence: defaults to 5Nice
    :param distortion: True for simulating distortions, False for no distortions;
    defaults to False
    :param noise_profile : A list of noies types. Each type must be one of
    ['Type 0','Type 1','Type 2','Type 4','Type 5','Type 6'];
    defaults to 'Type 0'
    """

    def __init__(self,
                 evolution_time: float,
                 num_time_steps: int,
                 dynamic_operators: List[array],
                 static_operators: List[array],
                 noise_operators: List[array],
                 measurement_operators: List[array],
                 initial_states: List[array],
                 num_realizations: int = 1,
                 pulse_shape: str = "Gaussian",
                 num_pulses: int = 5,
                 distortion: bool = False,
                 noise_profile: List[str] = 'Type 0'):

        time_step = evolution_time / num_time_steps
        self.time_range = [
            (0.5 * evolution_time / num_time_steps) +
            (j * evolution_time / num_time_steps) for j in range(num_time_steps)]

        # define a dummy input layer needed to generate the control pulses and noise
        dummy_input = layers.Input(shape=(1,))

        # define the custom tensorflow layer that generates
        # the control pulses for each direction and concatenate if necessary
        if len(dynamic_operators) > 1:
            pulses = [
                SignalGenerator(
                    evolution_time,
                    num_time_steps,
                    num_pulses,
                    pulse_shape)(dummy_input) for _ in dynamic_operators]
            pulse_parameters = layers.Concatenate(axis=-1)([pulse[0] for pulse in pulses])
            pulse_time_domain = layers.Concatenate(axis=-1)([pulse[1] for pulse in pulses])
        else:
            pulse_parameters, pulse_time_domain = SignalGenerator(
                evolution_time, num_time_steps, num_pulses, pulse_shape)(dummy_input)

        if distortion:
            distorted_pulse_time_domain = LTILayer(
                evolution_time, num_time_steps)(pulse_time_domain)
        else:
            distorted_pulse_time_domain = pulse_time_domain

        # define the custom tensorflow layer that generates the noise
        # realizations in time-domain and concatenate if necessary
        if len(noise_operators) > 1:
            noise = []
            for profile in noise_profile:
                if profile != 'Type 6':  # uncorrelated along different directions
                    noise.append(
                        NoiseLayer(
                            evolution_time,
                            num_time_steps,
                            num_realizations,
                            profile)(dummy_input))
                else:  # correlated with the previous direction
                    noise.append(
                        NoiseLayer(
                            evolution_time,
                            num_time_steps,
                            num_realizations,
                            profile)(noise[-1]))
            noise_time_domain = layers.Concatenate(axis=-1)(noise)
        else:
            noise_time_domain = NoiseLayer(
                evolution_time,
                num_time_steps,
                num_realizations,
                noise_profile[0])(dummy_input)

        # define the custom tensorflow layer that constructs the
        # H0 part of the Hamiltonian from parameters at each time step
        hamiltonian_0 = HamiltonianConstruction(
            dynamic_operators=dynamic_operators,
            static_operators=static_operators,
            name="H0")(distorted_pulse_time_domain)

        # define the custom tensorflow layer that
        # constructs the H1 part of the Hamiltonian
        # from parameters at each time step
        hamiltonian_1 = HamiltonianConstruction(
            dynamic_operators=noise_operators,
            static_operators=[], name="H1")(noise_time_domain)

        # define the custom tensorflow layer
        # that constructs the time-ordered evolution of H0
        unitary_0 = QuantumEvolution(
            time_step, return_sequences=True, name="U0")(hamiltonian_0)

        # define Uc which is U0(T)
        unitary_c = layers.Lambda(
            lambda u0: u0[:, -1, :, :, :], name="Uc")(unitary_0)

        # define custom tensorflow layer to calculate HI
        unitary_0ext = layers.Lambda(
            lambda x: tile(x, constant([1, 1, num_realizations, 1, 1], dtype=int32)))(unitary_0)
        hamiltonian_i = layers.Lambda(
            lambda x: matmul(
                matmul(x[0], x[1], adjoint_a=True), x[0]), name="HI")\
            ([unitary_0ext, hamiltonian_1])

        # define the custom defined tensorflow layer that constructs the
        # time-ordered evolution of HI
        unitary_i = QuantumEvolution(
            time_step, return_sequences=False, name="UI")(hamiltonian_i)

        # construct the Vo operators
        unitary_cext = layers.Lambda(
            lambda x: tile(x, constant([1, num_realizations, 1, 1], dtype=int32)))(unitary_c)
        observables= [
            VoLayer(operator, name="V%d" % idx_op)(
                [unitary_i, unitary_cext])
            for idx_op, operator in enumerate(measurement_operators)]

        # add the custom defined tensorflow layer that calculates the measurement outcomes
        expectations = [
            [QuantumMeasurement(rho, X, name="rho%dM%d" % (idx_rho, idx_X))(
                [observables[idx_X], unitary_c]) for idx_X, X in enumerate(measurement_operators)]
            for idx_rho, rho in enumerate(initial_states)]

        # concatenate all the measurement outcomes
        expectations = layers.Concatenate(axis=-1)(sum(expectations, []))

        # save the simulation parameters
        self.simulation_parameters = {
            "evolution_time" : evolution_time,
            "num_time_steps" : num_time_steps,
            "dynamic_operators": dynamic_operators,
            "static_operators": static_operators,
            "noise_operators": noise_operators,
            "measurement_operators": measurement_operators,
            "initial_states": initial_states,
            "num_realizations": num_realizations,
            "pulse_shape": pulse_shape,
            "num_pulses": num_pulses,
            "distortion": distortion,
            "noise_profile": noise_profile
        }
        # define now the tensorflow model
        self.model = Model(
            inputs=dummy_input,
            outputs=[
                        pulse_parameters,
                        pulse_time_domain,
                        distorted_pulse_time_domain,
                        noise_time_domain,
                        hamiltonian_0,
                        hamiltonian_1,
                        unitary_0,
                        unitary_c,
                        unitary_i,
                        expectations] + observables)

        # print a summary of the model showing the system_layers and their connections
        self.model.summary()

    def simulate(self, simulator_inputs: array, batch_size: int = 1) -> List[Tensor]:
        """
        This method is for predicting the measurement outcomes using the trained model.
        Usually called after training.
        :param simulator_inputs: A dummy numpy array of shape (number of experiments to simulate, 1)
        :param batch_size:  The number of experiments to process at each batch, chosen according
        to available memory; defaults to 1

        :returns: a list of arrays representing H0,H1,U0,U0(T),VO,expectations respectively
        """
        return self.model.predict(simulator_inputs, verbose=1, batch_size=batch_size)

    def get_simulation_parameters(self)->Dict[str, Any]:
        """Returns the parameters of simulator object as a dictionary

        :return: A dictionary object with simulation parameters
        """

        return self.simulation_parameters
