"""Example of running ML algorithm on simulated experiment data
"""
import numpy as np
import tensorflow.keras as K

from qmldataset import pauli_operators, create_custom_simulator, run_experiment


def create_open_quantum_system_simulator(distortion=False):
    """We create a simple simulator experiment as example
    """

    dimension = 2
    evolution_time = 1
    num_time_steps = 1024
    omega = 12
    dynamic_operators = [0.5 * pauli_operators[1]]
    static_operators = [0.5 * pauli_operators[3] * omega]
    noise_operators = [0.5 * pauli_operators[3]]
    measurement_operators = pauli_operators[1:]
    initial_states = [
        np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]),
        np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
        np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])
    ]
    num_realizations = 100
    num_pulses = 5
    noise_profile = ['Type 0']

    pulse_shape = "Square"

    simulator = create_custom_simulator(
        evolution_time=evolution_time,
        num_time_steps=num_time_steps,
        dimension=dimension,
        dynamic_operators=dynamic_operators,
        static_operators=static_operators,
        noise_operators=noise_operators,
        measurement_operators=measurement_operators,
        initial_states=initial_states,
        num_realizations=num_realizations,
        pulse_shape=pulse_shape,
        num_pulses=num_pulses,
        noise_profile=noise_profile,
        distortion=distortion
    )

    return simulator


def create_controlled_quantum_system(distortion=False):
    dimension = 2
    evolution_time = 1
    num_time_steps = 1024
    omega = 12
    dynamic_operators = [0.5 * pauli_operators[1]]
    static_operators = [0.5 * pauli_operators[3] * omega]
    noise_operators = [0.5 * pauli_operators[3]]
    measurement_operators = pauli_operators[1:]
    initial_states = [
        np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]),
        np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
        np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])
    ]
    num_realizations = 100
    num_pulses = 5
    noise_profile = ['Type 4']

    pulse_shape = "Square"

    simulator = create_custom_simulator(
        evolution_time=evolution_time,
        num_time_steps=num_time_steps,
        dimension=dimension,
        dynamic_operators=dynamic_operators,
        static_operators=static_operators,
        noise_operators=noise_operators,
        measurement_operators=measurement_operators,
        initial_states=initial_states,
        num_realizations=num_realizations,
        pulse_shape=pulse_shape,
        num_pulses=num_pulses,
        noise_profile=noise_profile,
        distortion=distortion
    )

    return simulator


def load_pulse_expectation(simulator, num_training, num_testing):

    training_input = []
    training_target = []

    for training_sample in range(num_training):

        experiment_result = run_experiment(simulator)
        training_input.append(np.expand_dims(experiment_result["pulses"][:, 0, :], axis=0))
        training_target.append(np.expand_dims(experiment_result["average_expectation"], axis=0))

    training_input = np.concatenate(training_input, axis=0)
    training_target = np.concatenate(training_target, axis=0)

    test_input = []
    test_target = []

    for test_sample in range(num_testing):

        experiment_result = run_experiment(simulator)
        test_input.append(np.expand_dims(experiment_result["pulses"][:, 0, :], axis=0))
        test_target.append(np.expand_dims(experiment_result["average_expectation"], axis=0))

    test_input = np.concatenate(test_input, axis=0)
    test_target = np.concatenate(test_target, axis=0)

    return training_input, training_target, test_input, test_target


def load_vo_dataset(simulator, num_examples):

    # initialize empty lists for storing the data
    pulses = []
    VX = []
    VY = []
    VZ = []

    for idx_ex in range(num_examples):

        experiment_result = run_experiment(simulator)

        pulses.append(np.expand_dims(experiment_result["pulses"][:, 0, :], axis=0))
        VX.append(np.expand_dims(experiment_result["average_vo"][0], axis=0))
        VY.append(np.expand_dims(experiment_result["average_vo"][1], axis=0))
        VZ.append(np.expand_dims(experiment_result["average_vo"][2], axis=0))

    pulses = np.concatenate(pulses, axis=0)
    VX = np.concatenate(VX, axis=0)
    VY = np.concatenate(VY, axis=0)
    VZ = np.concatenate(VZ, axis=0)

    print(pulses.shape)
    print(VX.shape)

    return pulses, VX, VY, VZ


def run_train_on_data(
        axnum, axnum2,
        training_input, training_target,
        testing_input, testing_target):
    input_layer = K.Input(shape=(None, axnum))
    output_layer = K.layers.LSTM(axnum2, return_sequences=False)(input_layer)
    ml_model = K.Model(input_layer, output_layer)
    ml_model.compile(optimizer=K.optimizers.Adam(), loss='mse')
    ml_model.fit(
        training_input, training_target,
        epochs=10, validation_data=(testing_input, testing_target))


def run_train_on_vo_data(axnum, axnum3, lstmflat, pulses, VX):
    input_layer = K.Input(shape=(None, axnum))

    output_layer = K.layers.Reshape((axnum3, axnum3))(K.layers.LSTM(lstmflat, return_sequences=False)(input_layer))

    ml_model = K.Model(input_layer, output_layer)

    ml_model.compile(optimizer=K.optimizers.Adam(), loss='mse')

    ml_model.fit(pulses, VX, epochs=10, validation_split=0.1)


def run_train_model_on_distorted_data(
        axnum, lstmout,
        training_input, training_target,
        testing_input, testing_target):
    input_layer = K.Input(shape=(None, axnum))

    output_layer = K.layers.LSTM(lstmout, return_sequences=False)(input_layer)

    ml_model = K.Model(input_layer, output_layer)

    ml_model.compile(optimizer=K.optimizers.Adam(), loss='mse')

    ml_model.fit(
        training_input, training_target,
        epochs=10, validation_data=(testing_input, testing_target))


def run_train_to_learn_quantum_controller(
        axnum4, axnum5,
        training_input, training_target,
        testing_input, testing_target):
    input_layer = K.Input(shape=axnum4)

    output_layer = K.layers.Reshape((axnum5,))(K.layers.Dense(axnum5)(input_layer))

    ml_model = K.Model(input_layer, output_layer)

    ml_model.compile(optimizer=K.optimizers.Adam(), loss='mse')

    ml_model.fit(
        training_input, training_target,
        epochs=10, validation_data=(testing_input, testing_target))


def characterize_open_quantum_system():
    # Characterization of an open quantum system
    simulator_open_system = create_open_quantum_system_simulator(distortion=False)

    num_training_ex = 7  # number of training examples
    num_testing_ex = 3  # number of testing examples

    training_input, \
    training_target, \
    testing_input, \
    testing_target = load_pulse_expectation(simulator_open_system, num_training_ex, num_testing_ex)

    axnum = training_input.shape[2]
    axnum2 = training_target.shape[1]

    print("Characterization of an open quantum system")
    run_train_on_data(axnum, axnum2, training_input, training_target, testing_input, testing_target)


def using_vo_operator():
    # Using the 𝑉𝑂 operators in a calculation
    simulator_controlled_system = create_controlled_quantum_system(distortion=False)
    num_examples = 10
    pulses, VX, VY, VZ = load_vo_dataset(simulator_controlled_system, num_examples)
    print(pulses.shape)
    axnum = pulses.shape[2]
    axnum3 = VX.shape[2]
    lstmflat = axnum3 ** 2

    print("Using the 𝑉𝑂 operators in a calculation")
    run_train_on_vo_data(axnum, axnum3, lstmflat, pulses, VX)


def modelling_control_distortion():
    # effect of distortion
    simulator_controlled_system = create_controlled_quantum_system(distortion=True)
    num_training_ex = 7  # number of training examples
    num_testing_ex = 3  # number of testing examples

    training_input, \
    training_target, \
    testing_input, \
    testing_target = load_pulse_expectation(simulator_controlled_system, num_training_ex, num_testing_ex)

    axnum = training_input.shape[2]
    lstmout = training_target.shape[1]

    print("Model the effect of control distortions")
    run_train_model_on_distorted_data(
        axnum, lstmout, training_input, training_target, testing_input, testing_target)


def learn_controller():
    # learning a controller
    simulator_controlled_system = create_controlled_quantum_system(distortion=True)
    num_training_ex = 7  # number of training examples
    num_testing_ex = 3  # number of testing examples

    training_input, \
    training_target, \
    testing_input, \
    testing_target = load_pulse_expectation(simulator_controlled_system, num_training_ex, num_testing_ex)

    print(training_input.shape)
    print(training_target.shape)
    axnum4 = training_input.shape[1]
    axnum5 = training_target.shape[1]

    print("Learning a controller for a quantum system")
    run_train_to_learn_quantum_controller(axnum4, axnum5, training_input, training_target, testing_input,
                                          testing_target)


if __name__ == "__main__":

    """Run one by one in case GPU is overloaded
    """

    # characterize_open_quantum_system()
    # using_vo_operator()
    # modelling_control_distortion()
    learn_controller()
