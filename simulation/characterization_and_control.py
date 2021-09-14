"""Example of how to use dataset produced by qmldtaset package for
characterization and control using machine learning.
"""

import zipfile
import pickle
import os
import numpy as np

import tensorflow.keras as K

data_folder = "/home/rchakrav/progs/qmldataset_result"
undistorted_data_file = "S_1q_X_N4Z"
distorted_data_file = "S_1q_X_N4Z_distortion"


def load_dataset(
        data_folder,
        dataset_file,
        num_training_ex,
        num_testing_ex):
    """Function to extract the data from the dataset."""

    # initialize empt lists for storing the data
    data_input = []
    data_target = []

    # 1) unzip the dataset zipfile
    fzip = zipfile.ZipFile(
        os.path.join(data_folder, "{}.zip".format(dataset_file)), mode='r')
    example_list = fzip.namelist()

    # loop over example files
    #########################################################
    for example in example_list:
        # 2) extract the example file from the dataset
        fzip.extract(example)

        # 3) load the example file
        f = open(example, "rb")
        data = pickle.load(f)

        # 4) extract the useful information from the example file:

        # For noise spectroscopy, we need to extract a set of control pulse and the
        # correpsonding informationally-complete observables

        # add the pair of input and target to the corresponding lists
        data_input.append(data["pulses"][0:1, :, 0, :])
        data_target.append(data["expectations"])

        # 5) close and delete the example file
        f.close()
        os.remove(example)
    #########################################################
    # 5) close the dataset zipfile
    fzip.close()

    # 6) split the data into training and testing
    data_input = np.concatenate(data_input, axis=0)
    data_target = np.concatenate(data_target, axis=0)

    training_input = data_input[0:num_training_ex, :]
    testing_input = data_input[num_training_ex:num_training_ex + num_testing_ex, :]

    training_target = data_target[0:num_training_ex, :]
    testing_target = data_target[num_training_ex:num_training_ex + num_testing_ex, :]

    return training_input, training_target, testing_input, testing_target


def load_vo_dataset(data_folder, dataset_file, num_examples):
    # initialize empt lists for storing the data
    pulses = []
    VX = []
    VY = []
    VZ = []
    # 1) unzip the dataset zipfile
    fzip = zipfile.ZipFile(
        os.path.join(data_folder, "{}.zip".format(dataset_file)), mode='r')
    example_list = fzip.namelist()

    # loop over example files
    #########################################################
    for idx_ex in range(num_examples):
        # 2) extract the example file from the dataset
        fzip.extract(example_list[idx_ex])

        # 3) load the example file
        f = open(example_list[idx_ex], "rb")
        data = pickle.load(f)

        # 4) extract the useful information from the example file:

        # For noise spectroscopy, we need to extract a set of control pulse and the
        # correpsonding informationally-complete observables

        # add the pair of input and target to the corresponding lists
        pulses.append(data["pulses"][0:1, :, 0, :])
        VX.append(data["Vo_operator"][0])
        VY.append(data["Vo_operator"][1])
        VZ.append(data["Vo_operator"][2])

        # 5) close and delete the example file
        f.close()
        os.remove(example_list[idx_ex])
    #########################################################
    # 5) close the dataset zipfile
    fzip.close()

    # 6) split the data into training and testing
    pulses = np.concatenate(pulses, axis=0)
    VX = np.concatenate(VX, axis=0)
    VY = np.concatenate(VY, axis=0)
    VZ = np.concatenate(VZ, axis=0)

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

    output_layer = K.layers.Reshape((axnum5,))(K.layers.Dense(axnum5)(input_layer) )

    ml_model = K.Model(input_layer, output_layer)

    ml_model.compile(optimizer=K.optimizers.Adam(), loss='mse')

    ml_model.fit(
        training_input, training_target,
        epochs=10, validation_data=(testing_input, testing_target))


def run_ml():

    # use a small number of example for quick test
    num_training_ex = 7  # number of training examples
    num_testing_ex = 3  # number of testing examples

    training_input, \
    training_target, \
    testing_input, \
    testing_target = load_dataset(
        data_folder, undistorted_data_file, num_training_ex, num_testing_ex)

    axnum = training_input.shape[2]
    axnum2 = training_target.shape[1]

    print("Characterization of an open quantum system")
    run_train_on_data(axnum, axnum2, training_input, training_target, testing_input, testing_target)

    num_examples = 10
    pulses, VX, VY, VZ = load_vo_dataset(data_folder, undistorted_data_file, num_examples)
    axnum3 = VX.shape[2]
    lstmflat = axnum3 ** 2

    print("Using the 𝑉𝑂 operators in a calculation")
    run_train_on_vo_data(axnum, axnum3, lstmflat, pulses, VX)

    num_training_ex = 7  # number of training examples
    num_testing_ex = 3  # number of testing examples

    training_input, \
    training_target, \
    testing_input, \
    testing_target = load_dataset(
        data_folder, distorted_data_file, num_training_ex, num_testing_ex)

    axnum = training_input.shape[2]
    lstmout = training_target.shape[1]

    print("Model the effect of control distortions")
    run_train_model_on_distorted_data(axnum, lstmout, training_input, training_target, testing_input, testing_target)

    num_training_ex = 7  # number of training examples
    num_testing_ex = 3  # number of testing examples

    training_input, \
    training_target, \
    testing_input, \
    testing_target = load_dataset(
        data_folder, undistorted_data_file, num_training_ex, num_testing_ex)

    print(training_input.shape)
    print(training_target.shape)
    axnum4 = training_input.shape[1]
    axnum5 = training_target.shape[1]

    print("Learning a controller for a quantum system")
    run_train_to_learn_quantum_controller(axnum4, axnum5, training_input, training_target, testing_input, testing_target)
    

if __name__ == "__main__":
    run_ml()
