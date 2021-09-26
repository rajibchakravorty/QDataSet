# QDataSet: Quantum Datasets for Machine Learning

## Overview 
Forked from the repository QDataSet introduced in [*QDataset: Quantum Datasets for Machine Learning* by Perrier, Youssry & Ferrie (2021)](https://arxiv.org/abs/2108.06661), 
a quantum dataset designed specifically to facilitate the training and development of QML algorithms. The QDataSet comprises 52 high-quality publicly available datasets derived 
from simulations of one- and two-qubit systems evolving in the presence and/or absence of noise.

See [Original repository](https://github.com/eperrier/QDataSet) for more detail.

This repository is a structured version of the code to provide with easy to use, install and run modules. Moreover, this repository does not save the experiment outcome and rather 
provides each iteration of experiment available in memory.

### Set up and Installation

We assume you have a environment (this package is tested in Anaconda Environment) and cloned this repo.
Repo folder is called `$QDATASET`

- Install Poetry
```
$QDATASET > curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
$QDATASET > poetry install
$QDATASET > python setup.py install 
```

### Usage

#### Running one of the predefined experiments

```
from qmldataset import create_default_simulator, run_experiment


def create_experiment():
    """In this sample usage we will create of the default configurations
    configured in the package.
    """
    experiment = '2q_IX_XI_XX_N1N5IZ_N1N5ZI'

    pulse_shape = "Square"
    distortion = True
    num_realizations = 100

    simulator = create_default_simulator(
        experiment_name=experiment,
        distortion=distortion,
        num_realizations=num_realizations,
        pulse_shape=pulse_shape
    )

    # run and gather of one experiment result
    experiment_result = run_experiment(
        simulator=simulator
    )

    for param in experiment_result:
        print("-- {} --".format(param))
        print("-- {} --".format(experiment_result[param]))

```

The code snippet above runs the simulator for one experiment (generating simulated outcome with a predefined number of noise realizations).

This repository supports a set of default 1q and 2q experiments [similar to the original repository].
The experiment options you can use are

```
'1q_X' - 1-qubit, Control X-Axis, No Noise
'1q_X_N1Z' - 1-qubit, Control X-Axis, Type 1 Noise on Z-Axis
'1q_X_N2Z' - 1-qubit, Control X-Axis, Type 2 Noise on Z-Axis
'1q_X_N3Z' - 1-qubit, Control X-Axis, Type 3 Noise on Z-Axis
'1q_X_N4Z' - 1-qubit, Control X-Axis, Type 4 Noise on Z-Axis
'1q_XY' - 1-qubit, Control X and Y-Axis, No Noise
'1q_XY_N1X_N5Z' - 1-qubit, Control X and Y-Axis, Type 1 and Type 5 Noises on Z - Axis
'1q_XY_N1X_N6Z' - 1-qubit, Control X and Y-Axis, Type 1 and Type 6 Noises on Z - Axis
'1q_XY_N3X_N6Z' - 1-qubit, Control X and Y-Axis, Type 3 and Type 5 Noises on Z - Axis
'2q_IX_XI_XX' - 2-qubit, Control X Axis on both qubits and interacting X-Axis, No Noise
'2q_IX_XI_N1N6IZ_N1N6ZI' - 2-qubit, Control X Axis on both qubits, Type 1 and Type 6 Noises on Z-Axis on both qubits
'2q_IX_XI_XX_N1N5IZ_N1N5ZI' - 2-qubit, Control X Axis on both qubits and interacting X-Axis, Type 1 and Type 5 Noises on Z-Axis on both qubits
'2q_IX_XI_XX_N1N6IZ_N1N6ZI' - 2-qubit, Control X Axis on both qubits and interacting X-Axis, Type 1 and Type 6 Noises on Z-Axis on both qubits
```

For each option, you can use `"Square"` or `"Gaussian"` as `pulse_shape`.

#### Running a custom experiment

You can create the same experiment with custom options as shown below.
```
from itertools import product
from numpy import array, kron
from qmldataset import pauli_operators, create_custom_simulator, run_experiment


def create_experiment():
    """In this sample usage we will create a custom experiment
    """
    dimension = 4
    evolution_time = 1
    num_time_steps = 1024
    omega = [10, 12]
    dynamic_operators = [
        kron(pauli_operators[1], pauli_operators[0]),
        kron(pauli_operators[0], pauli_operators[1]),
        kron(pauli_operators[1], pauli_operators[1])]
    static_operators = [
        omega[0] * kron(pauli_operators[3], pauli_operators[0]),
        omega[1] * kron(pauli_operators[0], pauli_operators[3])]
    noise_operators = [
        kron(pauli_operators[3], pauli_operators[0]),
        kron(pauli_operators[0], pauli_operators[3])]
    measurement_operators = [
        kron(meas_op_one, meas_op_two) for meas_op_one, meas_op_two in list(
            product(pauli_operators, pauli_operators))][1:]
    initial_states_1q = [
        array([[0.5, 0.5], [0.5, 0.5]]), array([[0.5, -0.5], [-0.5, 0.5]]),
        array([[0.5, -0.5j], [0.5j, 0.5]]), array([[0.5, 0.5j], [-0.5j, 0.5]]),
        array([[1, 0], [0, 0]]), array([[0, 0], [0, 1]])
    ]
    initial_states = [
        kron(init_state_one, init_state_two) for init_state_one, init_state_two in list(
            product(initial_states_1q, initial_states_1q))]
    num_realizations = 500
    num_pulses = 5
    noise_profile = ['Type 1', 'Type 5']
    pulse_shape = "Square"
    distortion = True

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

    # run and gather of one experiment result
    experiment_result = run_experiment(
        simulator=simulator
    )

    for param in experiment_result:
        print("-- {} --".format(param))
        print("-- {} --".format(experiment_result[param]))
```

### Options

- `pulse_shape` can be one of `"Gaussian"` or `"Square"`
- `noise_profile` is a list containing one or more of `"Type 0"`, `"Type 1"`, `"Type 2"`, `"Type 3"`, `"Type 4"`, `"Type 5"` and `"Type 6"`.

NOTE: `"Type 6"` is a correlated noise and must be used along with at least one other type.