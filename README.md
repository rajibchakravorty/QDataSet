# QDataSet: Quantum Datasets for Machine Learning

## Overview 
Forked from the repository QDataSet introduced in [*QDataset: Quantum Datasets for Machine Learning* by Perrier, Youssry & Ferrie (2021)](https://arxiv.org/abs/2108.06661), 
a quantum dataset designed specifically to facilitate the training and development of QML algorithms. The QDataSet comprises 52 high-quality publicly available datasets derived 
from simulations of one- and two-qubit systems evolving in the presence and/or absence of noise.

The datasets are structured to provide a wealth of information to enable machine learning practitioners to use the 
QDataSet to solve problems in applied quantum computation, such as quantum control, quantum spectroscopy and tomography. Accompanying the datasets in this repository are a set of 
workbooks demonstrating the use of the QDataSet in a range of optimisation contexts.

See [Original repository](https://github.com/eperrier/QDataSet) for more detail.

This repository is a structured version of the code to provide with easy to use, install and run modules.

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
from qmldataset import run_default_experiment

experiment = '1q_XZ_N4'

num_examples = 10
batch_size = 5
output_location = "../qmldataset_result/"
pulse_shape = "Square"

run_default_experiment(
    experiment_config=experiment,
    pulse_shape=pulse_shape,
    num_examples=num_examples,
    batch_size=batch_size,
    output_location=output_location
)
```

This repository supports a set of default 1q and 2q experiments [similar to the original repository].
The other experiment options you can use are

```
'1q_X'
'1q_XZ_N1'
'1q_XZ_N2'
'1q_XZ_N3'
'1q_XZ_N4'
'1q_XY'
'1q_XY_XZ_N1N5'
'1q_XY_XZ_N1N6'
'1q_XY_XZ_N3N6'
'2q_IX_XI_XX'
'2q_IX_XI_XX_IZ_ZI_N1N5'
'2q_IX_XI_XX_IZ_ZI_N1N6'
'2q_IX_XI_IZ_ZI_N1N6'
```

For each option, you can use `"Square"` or `"Gaussian"` as `pulse_shape`.

#### Running a custom experiment

You can create the same experiment with custom options as shown below.
```
from qmldataset import run_custom_experiment

name = "1q_XZ_N4"
dimension = 2
evolution_time = 1
num_time_steps = 1024
omega = 12
dynamic_operators = [0.5*pauli_operators[1]]
static_operators = [0.5*pauli_operators[3]*omega]
noise_operators = [0.5*pauli_operators[3]]
measurement_operators = pauli_operators[1:]
initial_states = [
    np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]),
    np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
    np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])
]
num_realizations = 2000
num_pulses = 5
noise_profile = ['Type 4']
pulse_shape = "Gaussian"

num_examples = 10
batch_size = 5
output_location = "../qmldataset_result/"

run_custom_experiment(
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
    experiment_name=name,
    num_examples=num_examples,
    batch_size=batch_size,
    output_location=output_location
)
```

### Options

- `pulse_shape` can be one of `"Gaussian"` or `"Square"`
- `noise_profile` can be one of `"Type 0"`, `"Type 1"`, `"Type 2"`, `"Type 3"`, `"Type 4"`, `"Type 5"` and `"Type 6"`.

NOTE: `"Type 6"` is a correlated noise and must be used along with at least one other type.