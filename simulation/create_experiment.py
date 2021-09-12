"""Sample use of qmldataset package
"""

from qmldataset import run_default_experiment


def create_experiment():
    """In this sample usage we will create of the default experiments
    configured in the package.
    """
    # experiment = '1q_X'
    # experiment = '1q_X_N1Z'
    # experiment = '1q_X_N2Z'
    # experiment = '1q_X_N3Z'
    experiment = '1q_X_N4Z'
    # experiment = '1q_XY',
    # experiment = '1q_XY_N1X_N5Z'
    # experiment = '1q_XY_N1X_N6Z'
    # experiment = '1q_XY_N3X_N6Z'
    # experiment = '2q_IX_XI_XX'
    # experiment = '2q_IX_XI_XX_N1N5IZ_N1N5ZI'
    # experiment = '2q_IX_XI_XX_N1N6IZ_N1N6ZI'
    # experiment = '2q_IX_XI_N1N6IZ_N1N6ZI'

    num_examples = 100  # other options, 10000
    batch_size = 5  # other options, 50
    output_location = "/home/rchakrav/progs/qmldataset_result/"
    pulse_shape = "Square"

    run_default_experiment(
        experiment_config=experiment,
        pulse_shape=pulse_shape,
        num_examples=num_examples,
        batch_size=batch_size,
        output_location=output_location
    )


if __name__ == '__main__':
    create_experiment()
