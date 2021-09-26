"""Sample use of qmldataset package
"""

from qmldataset import create_default_simulator, run_experiment


def create_experiment():
    """In this sample usage we will create of the default configurations
    configured in the package.
    """
    # experiment = '1q_X'
    # experiment = '1q_X_N1Z'
    # experiment = '1q_X_N2Z'
    # experiment = '1q_X_N3Z'
    # experiment = '1q_X_N4Z'
    # experiment = '1q_XY',
    # experiment = '1q_XY_N1X_N5Z'
    # experiment = '1q_XY_N1X_N6Z'
    # experiment = '1q_XY_N3X_N6Z'
    # experiment = '2q_IX_XI_XX'
    experiment = '2q_IX_XI_XX_N1N5IZ_N1N5ZI'
    # experiment = '2q_IX_XI_XX_N1N6IZ_N1N6ZI'
    # experiment = '2q_IX_XI_N1N6IZ_N1N6ZI'

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


if __name__ == '__main__':
    create_experiment()
