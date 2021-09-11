"""Sample use of qmldataset package
"""

from qmldataset import run_default_experiment


def create_experiment():
    """In this sample usage we will create of the default experiments
    configured in the package.
    """

    # experiment = '1q_X'
    # experiment = '1q_XZ_N1'
    # experiment = '1q_XZ_N2'
    # experiment = '1q_XZ_N3'
    experiment = '1q_XZ_N4'
    # experiment = '1q_XY'
    # experiment = '1q_XY_XZ_N1N5'
    # experiment = '1q_XY_XZ_N1N6'
    # experiment = '1q_XY_XZ_N3N6'
    # experiment = '2q_IX_XI_XX'
    # experiment = '2q_IX_XI_XX_IZ_ZI_N1N5'
    # experiment = '2q_IX_XI_XX_IZ_ZI_N1N6'
    # experiment = '2q_IX_XI_IZ_ZI_N1N6'

    num_examples = 1    # 10000
    batch_size = 1     # 50
    output_location = "../qmldataset_result/"
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
