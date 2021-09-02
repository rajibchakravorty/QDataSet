"""Sample use of qmldataset package
"""

from qmldataset import run_default_experiment


def create_experiment():
    """In this sample usage we will create of the default experiments
    configured in the package.
    """

    # experiment = 'G_1q_X'
    # experiment = 'G_1q_XZ_N1'
    # experiment = 'G_1q_XZ_N2'
    experiment = 'G_1q_XZ_N3'
    num_examples = 2    # 2000
    batch_size = 10     # 50
    output_location = "/home/rchakrav/progs/qmldataset_result/"

    run_default_experiment(
        experiment_config=experiment,
        num_examples=num_examples,
        batch_size=batch_size,
        output_location=output_location
    )


if __name__ == '__main__':
    create_experiment()