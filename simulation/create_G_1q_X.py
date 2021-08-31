"""Sample use of qmldataset package
"""

from qmldataset.experiments import run_default_experiment


def create_experiment():
    """In this sample usage we will create of the default experiments
    configured in the package.
    """

    experiment = 'G_1q_X'
    num_examples = 2
    batch_size = 10
    output_location = "./custom_result"

    run_default_experiment(
        experiment_config=experiment,
        num_examples=num_examples,
        batch_size=batch_size,
        output_location=output_location
    )


if __name__ == '__main__':
    create_experiment()