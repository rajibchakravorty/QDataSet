"""Generates Square, Gaussian and Zero pulses
"""

from typing import Tuple

from numpy import (
    array,
    int32,
    triu_indices
)

from tensorflow import (
    add_n,
    cast,
    concat,
    constant,
    divide,
    exp,
    greater,
    float32,
    logical_and,
    less,
    multiply,
    matmul,
    ones,
    random,
    reshape,
    shape,
    square,
    tile,
    zeros,
    Tensor
)


def create_core_parameters(
    max_control_pulse: int,
    total_time: float,
    num_time_steps: int
)->Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Creates basic configuration for the pulses

    :param total_time: Total time of evolution
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence

    :return: A tuple of tensor containing the time range of each pulse,
    pulse width,
    """
    time_range = constant(
        reshape([(0.5 * total_time / num_time_steps) +
                 (j * total_time / num_time_steps) for j in range(num_time_steps)],
                (1, num_time_steps, 1, 1)), dtype=float32)

    pulse_width = (0.5 * total_time / max_control_pulse)

    a_matrix = ones((max_control_pulse, max_control_pulse))
    a_matrix[triu_indices(max_control_pulse, 1)] = 0
    a_matrix = constant(
        reshape(a_matrix, (1, max_control_pulse, max_control_pulse)), dtype=float32)

    b_matrix = reshape(
        [idx + 0.5 for idx in range(max_control_pulse)], (1, max_control_pulse, 1)) * pulse_width
    b_matrix = constant(b_matrix, dtype=float32)

    return time_range, pulse_width, a_matrix, b_matrix


def square_control_pulse(
    inputs: Tensor,
    max_control_pulse: int,
    total_time: float,
    num_time_steps:int
):
    """Generates Square pulses and relevant parameters
    :param inputs: Input tensor to create square pulse
    """

    time_range, pulse_width, a_matrix, b_matrix = create_core_parameters(
        max_control_pulse,
        total_time,
        num_time_steps
    )

    # generate randomly the signal parameters
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([1, 1], dtype=int32))], 0)
    a_matrix = tile(a_matrix, temp_shape)
    b_matrix = tile(b_matrix, temp_shape)

    temp_shape = concat(
        [shape(inputs)[0:1],
         constant(array([max_control_pulse, 1], dtype=int32))], 0)

    amplitude = 100 * random.uniform(
        shape=temp_shape, minval=-1, maxval=1, dtype=float32)
    position = 0.5 * pulse_width + random.uniform(shape=temp_shape, dtype=float32) * (
            ((total_time - max_control_pulse * pulse_width) / (max_control_pulse + 1)) -
            0.5 * pulse_width)
    position = matmul(a_matrix, position) + b_matrix
    std = pulse_width * ones(temp_shape, dtype=float32)

    # combine the parameters into one tensor
    signal_parameters = concat([amplitude, position, std], -1)

    # construct the signal
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([1, 1, 1], dtype=int32))], 0)
    time_range = tile(time_range, temp_shape)
    tau = [
        reshape(matmul(position[:, idx, :], ones([1, num_time_steps])),
                (shape(time_range))) for idx in range(max_control_pulse)]
    A = [reshape(matmul(amplitude[:, idx, :], ones([1, num_time_steps])),
                 (shape(time_range))) for idx in range(max_control_pulse)]
    sigma = [
        reshape(matmul(std[:, idx, :], ones([1, num_time_steps])),
                (shape(time_range))) for idx in range(max_control_pulse)]
    signal = [multiply(A[idx], cast(logical_and(greater(time_range, tau[idx] - 0.5 * sigma[idx]),
                                                less(time_range, tau[idx] + 0.5 * sigma[idx])),
                                    float32)) for idx in range(max_control_pulse)]
    signal = add_n(signal)

    return signal_parameters, signal


def gaussian_control_pulse(
    inputs: Tensor,
    max_control_pulse: int,
    total_time: float,
    num_time_steps: int
):
    """
    Method to generate Gaussian pulses
    """

    time_range, pulse_width, a_matrix, b_matrix = create_core_parameters(
        max_control_pulse,
        total_time,
        num_time_steps
    )

    # generate randomly the signal parameters
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([1, 1], dtype=int32))], 0)
    a_matrix = tile(a_matrix, temp_shape)
    b_matrix = tile(b_matrix, temp_shape)

    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([max_control_pulse, 1], dtype=int32))], 0)
    amplitude = 100 * random.uniform(
        shape=temp_shape, minval=-1, maxval=1, dtype=float32)
    position = 0.5 * pulse_width + random.uniform(shape=temp_shape, dtype=float32) * (
                ((total_time - max_control_pulse * pulse_width) /
                 (max_control_pulse + 1)) - 0.5 * pulse_width)
    position = matmul(a_matrix, position) + b_matrix
    std = pulse_width * ones(temp_shape, dtype=float32) / 6

    # combine the parameters into one tensor
    signal_parameters = concat([amplitude, position, std], -1)

    # construct the signal
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([1, 1, 1], dtype=int32))], 0)
    time_range = tile(time_range, temp_shape)
    tau = [
        reshape(
            matmul(position[:, idx, :], ones([1, num_time_steps])),
            (shape(time_range))) for idx in range(max_control_pulse)]
    A = [
        reshape(
            matmul(amplitude[:, idx, :], ones([1, num_time_steps])),
            (shape(time_range))) for idx in range(max_control_pulse)]
    sigma = [
        reshape(matmul(std[:, idx, :], ones([1, num_time_steps])),
                (shape(time_range))) for idx in range(max_control_pulse)]

    signal = [
        multiply(A[idx], exp(-0.5 * square(divide(time_range - tau[idx], sigma[idx]))))
        for idx in range(max_control_pulse)]
    signal = add_n(signal)

    return signal_parameters, signal


def zero_control_pulse(
    inputs: Tensor,
    max_control_pulse: int,
    num_time_steps: int
):
    """
    Method to generate the zero pulse sequence [for free evolution analysis]
    """

    # construct zero signal
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([max_control_pulse, 3], dtype=int32))], 0)
    signal_parameters = zeros(temp_shape, dtype=float32)
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([num_time_steps, 1, 1], dtype=int32))], 0)
    signal = zeros(temp_shape, dtype=float32)

    return signal_parameters, signal
