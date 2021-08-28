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


def create_control_pulses(
    max_control_pulse: int,
    total_time: float,
    num_time_steps: int,
    inputs: Tensor,
    waveform: str = "Zero",
) -> Tuple[Tensor, Tensor]:
    """API method to create a control pulse with defined shape.

    :param inputs: Input tensor
    :param total_time: Total time of evolution
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence
    :param waveform: One of 'Gaussian', 'Square', 'Zero'; defaults to 'Zero'
    :returns: A tuple of tensors (signal parameters and signal)

    """

    signal_parameters, signal = None, None
    if waveform == 'Gaussian':
        signal_parameters, signal = gaussian_control_pulse(
            inputs,
            max_control_pulse,
            total_time,
            num_time_steps
        )
    elif waveform == 'Square':
        signal_parameters, signal = square_control_pulse(
            inputs,
            max_control_pulse,
            total_time,
            num_time_steps
        )
    else:
        signal_parameters, signal = zero_control_pulse(
            inputs,
            max_control_pulse,
            num_time_steps
        )

    return signal_parameters, signal


def create_core_parameters(
    max_control_pulse: int,
    total_time: float,
    num_time_steps: int,
    inputs: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Creates basic configuration for the pulses

    :param total_time: Total time of evolution
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence
    :param inputs: Input Tensor
    :return: A tuple of tensors containing the time range of each pulse,
    pulse width, and 2 matrices; the matrices determine the position of the pulses
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

    # generate randomly the signal parameters
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([1, 1], dtype=int32))], 0)
    a_matrix = tile(a_matrix, temp_shape)
    b_matrix = tile(b_matrix, temp_shape)

    return time_range, pulse_width, a_matrix, b_matrix


def square_control_pulse(
    inputs: Tensor,
    max_control_pulse: int,
    total_time: float,
    num_time_steps: int
) -> Tuple[Tensor, Tensor]:
    """Generates Square pulses and relevant parameters
    :param inputs: Input tensor
    :param total_time: Total time of evolution
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence

    :returns: A tuple of tensors (signal parameters and signal)
    """

    time_range, pulse_width, a_matrix, b_matrix = create_core_parameters(
        max_control_pulse,
        total_time,
        num_time_steps,
        inputs
    )

    temp_shape = concat(
        [shape(inputs)[0:1],
         constant(array([max_control_pulse, 1], dtype=int32))], 0)

    amplitude = 100 * random.uniform(
        temp_shape, -1, 1, float32)
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
    signal_amplitudes = [reshape(matmul(amplitude[:, idx, :], ones([1, num_time_steps])),
                 (shape(time_range))) for idx in range(max_control_pulse)]
    sigma = [
        reshape(matmul(std[:, idx, :], ones([1, num_time_steps])),
                (shape(time_range))) for idx in range(max_control_pulse)]
    signal = [multiply(signal_amplitudes[idx],
                       cast(logical_and(greater(time_range, tau[idx] - 0.5 * sigma[idx]),
                                        less(time_range, tau[idx] + 0.5 * sigma[idx])),
                            float32)) for idx in range(max_control_pulse)]
    signal = add_n(signal)

    return signal_parameters, signal


def gaussian_control_pulse(
    inputs: Tensor,
    max_control_pulse: int,
    total_time: float,
    num_time_steps: int
) -> Tuple[Tensor, Tensor]:
    """
    Method to generate Gaussian pulses
    :param inputs: Input tensor
    :param total_time: Total time of evolution
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence

    :returns: A tuple of tensors (signal parameters and signal)
    """

    time_range, pulse_width, a_matrix, b_matrix = create_core_parameters(
        max_control_pulse,
        total_time,
        num_time_steps,
        inputs
    )

    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([max_control_pulse, 1], dtype=int32))], 0)
    amplitude = 100 * random.uniform(
        temp_shape, -1, 1, float32)
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
    signal_amplitudes = [
        reshape(
            matmul(amplitude[:, idx, :], ones([1, num_time_steps])),
            (shape(time_range))) for idx in range(max_control_pulse)]
    sigma = [
        reshape(matmul(std[:, idx, :], ones([1, num_time_steps])),
                (shape(time_range))) for idx in range(max_control_pulse)]

    signal = [
        multiply(
            signal_amplitudes[idx],
            exp(-0.5 * square(divide(time_range - tau[idx], sigma[idx]))))
        for idx in range(max_control_pulse)]
    signal = add_n(signal)

    return signal_parameters, signal


def zero_control_pulse(
    inputs: Tensor,
    max_control_pulse: int,
    num_time_steps: int
) -> Tuple[Tensor, Tensor]:
    """
    Method to generate the zero pulse sequence [for free evolution analysis]
    :param inputs: Input tensor
    :param num_time_steps: Number of discrete time steps
    :param max_control_pulse: Maximum number of control pulses in the sequence
    :returns: A tuple of tensors (signal parameters and signal)

    """

    # construct zero signal
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([max_control_pulse, 3], dtype=int32))], 0)
    signal_parameters = zeros(temp_shape, dtype=float32)
    temp_shape = concat(
        [shape(inputs)[0:1], constant(array([num_time_steps, 1, 1], dtype=int32))], 0)
    signal = zeros(temp_shape, dtype=float32)

    return signal_parameters, signal
