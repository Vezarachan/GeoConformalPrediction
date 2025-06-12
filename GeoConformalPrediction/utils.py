import numpy as np
from numpy.typing import NDArray


def gaussian_kernel(d: NDArray) -> NDArray:
    """
    Gaussian distance decay function
    :param d: distances from test samples to calibration samples
    :return: list of weights for calibration samples
    """
    return np.exp(-0.5 * d ** 2)


def kernel_smoothing(z_test: NDArray, z_calib: NDArray, bandwidth: float) -> NDArray:
    """
    Kernel smoothing function
    :param z_test: the coordinates of test samples
    :param z_calib: the coordinates of calibration samples
    :param bandwidth: distance decay parameter
    :return: list of weights for calibration samples
    """
    z_test_norm = np.sum(z_test ** 2, axis=1).reshape(-1, 1)
    z_calib_norm = np.sum(z_calib ** 2, axis=1).reshape(1, -1)
    distances = np.sqrt(z_test_norm + z_calib_norm - 2 * np.dot(z_test, z_calib.T))
    weights = gaussian_kernel(distances / bandwidth)
    return weights


def weighted_quantile(scores: NDArray, weights: NDArray, q: float):
    """
    Calculate weighted quantile
    :param scores: nonconformity scores
    :param q: quantile level
    :param weights: geographic weights
    :return: weighted quantile at (1-alpha) miscoverage level
    """
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    N_test, N_calib = weights.shape
    scores = scores.reshape(1, -1)
    scores_repeated = np.repeat(scores, N_test, axis=0)
    sorted_idx = np.argsort(scores_repeated, axis=1)
    scores_sorted = np.take_along_axis(scores_repeated, sorted_idx, axis=1)
    weights_sorted = np.take_along_axis(weights, sorted_idx, axis=1)

    cumulative_weights = np.cumsum(weights_sorted, axis=1)
    total_weights = cumulative_weights[:, -1][:, None]
    normalized_cumulative_weights = cumulative_weights / total_weights
    idx = np.sum(normalized_cumulative_weights <= q, axis=1)
    quantiles = scores_sorted[np.arange(N_test), idx]
    return quantiles