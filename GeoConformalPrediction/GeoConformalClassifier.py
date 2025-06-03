import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Callable, Tuple
from numpy.typing import NDArray
from .utils import weighted_quantile, kernel_smoothing


def tps(calib_probs: NDArray, calib_labels: NDArray, test_probs: NDArray, test_labels: NDArray, alpha: float, weights: NDArray) -> Tuple[NDArray, float, float, NDArray]:
    N = calib_probs.shape[0]
    calib_scores = 1 - calib_probs[np.arange(N), calib_labels]
    q_level = np.ceil((1 - alpha) * (N + 1)) / N
    qhat = weighted_quantile(calib_scores, weights, q_level)
    qhat = qhat[:, None]
    prediction_sets = test_probs >= (1 - qhat.reshape(-1, 1))
    N_test = test_probs.shape[0]
    coverage = prediction_sets[np.arange(N_test), test_labels].mean()
    avg_set_size = np.sum(prediction_sets) / len(prediction_sets)
    geo_set_sizes = np.sum(prediction_sets, axis=1)
    return prediction_sets, coverage, avg_set_size, geo_set_sizes

def aps(calib_probs: NDArray, calib_labels: NDArray, test_probs: NDArray, test_labels: NDArray, alpha: float, weights: NDArray) -> Tuple[NDArray, float, float, NDArray]:
    N = calib_probs.shape[0]
    # sort by descending order
    sorted_calib_idx = np.argsort(calib_probs)[:, ::-1]
    # Cumulated calibration probabilities
    sorted_calib_probs = np.take_along_axis(calib_probs, sorted_calib_idx, axis=1).cumsum(axis=1)
    calib_scores = np.take_along_axis(sorted_calib_probs, np.argsort(sorted_calib_idx, axis=1), axis=1)[np.arange(N), calib_labels]
    q_level = np.ceil((1 - alpha) * (N + 1)) / N
    qhat = weighted_quantile(calib_scores, weights, q_level)
    qhat = qhat[:, None]
    sorted_test_idx = np.argsort(test_probs)[:, ::-1]
    sorted_test_probs = np.take_along_axis(test_probs, sorted_test_idx, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(sorted_test_probs <= qhat, np.argsort(sorted_test_idx, axis=1), axis=1)
    N_test = test_probs.shape[0]
    coverage = prediction_sets[np.arange(N_test), test_labels].mean()
    avg_set_size = np.sum(prediction_sets) / len(prediction_sets)
    geo_set_sizes = np.sum(prediction_sets, axis=1)
    return prediction_sets, coverage, avg_set_size, geo_set_sizes

def raps(calib_probs: NDArray, calib_labels: NDArray, test_probs: NDArray, test_labels: NDArray, alpha: float, weights: NDArray, lam_reg: float = 0.01, k_reg: int = 3, randomized: bool = True, disallow_zero_sets: bool = False) -> Tuple[NDArray, float, float, NDArray]:
    N_calib, k_classes_calib = calib_probs.shape
    # sort calibration probabilities in descending order
    sorted_calib_idx = np.argsort(calib_probs)[:, ::-1]
    sorted_calib_probs = np.take_along_axis(calib_probs, sorted_calib_idx, axis=1)
    # compute regularization vector
    reg_vec = np.array(k_reg * [0,] + (k_classes_calib - k_reg) * [lam_reg,])[None, :]
    sorted_calib_probs_reg = sorted_calib_probs + reg_vec
    calib_L = np.where(sorted_calib_idx == calib_labels[:, None])[1]
    calib_scores = sorted_calib_probs_reg.cumsum(axis=1)[np.arange(N_calib), calib_L] - np.random.rand(N_calib) * sorted_calib_probs_reg[np.arange(N_calib), calib_L]
    q_level = np.ceil((1 - alpha) * (N_calib + 1)) / N_calib
    qhat = weighted_quantile(calib_scores, weights, q_level)
    qhat = qhat[:, None]
    N_test = test_probs.shape[0]
    sorted_test_idx = np.argsort(test_probs)[:, ::-1]
    sorted_test_probs = np.take_along_axis(test_probs, sorted_test_idx, axis=1)
    sorted_test_probs_reg = sorted_test_probs + reg_vec
    indicators = (sorted_test_probs_reg.cumsum(axis=1) - np.random.rand(N_test, 1) * sorted_test_probs_reg) <= qhat if randomized else sorted_test_probs_reg.cumsum(axis=1) - sorted_test_probs_reg <= qhat
    if disallow_zero_sets: indicators[:, 0] = True
    prediction_sets = np.take_along_axis(indicators, np.argsort(sorted_test_idx, axis=1), axis=1)
    coverage = prediction_sets[np.arange(N_test), test_labels].mean()
    avg_set_size = np.sum(prediction_sets) / len(prediction_sets)
    geo_set_sizes = np.sum(prediction_sets, axis=1)
    return prediction_sets, coverage, avg_set_size, geo_set_sizes

def topk(calib_probs: NDArray, calib_labels: NDArray, test_probs: NDArray, test_labels: NDArray, alpha: float, weights: NDArray) -> Tuple[NDArray, float, float, NDArray]:
    N_calib = calib_probs.shape[0]
    # sort by descending order
    sorted_calib_idx = np.argsort(calib_probs, axis=1)[:, ::-1]
    # get rank positions of the true labels
    calib_label_rank = np.argsort(sorted_calib_idx, axis=1)
    calib_scores = calib_label_rank[np.arange(N_calib), calib_labels]
    q_level = np.ceil((1 - alpha) * (N_calib + 1)) / N_calib
    qhat = weighted_quantile(calib_scores, weights, q_level)
    qhat = qhat[:, None]
    N_test = test_probs.shape[0]
    sorted_test_idx = np.argsort(test_probs)[:, ::-1]
    test_label_rank = np.argsort(sorted_test_idx, axis=1)
    prediction_sets = np.take_along_axis(test_label_rank <= qhat, np.argsort(sorted_test_idx, axis=1), axis=1)
    coverage = prediction_sets[np.arange(N_test), test_labels].mean()
    avg_set_size = np.sum(prediction_sets) / len(prediction_sets)
    geo_set_sizes = np.sum(prediction_sets, axis=1)
    return prediction_sets, coverage, avg_set_size, geo_set_sizes

class GeoConformalClassifierResult:
    def __init__(self, prediction_sets: NDArray, geo_uncertainty: NDArray, uncertainty: float, coords: NDArray,
                 pred_value: NDArray, true_value: NDArray, coverage: float, crs: str = 'EPSG:4326'):
        self.prediction_sets = prediction_sets
        self.geo_uncertainty = geo_uncertainty  # set size
        self.uncertainty = uncertainty # avg set size
        self.coords = coords
        self.pred_value = pred_value
        self.true_value = true_value
        self.coverage = coverage
        self.crs = crs

    def to_gpd(self) -> gpd.GeoDataFrame:
        result = np.column_stack([self.geo_uncertainty, self.pred_value, self.true_value, self.coords])
        geo_uncertainty_pd = pd.DataFrame(result)
        geo_uncertainty_pd.columns = ['geo_uncertainty', 'pred_value', 'true_value', 'x', 'y']
        geo_uncertainty_gpd = gpd.GeoDataFrame(geo_uncertainty_pd, crs=self.crs,
                                               geometry=gpd.points_from_xy(x=geo_uncertainty_pd.x,
                                                                           y=geo_uncertainty_pd.y))
        return geo_uncertainty_gpd

class GeoConformalClassifier:
    def __init__(self, predict_f: Callable, x_calib: NDArray, y_calib: NDArray, coord_calib: NDArray,
                 bandwidth: float | int, miscoverage_level: float = 0.1, nonconformity_score: str = 'aps'):
        self.predict_f = predict_f
        self.bandwidth = bandwidth
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.coord_calib = coord_calib
        self.miscoverage_level = miscoverage_level
        self.nonconformity_score = nonconformity_score

    def geo_conformalized(self, x_test: NDArray, y_test: NDArray, coord_test: NDArray):
        y_calib_probs = self.predict_f(self.x_calib)
        y_test_probs = self.predict_f(x_test)
        y_test_pred = np.argmax(y_test_probs, axis=1)
        weights = kernel_smoothing(coord_test, self.coord_calib, self.bandwidth)
        if self.nonconformity_score == 'aps':
            prediction_sets, coverage, avg_set_size, geo_set_size = aps(y_calib_probs, self.y_calib, y_test_probs, y_test, self.miscoverage_level, weights)
        elif self.nonconformity_score == 'raps':
            prediction_sets, coverage, avg_set_size, geo_set_size = raps(y_calib_probs, self.y_calib, y_test_probs, y_test, self.miscoverage_level, weights)
        elif self.nonconformity_score == 'tps':
            prediction_sets, coverage, avg_set_size, geo_set_size = tps(y_calib_probs, self.y_calib, y_test_probs, y_test, self.miscoverage_level, weights)
        elif self.nonconformity_score == 'topk':
            prediction_sets, coverage, avg_set_size, geo_set_size = topk(y_calib_probs, self.y_calib, y_test_probs, y_test, self.miscoverage_level, weights)
        return GeoConformalClassifierResult(prediction_sets, geo_set_size, avg_set_size, coord_test, y_test_pred, y_test, coverage)
