import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Callable
from numpy.typing import NDArray
from .utils import gaussian_kernel, kernel_smoothing, weighted_quantile


def abs_nonconformity_score_f(pred: NDArray, gt: NDArray) -> NDArray:
    """
            Default equation for computing nonconformity score
            :param pred: predicted values
            :param gt: ground truth values
            :return: list of nonconformity scores
            """
    return np.abs(pred - gt)

class GeoConformalRegressorResults:
    def __init__(self, geo_uncertainty: NDArray, uncertainty: float, coords: NDArray, pred_value: NDArray, true_value: NDArray,
                 upper_bound: NDArray, lower_bound: NDArray, coverage: float, crs: str = 'EPSG:4326'):
        self.uncertainty = uncertainty
        self.geo_uncertainty = geo_uncertainty
        self.coords = coords
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.pred_value = pred_value
        self.true_value = true_value
        self.coverage = coverage
        self.crs = crs

    def to_gpd(self) -> gpd.GeoDataFrame:
        result = np.column_stack([self.geo_uncertainty, self.pred_value, self.true_value, self.upper_bound, self.lower_bound, self.coords])
        geo_uncertainty_pd = pd.DataFrame(result)
        geo_uncertainty_pd.columns = ['geo_uncertainty', 'pred_value', 'true_value', 'upper_bound', 'lower_bound', 'x', 'y']
        geo_uncertainty_gpd = gpd.GeoDataFrame(geo_uncertainty_pd, crs=self.crs,
                                               geometry=gpd.points_from_xy(x=geo_uncertainty_pd.x,
                                                                           y=geo_uncertainty_pd.y))
        return geo_uncertainty_gpd


class GeoConformalRegressor:
    """
    Parameters
    ----------
    predict_f: spatial prediction function (regression or interpolation)
    """
    def __init__(self,
                 predict_f: Callable, x_calib: NDArray, y_calib: NDArray, coord_calib: NDArray,
                 bandwidth: float | int, miscoverage_level: float = 0.1):
        self.predict_f = predict_f
        self.bandwidth = bandwidth
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.coord_calib = coord_calib
        self.miscoverage_level = miscoverage_level

    def geo_conformalize(self,
                         x_test: NDArray,
                         y_test: NDArray,
                         coord_test: NDArray) -> GeoConformalRegressorResults:
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.array(abs_nonconformity_score_f(y_calib_pred, self.y_calib))
        N = nonconformity_scores.shape[0]
        q_level = np.ceil((1 - self.miscoverage_level) * (N + 1)) / N
        weights = kernel_smoothing(coord_test, self.coord_calib, self.bandwidth)
        geo_uncertainty = weighted_quantile(nonconformity_scores, weights, q_level)
        uncertainty = np.quantile(nonconformity_scores, q_level)
        y_test_pred = self.predict_f(x_test)
        upper_bound = y_test_pred + geo_uncertainty
        lower_bound = y_test_pred - geo_uncertainty
        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        return GeoConformalRegressorResults(geo_uncertainty, uncertainty, coord_test, y_test_pred, y_test, upper_bound, lower_bound, coverage)