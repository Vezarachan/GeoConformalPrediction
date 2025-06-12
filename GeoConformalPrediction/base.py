from typing import Callable, Tuple
from numpy.typing import NDArray


class GeoConformalBase:
    def __init__(self, predict_f: Callable, x_calib: NDArray, y_calib: NDArray, coord_calib: NDArray,
                 bandwidth: float | int, miscoverage_level: float = 0.1):
        self.predict_f = predict_f
        self.bandwidth = bandwidth
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.coord_calib = coord_calib
        self.miscoverage_level = miscoverage_level

    def geo_conformalized(self, x_test: NDArray, y_test: NDArray, coord_test: NDArray):
        raise NotImplementedError