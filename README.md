[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![PyPI](https://img.shields.io/badge/PyPI-3775A9?logo=pypi&logoColor=fff)](#)


# GeoConformal Prediction
## Purpose and Goals
A powerful, model-agnostic tool to measure spatially varying uncertainty of machine learning models. GeoConfromal is an extension of Conformal prediction.

GeoConformal, in theory, supports any machine learning model with spatial (e.g., coordinates) and aspatial (e.g., area of the living) features as the input.

## Usage
### Regression & Interpolation
```python
from GeoConformalPrediction import GeoConformalRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp, loc_train, loc_temp = train_test_split(X, y, loc, train_size=0.8, random_state=42)
X_calib, X_test, y_calib, y_test, loc_calib, loc_test = train_test_split(X_temp, y_temp, loc_temp, train_size=0.5, random_state=42)

model = XGBRegressor(n_estimators=500, max_depth=3, min_child_weight=1.0, colsample_bytree=1.0).fit(X_train.values, y_train.values)

geocp_regressoer = GeoConformalRegressor(predict_f=model.predict, x_calib=X_calib.values, y_calib=y_calib.values, coord_calib=loc_calib.values, bandwidth=0.15, miscoverage_level=0.1)

results = geocp_regressoer.geo_conformalize(X_test.values, y_test.values, loc_test.values)
```

### Classification
```python
from GeoConformalPrediction import GeoConformalClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp, loc_train, loc_temp = train_test_split(X, y, loc, train_size=0.8, random_state=42)
X_calib, X_test, y_calib, y_test, loc_calib, loc_test = train_test_split(X_temp, y_temp, loc_temp, train_size=0.5, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=2, min_child_weight=1.0, colsample_bytree=1.0).fit(X_train, y_train)

geocp_classifier = GeoConformalClassifier(predict_f=model.predict_proba, x_calib=X_calib.values, y_calib=y_calib, coord_calib=loc_calib.values, bandwidth=0.2, miscoverage_level=0.1, nonconformity_score='aps')

results = geocp_classifier.geo_conformalize(X_test.values, y_test.values, loc_test.values)
```

This repository hosts the code base for the paper

**GeoConformal prediction: a model-agnostic framework of measuring the uncertainty of spatial prediction** <br>
Xiayin Lou, Peng Luo, Liqiu Meng <br>
Annals of the American Association of Geographers <br>
[Link to Paper](https://arxiv.org/abs/2412.08661)

If you find this work useful, please consider cite:
```
@article{lou2024geoconformal,
  title={GeoConformal prediction: a model-agnostic framework of measuring the uncertainty of spatial prediction},
  author={Lou, Xiayin and Luo, Peng and Meng, Liqiu},
  journal={arXiv preprint arXiv:2412.08661},
  year={2024}
}
```