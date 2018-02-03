# skits
[![Build Status](https://travis-ci.org/EthanRosenthal/skits.svg?branch=master)](https://travis-ci.org/EthanRosenthal/skits)

A library for
**S**ci**K**it-**I**nspired **T**ime **S**eries models.

This library has two primary goals:

1. Train time series prediction models using a similar API to `scikit-learn`.
2. Allow for fully reversible pipelines, such that predictions can be directly compared to the original time series.

The library consists of `transformers` and `pipelines`. 

## Installation

Clone the library, create a virtual environment, and install the dependencies.

Do this in one fell swoop with conda

```bash
conda env create -f environment.yml -n skits
```

Or install with pip after creating a virtual environment

```bash
pip install -r requirements.txt
```

## Transfomers

The transformers expect to receive time series data, and then end up storing some data about the time series such that they can fully invert a transform. The following example shows how to create a `DifferenceTransformer`, which subtracts the point shifted by `period` away from each point, transform data and then invert it back to its original form.

```python
import numpy as np
from skits.transformer import DifferenceTransformer

y = np.random.random(10)
# scikit-learn expects 2D design matrices,
# so we duplicate the time series.
X = y[:, np.newaxis] 

dt = DifferenceTransformer(period=2)

Xt = dt.fit_transform(X,y)
X_inv = dt.inverse_transform(Xt)

assert np.allclose(X, X_inv)
```

## Pipelines

There are two types of pipelines. The `ForecasterPipeline` is for forecasting time series (duh). Specifically, one should build this pipeline with a regressor as the final step such that one can make appropriate predictions. The functionality is similar to a regular `scikit-learn` pipeline. Differences include the addition of a `forecast()` method along with a `to_scale` keyword argument to `predict()` such that one can make sure that their prediction is on the same scale as the original data.

These classes are likely subject to change as they are fairly hacky right now. For example, one must transform both `X` and `y` for all transformations before the introduction of an `AutoregressiveTransformer`. While the pipeline handles this, one must prefix all of these transformations with `pre_` in the step names.

Anywho, here's an example:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

from skits.pipeline import ForecasterPipeline
from skits.transformer import ReversibleImputer, AutoregressiveTransformer
                               
steps = [
    ('lag_transformer', AutoregressiveTransformer(num_lags=3)),
    ('post_lag_imputer_2', ReversibleImputer()),
    ('regressor', LinearRegression(fit_intercept=False))
]
                               
l = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * 5 * l) + np.random.normal(0, .1, size=100)
X = y[:, np.newaxis]

pipeline = ForecasterPipeline(steps)

pipeline.fit(X, y)
y_pred = pipeline.predict(X, to_scale=True, refit=True)
```

And this ends up looking like:

```python
import matplotlib.pyplot as plt

plt.plot(y)
plt.plot(y_pred)
plt.legend(['y_true', 'y_pred'])
```
![pred](pred.png)

And forecasting looks like

```python
start_idx = 70
plt.plot(y, lw=2);
plt.plot(pipeline.forecast(y[:, np.newaxis], start_idx=start_idx), lw=2);
ylim = (-1.2, 1.2)
plt.plot((start_idx, start_idx), ylim, lw=4);
plt.ylim(ylim);
plt.legend(['y_true', 'y_pred', 'forecast start'], bbox_to_anchor=(1, 1));
```
![forecast](forecast.png)