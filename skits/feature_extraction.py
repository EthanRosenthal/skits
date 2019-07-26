import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AutoregressiveTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, num_lags=5, pred_stride=1):
        super().__init__()
        self.num_lags = num_lags
        self.pred_stride = pred_stride

    def fit(self, X, y=None):
        self._final_points = X[-self.pred_stride:, 0]
        return self

    def transform(self, X, y=None):

        X_trans = []
        for i in range(self.num_lags, 0, -1):
            X_trans.append(
                [X[self.num_lags - i:-(i + self.pred_stride - 1), 0]])

        X_trans = np.vstack(X_trans).T

        X_trans = self._fill_missing(X_trans)
        return X_trans

    def _fill_missing(self, X):
        missing_vals = np.zeros((self.num_lags + self.pred_stride - 1,
                                 self.num_lags),
                                dtype=X.dtype)
        missing_vals[:] = np.nan
        return np.vstack((missing_vals, X))

    def inverse_transform(self, X):
        return np.concatenate((
            X[self.num_lags + self.pred_stride - 1, :], # Slice along first window
            X[self.num_lags + self.pred_stride:, -1], # Then grab all single-lag up to last one
            np.atleast_1d(self._final_points))
        )[:, np.newaxis]


class SeasonalTransformer(AutoregressiveTransformer):

    def __init__(self, seasonal_period=1, pred_stride=1):
        pred_stride = seasonal_period + pred_stride - 1
        super().__init__(num_lags=1, pred_stride=pred_stride)


class IntegratedTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, num_lags=1, pred_stride=1):
        super().__init__()
        self.num_lags = num_lags
        self.pred_stride = pred_stride
        self.ar1 = AutoregressiveTransformer(
            num_lags=self.num_lags,
            pred_stride=self.pred_stride)
        self.ar2 = AutoregressiveTransformer(
            num_lags=self.num_lags,
            pred_stride=1 + self.pred_stride)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xt1 = self.ar1.transform(X)
        Xt2 = self.ar2.transform(X)
        return Xt1 - Xt2

class RollingMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window=5):
        super().__init__()
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        z2 = np.cumsum(np.pad(X, ((self.window,0),(0,0)), 'constant', constant_values=0), axis=0)
        z1 = np.cumsum(np.pad(X, ((0,self.window),(0,0)), 'constant', constant_values=X[-1]), axis=0)
        zc = (z1 - z2)[(self.window-1):-1]/self.window
        return np.vstack((np.repeat(np.nan,self.window).reshape(-1,1), zc[0:len(X)-self.window]))

class TrendTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, shift=0):
        super().__init__()
        self.shift=shift

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.expand_dims(np.arange(self.shift,self.shift+X.shape[0]), axis=1)


class FourierTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, period=10, max_order=10, step_size=1):
        super().__init__()
        self.period = period
        self.max_order = max_order
        self.step_size = step_size

    def fit(self, X, y=None):
        return self

    def _get_trig_args(self, X):
        trig_args = ((2 * np.pi / self.period)
                     * np.arange(1, self.max_order + 1, self.step_size))
        time = np.arange(X.shape[0])
        trig_args = trig_args[np.newaxis, :] * time[:, np.newaxis]
        return trig_args

    def transform(self, X, y=None):
        trig_args = self._get_trig_args(X)
        cos = np.cos(trig_args)
        sin = np.sin(trig_args)
        fourier_terms = np.hstack((cos, sin))
        return fourier_terms

