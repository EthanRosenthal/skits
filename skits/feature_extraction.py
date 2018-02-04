import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AutoregressiveTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, num_lags=5, pred_stride=1):
        super().__init__()
        self.num_lags = num_lags
        self.pred_stride = pred_stride

    def fit(self, X, y=None):
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
