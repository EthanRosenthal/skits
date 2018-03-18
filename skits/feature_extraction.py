import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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


class TrendTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.expand_dims(np.arange(X.shape[0]), axis=1)


class FourierTransformer(BaseEstimator, TransformerMixin):

    needs_refit = True

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

