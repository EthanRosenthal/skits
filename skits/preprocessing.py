import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skits.feature_extraction import AutoregressiveTransformer


def expand_dim_if_needed(arr):
    if arr.ndim == 1:
        return np.expand_dims(arr, axis=1)
    return arr


class ReversibleImputer(BaseEstimator, TransformerMixin):

    needs_refit = True

    def __init__(self, y_only=False):
        super().__init__()
        self.y_only = y_only

    def fit(self, X, y=None):
        mask = np.isnan(X)
        self._missing_idxs = np.where(mask)
        self._mean = X[~mask].mean()
        return self

    def transform(self, X, y=None, refit=False):
        if refit:
            self.fit(X, y=y)

        check_is_fitted(self, '_mean')
        X[self._missing_idxs] = self._mean
        return X

    def inverse_transform(self, X):
        X[self._missing_idxs] = np.nan
        return X


class DifferenceTransformer(BaseEstimator, TransformerMixin):

    needs_refit = True

    def __init__(self, period=1):
        super().__init__()
        self.period = period

    def fit(self, X, y=None):
        missing = np.where(np.isnan(X))[0]

        if len(missing) > 0:
            self._missing_idx_start = np.max(missing) + 1
        else:
            self._missing_idx_start = 0

        self._missing_idx_end = self._missing_idx_start + self.period
        self._missing_vals = X[self._missing_idx_start:self._missing_idx_end, :]
        return self

    def transform(self, X, y=None, refit=False):
        if refit:
            self.fit(X, y=y)

        check_is_fitted(self, '_missing_idx_start')

        X_shift = np.roll(X, self.period, axis=0)
        X_trans = X - X_shift
        X_trans[:self._missing_idx_end, :] = np.nan

        return X_trans

    def inverse_transform(self, X):
        # TODO: Figure out why I have to make a copy here
        X = X.copy()

        # Fill in the missing values
        X[self._missing_idx_start:self._missing_idx_end, :] = self._missing_vals
        X_inv = X[self._missing_idx_start:, :]

        # Take stock
        period_rem = np.remainder(len(X_inv), self.period)
        stride_shape = [self.period,
                        int(np.floor(len(X_inv) / self.period) + period_rem)]
        stride = X.strides[0]

        # Clusterfuck
        inv = (as_strided(X_inv,
                          shape=stride_shape,
                          strides=[stride, stride * self.period])
               .cumsum(axis=1)
               .reshape(stride_shape[0] * stride_shape[1], order='F')
               [:len(X_inv)])[:, np.newaxis]

        return np.vstack((X[:self._missing_idx_start, :], inv))


class LogTransformer(BaseEstimator, TransformerMixin):

    needs_refit = False

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, refit=False):
        with np.errstate(divide='raise', invalid='raise'):
            try:
                Xt = np.log(1+X)
            except FloatingPointError:
                raise ValueError('X cannot have negative values')
        return Xt

    def inverse_transform(self, X):
        return np.exp(X)-1


class HorizonTransformer(BaseEstimator, TransformerMixin):

    needs_refit = True
    y_only = True

    def __init__(self, horizon=2):
        super().__init__()
        if horizon < 2:
            raise ValueError('horizon must be greater than 1')
        self.horizon = horizon
        self.autoregressive_transformer = AutoregressiveTransformer(
            num_lags=self.horizon,
            pred_stride=1
        )

    def fit(self, X, y=None):
        self.autoregressive_transformer.fit(expand_dim_if_needed(X))
        return self

    def transform(self, X, y=None, refit=False):
        X = expand_dim_if_needed(X)
        if refit:
            self.autoregressive_transformer.fit(X)
        Xt = self.autoregressive_transformer.transform(X)
        # Need to move beginning of Xt to the end.
        Xt = np.vstack((Xt[self.horizon:, :], Xt[:self.horizon, :]))
        # TODO: replace beginning with nans?
        return Xt

    def inverse_transform(self, X, y=None):
        Xt = np.vstack((X[-self.horizon:, :], X[:-self.horizon, :]))
        return self.autoregressive_transformer.inverse_transform(Xt)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

