import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

    
class ReversibleImputer(BaseEstimator, TransformerMixin):

    needs_refit = True

    def __init__(self):
        super().__init__()
    
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
