import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

    
class AutoregressiveTransformer(BaseEstimator, TransformerMixin):

    needs_refit = True
    
    def __init__(self, num_lags=5, pred_stride=1):
        self.num_lags = num_lags
        self.pred_stride = pred_stride
    
    def fit(self, X, y=None):
        self._final_points = X[-self.pred_stride:, 0]
        return self

    def transform(self, X, y=None, refit=False):

        if refit:
            self.fit(X, y=y)
        
        check_is_fitted(self, '_final_points')

        X_trans = []
        for i in range(self.num_lags, 0, -1):
            X_trans.append([X[self.num_lags - i:-(i + self.pred_stride - 1), 0]])
            
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

            # First slice along the first window
            X[self.num_lags + self.pred_stride - 1, :],

            # Then, grab all lags up to the last
            X[self.num_lags + self.pred_stride:, -1],

            # And finally grab the last points
            np.atleast_1d(self._final_points)

        ))[:, np.newaxis]
    
    
class ReversibleImputer(BaseEstimator, TransformerMixin):

    needs_refit = True

    def __init__(self):
        pass
    
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
