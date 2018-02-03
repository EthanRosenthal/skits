import numpy as np
import pytest

from skits.transformer import (ReversibleImputer, AutoregressiveTransformer,
                               DifferenceTransformer)


class TestReversibleImputer:

    def test_transform(self):
        ri = ReversibleImputer()
        X = np.array([1, 1, 100, 0, 2], dtype=np.float64)[:, np.newaxis]
        X[2] = np.nan
        Xt = ri.fit_transform(X)
        expected = np.array([1, 1, 1, 0, 2], dtype=np.float64)[:, np.newaxis]
        assert np.allclose(expected, Xt)

    def test_inverse_transform(self):
        ri = ReversibleImputer()
        X = np.random.random(20)[:, np.newaxis]
        X[[0, 5, 13], :] = np.nan
        X_inv = ri.inverse_transform(ri.fit_transform(X))
        assert np.allclose(X, X_inv, equal_nan=True)


class TestAutoregressiveTransfomer:

    def test_autoregressive_transform_1(self):
        X = np.arange(1, 6, dtype=np.float64)[:, np.newaxis]
        at = AutoregressiveTransformer(num_lags=2, pred_stride=1)
        X_trans = at.fit_transform(X)

        expected = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan],
            [1, 2],
            [2, 3],
            [3, 4]
        ], dtype=np.float64)
        assert np.allclose(X_trans, expected, equal_nan=True)

    def test_autoregressive_transform_2(self):
        X = np.arange(1, 6, dtype=np.float64)[:, np.newaxis]
        at = AutoregressiveTransformer(num_lags=2, pred_stride=2)
        X_trans = at.fit_transform(X)

        expected = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [1, 2],
            [2, 3],
        ], dtype=np.float64)
        assert np.allclose(X_trans, expected, equal_nan=True)

    def test_autoregressive_transform_3(self):
        X = np.arange(1, 9, dtype=np.float64)[:, np.newaxis]
        at = AutoregressiveTransformer(num_lags=3, pred_stride=2)
        X_trans = at.fit_transform(X)

        expected = np.array([
            [np.nan, np.nan, np.nan], # y = 1
            [np.nan, np.nan, np.nan], # y = 2
            [np.nan, np.nan, np.nan], # y = 3
            [np.nan, np.nan, np.nan], # y = 4
            [1, 2, 3],                # y = 5
            [2, 3, 4],                # y = 6
            [3, 4, 5],                # y = 7
            [4, 5, 6],                # y = 8
        ], dtype=np.float64)
        assert np.allclose(X_trans, expected, equal_nan=True)

    def test_autoregressive_inverse_transform(self):
        lt = AutoregressiveTransformer(num_lags=13, pred_stride=4)
        X = np.random.random((101, 1))
        X_trans = lt.fit_transform(X)
        X_inv = lt.inverse_transform(X_trans)
        assert np.allclose(X_inv, X)


class TestDifferenceTransformer:

    def test_difference_transformer_transform_explicit(self):
        # Differences should work
        X = np.arange(4, dtype=np.float32)[:, np.newaxis]
        dt = DifferenceTransformer()
        X_diff = dt.fit_transform(X)
        expected = np.array([np.nan, 1.0, 1.0, 1.0])[:, np.newaxis]
        assert np.allclose(expected, X_diff, equal_nan=True)

    def test_difference_transformer_double_transform_explicit(self):
        X = np.array([2, 4, 8, 16], dtype=np.float32)[:, np.newaxis]
        dt1 = DifferenceTransformer()
        dt2 = DifferenceTransformer()
        X_doubdiff = dt2.fit_transform(dt1.fit_transform(X))
        expected = np.array([np.nan, np.nan, 2.0, 4.0])[:, np.newaxis]
        assert np.allclose(expected, X_doubdiff, equal_nan=True)

    @pytest.mark.parametrize('shape, period', [
        (10, 1),
        (11, 1),
        (20, 2),
        (21, 2),
        (10, 3),
        (13, 3),
    ])
    def test_difference_transformer_inverse_transform(self, shape, period):
        X = np.random.random(shape)[:, np.newaxis]
        dt = DifferenceTransformer(period=period)
        assert np.allclose(X, dt.inverse_transform(dt.fit_transform(X)))

    @pytest.mark.parametrize('dt1, dt2', [
        (DifferenceTransformer(period=1), DifferenceTransformer(period=3)),
        (DifferenceTransformer(period=3), DifferenceTransformer(period=1)),
        (DifferenceTransformer(period=2), DifferenceTransformer(period=2)),
    ])
    def test_chained_difference_transformer(self, dt1, dt2):

        X = np.random.random(20)[:, np.newaxis]

        inv = dt1.inverse_transform(
                  dt2.inverse_transform(
                      dt2.fit_transform(
                          dt1.fit_transform(X)
                      )
                  )
              )

        assert np.allclose(X, inv)
