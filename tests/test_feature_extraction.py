import numpy as np

from skits.feature_extraction import (AutoregressiveTransformer,
                                      SeasonalTransformer)


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


class TestSeasonalTransfomer:

    def test_seasonal_sine_wave(self):
        """
        A sine wave fed through a SeasonalTransformer with its exact periodicity
        should return the same sine wave.

        This tests that the difference between the two is zero.
        """
        X = np.sin(np.linspace(-4, 4, 801) * 2 * np.pi)[:, np.newaxis]
        st = SeasonalTransformer(seasonal_period=100)
        Xt = st.fit_transform(X)
        diff = X - Xt
        err = diff[~np.isnan(diff)].sum()
        assert np.allclose(err, 0.0)
