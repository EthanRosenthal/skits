import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import FeatureUnion

from skits.feature_extraction import (AutoregressiveTransformer,
                                      SeasonalTransformer)
from skits.pipeline import ForecasterPipeline, ClassifierPipeline
from skits.preprocessing import ReversibleImputer, DifferenceTransformer


SEED = 666


class TestPipelines:

    steps = [
        ('pre_differencer', DifferenceTransformer(period=1)),
        ('pre_imputer_1', ReversibleImputer()),
        ('features', FeatureUnion([
            ('ar_transformer', AutoregressiveTransformer(num_lags=3)),
            ('seasonal_transformer', SeasonalTransformer(seasonal_period=4))])),
        ('post_lag_imputer_2', ReversibleImputer()),
    ]

    dt = DifferenceTransformer(period=1)
    ri1 = ReversibleImputer()
    fe = FeatureUnion([
         ('ar_transformer', AutoregressiveTransformer(num_lags=3)),
         ('seasonal_transformer', SeasonalTransformer(seasonal_period=4))])
    ri2 = ReversibleImputer()

    def test_predict(self):
        # Let's just see if it works
        # TODO: Make this a real test
        np.random.seed(SEED)
        l = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * 5 * l) + np.random.normal(0, .1, size=100)

        # Ignore the DifferenceTransformer. It's actually bad.
        steps = list(self.steps[1:])
        steps.append(('regressor', LinearRegression(fit_intercept=False)))

        pipeline = ForecasterPipeline(steps)

        pipeline.fit(y[:, np.newaxis], y)
        y_pred = pipeline.predict(y[:, np.newaxis], to_scale=True, refit=True)
        assert np.mean((y_pred - y.squeeze())**2) < 0.05

    def test_forecast(self):
        # Let's just see if it works
        # TODO: Make this a real test

        l = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * 5 * l) + np.random.normal(0, .1, size=100)

        steps = list(self.steps)
        steps.append(('regressor', LinearRegression(fit_intercept=False)))

        pipeline = ForecasterPipeline(steps)
        pipeline.fit(y[:, np.newaxis], y)

        pipeline.forecast(y[:, np.newaxis], 20)

    def test_classifier(self):
        # Let's just see if it works
        # TODO: Make this a real test
        np.random.seed(SEED)

        l = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * 5 * l) + np.random.normal(0, .1, size=100)

        steps = list(self.steps)
        steps.append(('classifier', LogisticRegression(fit_intercept=False)))

        pipeline = ClassifierPipeline(steps)

        y_true = y > 0
        pipeline.fit(y[:, np.newaxis], y_true)
        y_pred = pipeline.predict(y[:, np.newaxis])
        assert (y_pred == y_true).mean() > 0.75

