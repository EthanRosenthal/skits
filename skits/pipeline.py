import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory

from skits.preprocessing import expand_dim_if_needed


def _fit_transform_one(transformer, weight, X, y,
                       **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer


def _needs_refit(transform):

    return hasattr(transform, 'needs_refit') and transform.needs_refit


class _BasePipeline(Pipeline):

    def __init__(self, steps, memory=None):
        super().__init__(steps, memory=memory)
        self._skits_validation()

    def _skits_validation(self):
        feature_extractors_found = False
        for name, step in self.steps:
            if name.startswith('pre') and feature_extractors_found:
                raise ValueError('All preprocessors must come before '
                                 'feature extractors')
            elif isinstance(step, FeatureUnion):
                feature_extractors_found = True

    def _transform(self, X, refit=False):
        Xt = X
        for name, transform in self.steps:
            if transform is not None:
                if _needs_refit(transform):
                    Xt = transform.transform(Xt, refit=True)
                else:
                    Xt = transform.transform(Xt)
        return Xt


class ForecasterPipeline(_BasePipeline):

    def __init__(self, steps, memory=None):
        super().__init__(steps, memory=memory)

    def _fit(self, X, y=None, **fit_params):
        """
        All of this stolen from scikit-learn except for
        "if name.startsiwth('pre_')..." at the bottom
        """
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                # For the HorizonTransformer right now.
                y_only = getattr(transformer, 'y_only', False)
                _Xt = y.copy() if y_only else Xt

                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                _Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, _Xt, y,
                    **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)

                if y_only:
                    y = _Xt
                else:
                    Xt = _Xt

                # This is so ugly :(
                if name.startswith('pre_') and not y_only:
                    y = transformer.transform(y[:, np.newaxis]).squeeze().copy()

        if self._final_estimator is None:
            return Xt, {}

        return Xt, fit_params_steps[self.steps[-1][0]], y

    def fit(self, X, y=None, start_idx=0, end_idx=-1,
            **fit_params):
        """
        Stolen from scikit-learn

        Fit the model
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, fit_params, y = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt[start_idx:, :], y[start_idx:],
                                      **fit_params)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Stolen from scikit-learn

        Fit the model and transform with the final estimator
        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
        last_step = self._final_estimator
        Xt, fit_params, y = self._fit(X, y, **fit_params)
        if hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, y, **fit_params)
        elif last_step is None:
            return Xt
        else:
            return last_step.fit(Xt, y, **fit_params).transform(Xt)

    def inverse_transform(self, X, y=None):
        Xt = X
        for name, step in self.steps[-2::-1]:
            if name.startswith('pre_') and not getattr(step, 'y_only', False):
                if Xt.ndim == 1:
                    Xt = Xt[:, np.newaxis]
                Xt = step.inverse_transform(Xt)

        return Xt

    def transform_y(self, y):
        yt = y
        for name, step in self.steps:
            if getattr(step, 'y_only', False):
                yt = expand_dim_if_needed(yt)
                yt = step.transform(yt, refit=True)
        return yt

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, to_scale=False, refit=True, start_idx=0):
        """
        NOTE: Most of this method stolen from scikit-learn

        Apply transforms to the data, and predict with the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_pred : array-like
        """
        Xt = X.copy()
        for name, transform in self.steps[:-1]:
            if transform is not None:
                y_only = getattr(transform, 'y_only', False)
                if y_only:
                    continue
                if _needs_refit(transform) and refit:
                    Xt = transform.transform(Xt, refit=True)
                else:
                    Xt = transform.transform(Xt)

        prediction = self.steps[-1][-1].predict(Xt)
        if to_scale:
            prediction = expand_dim_if_needed(prediction)
            for idx in range(prediction.shape[1]):
                prediction[:, [idx]] = self.inverse_transform(prediction[:, [idx]])
        return prediction[start_idx:].squeeze()

    def forecast(self, X, start_idx, forecast_window=None, trans_window=None):
        """
        Run out of sample predictions. That is, predict on X up until start_idx,
        use the predictions to concatenate data onto X, and continue predicting
        for the full length of X (or forecast_window if specified).

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        start_idx : int
            Index of X on which to start forecasting.
        forecast_window : int (optional)
            Number of steps to forecast into the future (defaults to forecast
            the remainder of X, but can be longer or shorter).
        trans_window : int (optional)
            Number of previous values of X necessary for transforming X into
            features. Set this to speed up forecasting such that you do not have
            to re-transform X for every prediction step of forecasting.

        Returns
        -------

        """
        # TODO:
        # Assert the model is fitted
        # Assert start_idx is > num_lags + pred_stride. This one can cause
        # everything to turn nan.

        if trans_window is not None:
            assert start_idx > trans_window, (
                'start_idx must be > than trans_window')
        if forecast_window is not None:
            assert forecast_window > 0, (
                    'forecast_window must be a positive')
            end_idx = start_idx + forecast_window
        else:
            end_idx=X.shape[0]
        # Run the prediction up until the starting index.
        # Note: We have to expand dims for multioutput results.
        X_init = expand_dim_if_needed(
            self.predict(X[:start_idx], to_scale=True, refit=True)
        )[:, 0]

        # Create the final prediction matrix and fill in all predictions up to
        # the starting index.
        X_total = np.empty((end_idx, 1), dtype=np.float32)
        X_total[:X_init.shape[0], 0] = X_init

        # For each out of sample point (aka >= start_idx)
        for idx in range(start_idx, end_idx):

            # Predict the next point
            start = 0 if trans_window==None else idx - trans_window
            next_point = expand_dim_if_needed(
                self.predict(X_total[start:idx], to_scale=True,
                             refit=False)
            )[-1, 0]

            # And add that point to the total prediction matrix.
            X_total[idx, 0] = next_point

        return X_total


class ClassifierPipeline(_BasePipeline):

    def __init__(self, steps, memory=None):
        super().__init__(steps, memory=memory)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, refit=True):
        """
        NOTE: Most of this method stolen from scikit-learn.
        Only difference is that I want to squeeze the output.

        Apply transforms to the data, and predict with the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_pred : array-like
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                if _needs_refit(transform) and refit:
                    Xt = transform.transform(Xt, refit=True)
                else:
                    Xt = transform.transform(Xt)

        prediction = self.steps[-1][-1].predict(Xt)
        return prediction.squeeze()
