# from functools import wraps

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory


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
                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)

                # This is so ugly :(
                if name.startswith('pre_'):
                    y = transformer.transform(y[:, np.newaxis]).squeeze().copy()

        if self._final_estimator is None:
            return Xt, {}

        return Xt, fit_params_steps[self.steps[-1][0]], y

    def fit(self, X, y=None, **fit_params):
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
            self._final_estimator.fit(Xt, y, **fit_params)

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

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, to_scale=False, refit=True):
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
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                if _needs_refit(transform) and refit:
                    Xt = transform.transform(Xt, refit=True)
                else:
                    Xt = transform.transform(Xt)

        prediction = self.steps[-1][-1].predict(Xt)
        if to_scale:
            for name, step in self.steps[-2::-1]:
                if name.startswith('pre_'):
                    if prediction.ndim == 1:
                        prediction = prediction[:, np.newaxis]
                    prediction = step.inverse_transform(prediction)

        return prediction.squeeze()

    def forecast(self, X, start_idx):
        # TODO:
        # Assert the model is fitted
        # Assert start_idx is > num_lags + pred_stride
        # Don't bother to re-transform the whole series every time.

        X_init = self.predict(X[:start_idx], to_scale=True,
                              refit=True)[:, np.newaxis]

        for idx in range(start_idx, len(X)):
            next_point = self.predict(X_init, to_scale=True,
                                      refit=False)[-1, np.newaxis]
            X_init = np.vstack((X_init, next_point))

        return X_init


class ClassifierPipeline(_BasePipeline):

    def __init__(self,steps, memory=None):
        super().__init__(steps, memory=memory)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X):
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
                if _needs_refit(transform):
                    Xt = transform.transform(Xt, refit=True)
                else:
                    Xt = transform.transform(Xt)

        prediction = self.steps[-1][-1].predict(Xt)
        return prediction.squeeze()
