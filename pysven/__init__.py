import numbers

import numpy as np
from scipy import sparse
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel, _deprecate_normalize, _pre_fit
from sklearn.svm import LinearSVC
from sklearn.utils import check_scalar, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted


class SVEN(MultiOutputMixin, RegressorMixin, LinearModel):
    """

    Parameters
    ----------
    t : float
        The upper bound of the SVElasticNet.
    lambda_ : float
        The regularization parameter.
    max_iter : int, optional (default=100)
        The maximum number of iterations.
    tol : float, optional (default=0.0001)
        The tolerance for the optimization.
    copy_X : bool, optional (default=True)
        If True, X will be copied; else, it may be overwritten.
    fit_intercept : bool, optional (default=True)
        If True, the intercept is estimated; else, it is not.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        The coefficients of the model.

    References
    ----------
    A Reduction of the Elastic Net to Support Vector
    Machines with an Application to GPU Computing

    https://arxiv.org/pdf/1409.1976.pdf

    https://github.com/andyc1997/support-vector-elastic-net

    Examples
    --------
    >>> from pysven import SVEN
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> regr = SVEN(t=1,lambda_=1,random_state=0)
    >>> regr.fit(X, y)
    SVEN(lambda_=1, random_state=0, t=1)
    >>> print(regr.coef_)
    [0. 1.]
    >>> print(regr.predict([[0, 0]]))
    [4.08007826]
    """

    def __init__(self, t=1, lambda_=1, max_iter=100, tol=0.0001, copy_X=True, fit_intercept=True, random_state=None, normalize=False):
        # hyperparameters
        self.t = t
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.normalize = normalize

    def fit(self, X, y, sample_weight=None):
        """Fit model with SVEN

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data.
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary.
        sample_weight : float or array-like of shape (n_samples,), default=None
            Sample weights. Internally, the `sample_weight` vector will be
            rescaled to sum to `n_samples`.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        _normalize = _deprecate_normalize(
            self.normalize, default=False, estimator_name=self.__class__.__name__
        )
        check_scalar(
            self.t,
            "t",
            target_type=numbers.Real,
            min_val=0.0,
        )

        check_scalar(
            self.lambda_,
            "lambda_",
            target_type=numbers.Real,
            min_val=0.0,
        )

        if self.max_iter is not None:
            check_scalar(
                self.max_iter, "max_iter", target_type=numbers.Integral, min_val=1
            )

        check_scalar(self.tol, "tol", target_type=numbers.Real, min_val=0.0)

        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        X_copied = self.copy_X and self.fit_intercept
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csc",
            order="F",
            dtype=[np.float64, np.float32],
            copy=X_copied,
            multi_output=True,
            y_numeric=True,
        )
        y = check_array(
            y, order="F", copy=False, dtype=X.dtype.type, ensure_2d=False
        )

        should_copy = self.copy_X and not X_copied
        X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
            X,
            y,
            None,
            False,
            _normalize,
            self.fit_intercept,
            copy=should_copy,
            check_input=True,
            sample_weight=sample_weight,
        )


        n_samples, n_features = X.shape

        # artificial dataset
        constructX = (
            np.concatenate((X - np.expand_dims(y,1) / self.t, X + np.expand_dims(y,1) / self.t), axis=1)
        ).T
        constructy = np.concatenate((np.ones((n_features)), -np.ones((n_features))), axis=0)

        # initialize variables
        C = 1 / (2 * self.lambda_)

        # train SVM
        if n_samples > 2 * n_features:
            self.SVC = LinearSVC(C=C, dual=False, max_iter=self.max_iter, random_state=self.random_state, tol=self.tol).fit(constructX, constructy)
        else:
            self.SVC = LinearSVC(C=C, dual=True, max_iter=self.max_iter, random_state=self.random_state, tol=self.tol).fit(constructX, constructy)

        self.alpha_ = C * np.maximum(
            np.zeros(constructy.shape),
            np.ones(constructy.shape) - constructy * (constructX @ self.SVC.coef_[0].T),
        )

        self.coef_ = (
                self.t
                * (self.alpha_[0:n_features] - self.alpha_[n_features: 2 * n_features])
                / np.sum(self.alpha_)
        )

        self._set_intercept(X_offset, y_offset, X_scale)
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        # check for finiteness of coefficients
        if not all(np.isfinite(w).all() for w in [self.coef_, self.intercept_]):
            raise ValueError(
                "SVEN resulted in non-finite parameter"
                " values. The input data may contain large values and need to"
                " be preprocessed."
            )

        return self

    @property
    def sparse_coef_(self):
        """Sparse representation of the fitted `coef_`."""
        return sparse.csr_matrix(self.coef_)

    def _decision_function(self, X):
        """Decision function of the linear model.

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)
        Returns
        -------
        T : ndarray of shape (n_samples,)
            The predicted decision function.
        """
        check_is_fitted(self)
        if sparse.isspmatrix(X):
            return safe_sparse_dot(X, self.coef_.T, dense_output=True) +self.intercept_
        else:
            return super()._decision_function(X)