import numpy as np
from .utils import check_random_state, pinvh
import scipy as sp


def invert_indices(n_features, indices):
    inv = np.ones(n_features, dtype=np.bool)
    inv[indices] = False
    inv, = np.where(inv)
    return inv


class MVN(object):
    """Multivariate normal distribution.

    Some utility functions for MVNs. See
    http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    for more details.

    Parameters
    ----------
    mean : array, shape (n_features), optional
        Mean of the MVN.

    covariance : array, shape (n_features, n_features), optional
        Covariance of the MVN.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    """
    def __init__(self, mean=None, covariance=None, verbose=0,
                 random_state=None):
        self.mean = mean
        self.covariance = covariance
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.norm = None

        #cache for processing multiple samples
        self.i1_cache_ = None
        self.i2_cache_ = None
        self.prec_22_cache_ = None

    def _check_initialized(self):
        if self.mean is None:
            raise ValueError("Mean has not been initialized")
        if self.covariance is None:
            raise ValueError("Covariance has not been initialized")

    def from_samples(self, X, bessels_correction=True):
        """MLE of the mean and covariance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true function.

        Returns
        -------
        self : MVN
            This object.
        """
        self.mean = np.mean(X, axis=0)
        bias = 0 if bessels_correction else 1
        self.covariance = np.cov(X, rowvar=0, bias=bias)
        self.norm = None
        return self

    def sample(self, n_samples):
        """Sample from multivariate normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the MVN.
        """
        self._check_initialized()
        return self.random_state.multivariate_normal(
            self.mean, self.covariance, size=(n_samples,))

    def to_probability_density(self, X):
        """Compute probability density.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        p : array, shape (n_samples,)
            Probability densities of data.
        """
        self._check_initialized()

        X = np.atleast_2d(X)
        n_features = X.shape[1]

        C = self.covariance
        try:
            L = sp.linalg.cholesky(C, lower=True)
        except np.linalg.LinAlgError:
            C = self.covariance + 1e-3 * np.eye(n_features)
            L = sp.linalg.cholesky(C, lower=True)
        D = X - self.mean
        cov_sol = sp.linalg.solve_triangular(L, D.T, lower=True).T
        if self.norm is None:
            self.norm = 1. / (2*np.pi) ** (0.5 * n_features) / (sp.linalg.det(L) + 1e-10)

        DpD = np.sum(cov_sol ** 2, axis=1)
        return self.norm * np.exp(-0.5 * DpD)

    def to_log_probability_density(self, X):
        self._check_initialized()

        X = np.atleast_2d(X)
        n_features = X.shape[1]

        C = self.covariance
        try:
            L = sp.linalg.cholesky(C, lower=True)
        except np.linalg.LinAlgError:
            C = self.covariance + 1e-3 * np.eye(n_features)
            L = sp.linalg.cholesky(C, lower=True)
        D = X - self.mean
        cov_sol = sp.linalg.solve_triangular(L, D.T, lower=True).T
        if self.norm is None:
            self.norm = 1. / (2*np.pi) ** (0.5 * n_features) / (sp.linalg.det(L) + 1e-10)

        DpD = np.sum(cov_sol ** 2, axis=1)
        return -0.5*DpD - np.log(self.norm)

    def marginalize(self, indices):
        """Marginalize over everything except the given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to keep.

        Returns
        -------
        marginal : MVN
            Marginal MVN distribution.
        """
        self._check_initialized()
        return MVN(mean=self.mean[indices],
                   covariance=self.covariance[np.ix_(indices, indices)])

    def condition(self, indices, x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : MVN
            Conditional MVN distribution p(Y | X=x).
        """
        self._check_initialized()
        mean, covariance = self._condition(
            invert_indices(self.mean.shape[0], indices), indices, x)
        return MVN(mean=mean, covariance=covariance,
                   random_state=self.random_state)

    def predict(self, indices, X):
        """Predict means and covariance of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.

        covariance : array, shape (n_features_2, n_features_2)
            Covariance of the predicted features.
        """
        self._check_initialized()
        return self._condition(invert_indices(self.mean.shape[0], indices),
                               indices, X)
    def gradient(self, X):
        """Gradient for the loglikelihood at the given value"""
        self._check_initialized()
        C = self.covariance
        try:
            L = sp.linalg.cholesky(C, lower=True)
        except np.linalg.LinAlgError:
            C = self.covariance + 1e-3 * np.eye(n_features)
            L = sp.linalg.cholesky(C, lower=True)
        D = X - self.mean
        cov_sol = sp.linalg.solve_triangular(L, D.T, lower=True).T
        grads = -sp.linalg.solve_triangular(L.T, cov_sol.T, lower=False).T
        return grads

    def _condition(self, i1, i2, X):
        cov_12 = self.covariance[np.ix_(i1, i2)]
        cov_11 = self.covariance[np.ix_(i1, i1)]
        cov_22 = self.covariance[np.ix_(i2, i2)]
        cache_hit = False
        if self.i1_cache_ is not None and self.i2_cache_ is not None:
            #check if we are retrieving the same covariance submatrices as the cache
            if np.array_equal(self.i1_cache_, i1) and np.array_equal(self.i2_cache_, i2) and self.prec_22_cache_ is not None:
                prec_22 = self.prec_22_cache_
                cache_hit = True
        if not cache_hit:
        #calculate for now and update the cache...
            prec_22 = pinvh(cov_22)
            self.prec_22_cache_ = prec_22
            self.i1_cache_ = i1
            self.i2_cache_ = i2

        regression_coeffs = cov_12.dot(prec_22)

        mean = self.mean[i1] + regression_coeffs.dot((X - self.mean[i2]).T).T
        covariance = cov_11 - regression_coeffs.dot(cov_12.T)
        return mean, covariance

    def to_ellipse(self, factor=1.0):
        """Compute error ellipse.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        angle : float
            Rotation angle of the ellipse.

        width : float
            Width of the ellipse.

        height : float
            Height of the ellipse.
        """
        self._check_initialized()
        vals, vecs = sp.linalg.eigh(self.covariance)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.arctan2(*vecs[:, 0][::-1])
        width, height = factor * np.sqrt(vals)
        return angle, width, height


def plot_error_ellipse(ax, mvn):
    """Plot error ellipse of MVN.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mvn : MVN
        Multivariate normal distribution.
    """
    from matplotlib.patches import Ellipse
    for factor in np.linspace(0.5, 4.0, 8):
        angle, width, height = mvn.to_ellipse(factor)
        ell = Ellipse(xy=mvn.mean, width=width, height=height,
                      angle=np.degrees(angle))
        ell.set_alpha(0.25)
        ax.add_artist(ell)

def check_loglikelihood_grads(mvn, X):
    """
    Check gradient for loglikelihood of given multivariate Gaussian distribution
    """
    if len(X.shape) == 1:
        res = sp.optimize.check_grad(lambda x:mvn.to_log_probability_density(x), mvn.gradient, X)
    else:
        res = [sp.optimize.check_grad(lambda x:mvn.to_log_probability_density(x), mvn.gradient, sample) for sample in X]
        res = np.mean(res)
    return res
