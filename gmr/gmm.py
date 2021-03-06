from __future__ import print_function
import numpy as np
from .utils import check_random_state
from .mvn import MVN

import sklearn.mixture as skmixture

class GMM(object):
    """Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of MVNs that compose the GMM.

    priors : array, shape (n_components,), optional
        Weights of the components.

    means : array, shape (n_components, n_features), optional
        Means of the components.

    covariances : array, shape (n_components, n_features, n_features), optional
        Covariances of the components.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    """
    """
    <hyin/Jun-15th-2016> Make this as a wrapper of the scikit learn GMM as it supports
    more features
    """
    def __init__(self, n_components, priors=None, means=None, covariances=None,
                 verbose=0, random_state=None, covariance_type='diag', thresh=None, tol=0.001, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='kmeans'):
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

        self.covariance_type = covariance_type
        self.thresh = thresh
        self.tol = tol
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

    def _check_initialized(self):
        if self.priors is None:
            raise ValueError("Priors have not been initialized")
        if self.means is None:
            raise ValueError("Means have not been initialized")
        if self.covariances is None:
            raise ValueError("Covariances have not been initialized")

    def from_samples(self, X, R_diff=1e-4, n_iter=100):
        """MLE of the mean and covariance.

        Expectation-maximization is used to infer the model parameters. The
        objective function is non-convex. Hence, multiple runs can have
        different results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true function.

        R_diff : float
            Minimum allowed difference of responsibilities between successive
            EM iterations.

        n_iter : int
            Maximum number of iterations.

        Returns
        -------
        self : MVN
            This object.
        """
        n_samples, n_features = X.shape

        if self.priors is None:
            self.priors = np.ones(self.n_components,
                                  dtype=np.float) / self.n_components

        if self.means is None:
            indices = self.random_state.choice(
                np.arange(n_samples), self.n_components)
            self.means = X[indices]

        if self.covariances is None:
            self.covariances = np.empty((self.n_components, n_features,
                                         n_features))
            for k in range(self.n_components):
                self.covariances[k] = np.eye(n_features)

        R = np.zeros((n_samples, self.n_components))
        for _ in range(n_iter):
            R_prev = R

            # Expectation
            R = self.to_responsibilities(X)

            if np.linalg.norm(R - R_prev) < R_diff:
                if self.verbose:
                    print("EM converged.")
                break

            # Maximization
            w = R.sum(axis=0)
            R_n = R / w
            self.priors = w / w.sum()
            self.means = R_n.T.dot(X)
            for k in range(self.n_components):
                Xm = X - self.means[k]
                self.covariances[k] = (R_n[:, k, np.newaxis] * Xm).T.dot(Xm)

        return self

    def fit(self, X):
        """
        create an scikit learn GMM to fit the data, retrieve the resultant model parameters
        use KMeans initialization and support different types of covariances
        return the BIC score for potential model selection
        """
        skGMM = skmixture.GaussianMixture(  n_components=self.n_components,
                                            covariance_type=self.covariance_type,
                                            random_state=self.random_state,
                                            # thresh=self.thresh,
                                            tol=self.tol,
                                            # min_covar=self.min_covar,
                                            max_iter=self.n_iter,
                                            n_init=self.n_init,
                                            # params=self.params,
                                            init_params=self.init_params,
                                            verbose=self.verbose)
        skGMM.fit(X)

        #retrieve estimated parameters
        self.priors = skGMM.weights_
        self.means = skGMM.means_
        if self.covariance_type == 'spherical' or self.covariance_type == 'diag':
            self.covariances = np.array([np.diag(covar) for covar in skGMM.covariances_])
        elif self.covariance_type == 'tied':
            self.covariances = np.array([skGMM.covariances_ for i in range(self.n_components)])
        elif self.covariance_type == 'full':
            self.covariances = skGMM.covariances_

        return skGMM.bic(X)

    def save_model(self, fname):
        if self.priors is None or self.means is None or self.covariances is None:
            print('Model parameters have not been initialized.')
        else:
            model_dict = {'priors':self.priors, 'means':self.means, 'covariances':self.covariances}
            np.save(fname, model_dict)
            print('Model saved to {0}'.format(fname))
        return

    def load_model(self, fname):
        model_dict = np.load(fname).item()

        if model_dict is None:
            print('Failed to load model {0}'.format(fname))
        else:
            self.n_components = len(model_dict['priors'])
            self.priors = model_dict['priors']
            self.means = model_dict['means']
            self.covariances = model_dict['covariances']

        return

    def sample(self, n_samples):
        """Sample from Gaussian mixture distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the GMM.
        """
        self._check_initialized()

        mvn_indices = self.random_state.choice(
            self.n_components, size=(n_samples,), p=self.priors)
        mvn_indices.sort()
        split_indices = np.hstack(
            ((0,), np.nonzero(np.diff(mvn_indices))[0] + 1, (n_samples,)))
        clusters = np.unique(mvn_indices)
        lens = np.diff(split_indices)
        samples = np.empty((n_samples, self.means.shape[1]))
        for i, (k, n_samples) in enumerate(zip(clusters, lens)):
            samples[split_indices[i]:split_indices[i + 1]] = MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).sample(n_samples=n_samples)
        return samples

    def to_responsibilities(self, X):
        """Compute responsibilities of each MVN for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        R : array, shape (n_samples, n_components)
        """
        self._check_initialized()

        n_samples = X.shape[0]
        R = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            R[:, k] = self.priors[k] * MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).to_probability_density(X)
        R /= R.sum(axis=1)[:, np.newaxis]
        return R

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
        # self._check_initialized()
        #
        # p = [MVN(mean=self.means[k], covariance=self.covariances[k],
        #          random_state=self.random_state).to_probability_density(X)
        #      for k in range(self.n_components)]
        p = self.to_components_probability_density(X)
        return np.dot(self.priors, p)

    def gradient(self, X):
        """
        Gradient for the likelihood at the given value
        """
        #responsibilities = self.to_responsibilities(X)
        grads = [np.empty(X.shape) for k in range(self.n_components)]
        for k in range(self.n_components):
            tmp_mvn = MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state)

            grads[k] = tmp_mvn.gradient(X) * (self.priors[k] * tmp_mvn.to_probability_density(X))

        grads = np.sum(grads, axis=0)
        return grads

    def to_components_probability_density(self, X):
        self._check_initialized()
        p = [MVN(mean=self.means[k], covariance=self.covariances[k],
                 random_state=self.random_state).to_probability_density(X)
             for k in range(self.n_components)]
        return p

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
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        self._check_initialized()

        n_features = self.means.shape[1] - len(indices)
        priors = np.empty(self.n_components)
        means = np.empty((self.n_components, n_features))
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            conditioned = mvn.condition(indices, x)
            priors[k] = (self.priors[k] *
                         mvn.marginalize(indices).to_probability_density(x))
            means[k] = conditioned.mean
            covariances[k] = conditioned.covariance
        priors /= priors.sum()
        return GMM(n_components=self.n_components, priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state)

    def predict(self, indices, X):
        """Predict means of posteriors.

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
        """
        self._check_initialized()

        n_samples, n_features_1 = X.shape
        n_features_2 = self.means.shape[1] - n_features_1
        Y = np.empty((n_samples, n_features_2))
        for n in range(n_samples):
            conditioned = self.condition(indices, X[n])
            Y[n] = conditioned.priors.dot(conditioned.means)
        return Y

    def predict_with_covariance(self, indices, X):
        """
        Predict as a single Gaussian approximation
        return both mean and covariance
        """
        self._check_initialized()

        n_samples, n_features_1 = X.shape
        n_features_2 = self.means.shape[1] - n_features_1
        Y = np.empty((n_samples, n_features_2))
        Y_Sigma = np.empty((n_samples, n_features_2, n_features_2))
        for n in range(n_samples):
            conditioned = self.condition(indices, X[n])
            Y[n] = conditioned.priors.dot(conditioned.means)
            Y_Sigma[n] = np.sum([conditioned.priors[i] * (conditioned.covariances[i] +
                np.array([conditioned.means[i]]).T.dot(np.array([conditioned.means[i]])))for i in range(len(conditioned.priors))], axis=0) - np.array([Y[n]]).T.dot(np.array([Y[n]]))
        return Y, Y_Sigma

    def to_ellipses(self, factor=1.0):
        """Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : array, shape (n_components, 3)
            Parameters that describe the error ellipses of all components:
            angles, widths and heights.
        """
        self._check_initialized()

        res = []
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse(factor)))
        return res


def plot_error_ellipses(ax, gmm, colors=None):
    """Plot error ellipses of GMM components.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    gmm : GMM
        Gaussian mixture model.
    """
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    for factor in np.linspace(0.5, 4.0, 8):
        for mean, (angle, width, height) in gmm.to_ellipses(factor):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(0.25)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)

import scipy as sp

def check_likelihood_grads(gmm, X):
    """
    Check gradient for loglikelihood of given multivariate Gaussian distribution
    """
    if len(X.shape) == 1:
        res = sp.optimize.check_grad(lambda x:gmm.to_probability_density(x), gmm.gradient, X)
    else:
        def tmp_func(x):
            return gmm.to_probability_density(np.array([x]))[0]
        def tmp_grad_func(x):
            return gmm.gradient(np.array([x]))[0]
        res = [sp.optimize.check_grad(tmp_func, tmp_grad_func, sample) for sample in X]
        res = np.mean(res)
    return res
