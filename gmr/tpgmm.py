"""
A naive implementation of Task-parametrized GMM.
"""
import numpy as np
from gmm import GMM
from .utils import check_random_state

from sklearn.cluster import KMeans

class TPGMM(object):
    """Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of MVNs that compose the GMM.

    n_feature_lens: array
        feature length of in each factorized GMM

    priors : array of shape (n_components,), optional
        Weights of the components.

    means : list,  a list of arrays of shape (n_components, n_features), optional
        Means of the components.

    covariances : list, a list of arrays of shape (n_components, n_features, n_features), optional
        Covariances of the components.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    ...
    Other parameters are similar to the sklearn implementation
    """
    def __init__(self, n_components, n_feature_lens, priors=None, means=None, covariances=None,
                 verbose=0, random_state=None, covariance_type='full', tol=1e-5, min_covar=1e-5, n_iter=100):
        self.n_components = n_components
        self.n_feature_lens = n_feature_lens
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.n_iter = n_iter

        self.gmms = None
        self.output_mdl = None
        return

    def fit(self, X):
        if sum(self.n_feature_lens) != X.shape[1]:
            print 'Invalid feature lengths. Check the parameter of n_feature_lens'
        #separate X into the list of data for each feature factorization
        X_lst = [X[:, sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len)] for idx, n_feat_len in enumerate(self.n_feature_lens)]

        #maybe we need a k-means initialization here
        if self.priors is None or self.means is None or self.covariances is None:
            print 'Initializing with KMeans...'
            self.initialize_KMeans(X)
            # print 'Initializing with time separation...'
            # self.initialize_time_segmentation(X)

        #prepare GMM models
        #note the priors are tied
        self.gmms = [GMM(   n_components=self.n_components, priors=self.priors, means=self.means[i], covariances=self.covariances[i],
                                covariance_type=self.covariance_type) for i in range(len(self.n_feature_lens))]
        likelihoods_hist = []
        for i in range(self.n_iter):
            #E-step.
            responsibilities, likelihoods = self.to_responsibilities(X_lst)

            gamma = responsibilities / np.tile(np.sum(responsibilities, axis=0)+1e-10, (responsibilities.shape[0], 1))
            #M-step, update parameters
            #priors
            self.priors = np.mean(responsibilities, axis=0)
            for gmm in self.gmms:
                gmm.priors = self.priors

            #means & covariances
            for k in range(self.n_components):
                for X_samples, gmm in zip(X_lst, self.gmms):
                    gmm.means[k] = gamma[:, k].dot(X_samples)
                    X_samples_tmp = X_samples - gmm.means[k]
                    if self.covariance_type == 'diag':
                        gmm.covariances[k] = np.diag(np.sum(X_samples_tmp ** 2 * np.array([gamma[:, k]]).T, axis=0))
                    elif self.covariance_type == 'full':
                        gmm.covariances[k] = X_samples_tmp.T.dot(np.diag(gamma[:, k])).dot(X_samples_tmp) + np.eye(X_samples_tmp.shape[1]) * self.min_covar
                    else:
                        #unsupported covariance type, give up updating the covariances
                        pass
            likelihoods_hist.append(np.mean(np.log(np.sum(likelihoods+1e-14, axis=1))))
            if len(likelihoods_hist) > 1:
                if (likelihoods_hist[-1] - likelihoods_hist[-2]) < self.tol:
                    print 'EM steps converged within the tolerance at step {0}.'.format(i+1)
                    break
            if i % 10 == 0:
                print 'Data log-likelihood at the {0}-th step: {1}'.format(i+1, likelihoods_hist[-1])
        return likelihoods_hist

    def to_probability_density(self, X):
        if self.gmms is None:
            print 'Model has not been initialized yet.'

        factorization = False
        if isinstance( X, np.ndarray ):
            if len(X.shape) == 2:
                factorization = True
        if factorization:
            if sum(self.n_feature_lens) != X.shape[1]:
                print 'Invalid feature lengths. Check the parameter of n_feature_lens'
            #separate X into the list of data for each feature factorization
            X_lst = [X[:, sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len)] for idx, n_feat_len in enumerate(self.n_feature_lens)]
        else:
            X_lst = X

        likelihoods_factorized = [np.array(gmm.to_components_probability_density(X_samples)).T for X_samples, gmm in zip(X_lst, self.gmms)]
        # likelihoods = np.prod(likelihoods_factorized, axis=0)
        likelihoods = np.ones(likelihoods_factorized[0].shape)
        for l in likelihoods_factorized:
            likelihoods = likelihoods * l

        likelihoods = likelihoods * np.tile(self.priors, (X_samples.shape[0], 1))

        return likelihoods

    def to_responsibilities(self, X):
        likelihoods = self.to_probability_density(X)
        responsibilities = likelihoods / np.tile(np.sum(likelihoods, axis=1)+1e-10, (likelihoods.shape[1], 1)).T
        return responsibilities, likelihoods

    def to_transformed_features(self, transformations):
        '''
        transform each feature components to the original space and jointly decide the Gaussian parameters
        the linear transformation will translate and skew the gaussian parameters to make them comparable in the original space
        transformations is an array of linear transformation with the type of {'A':affine_mat, 'b':bias}
        it is of the same length as the self.n_feature_lens
        '''
        assert(len(transformations) == len(self.n_feature_lens))
        assert(self.gmms is not None)
        means = [None] * self.n_components
        covars = [None] * self.n_components

        for k in range(self.n_components):
            for trans, gmm in zip(transformations, self.gmms):
                transformed_mean = trans['A'].dot(gmm.means[k]) + trans['b']
                transformed_covar = trans['A'].dot(gmm.covariances[k]).dot(trans['A'])
                transformed_covar_inv = np.linalg.pinv(transformed_covar)

                if means[k] is None:
                    means[k] = transformed_covar_inv.dot(transformed_mean)
                else:
                    means[k] += transformed_covar_inv.dot(transformed_mean)
                if covars[k] is None:
                    covars[k] = transformed_covar_inv
                else:
                    covars[k] += transformed_covar_inv
            covars[k] = np.linalg.pinv(covars[k])
            means[k] = covars[k].dot(means[k])

        return np.array(means), np.array(covars)

    def predict(self, indices, X, transformations):
        '''
        X is the query in the interested feature space
        indices is an array of the dimensions we are conditioning on
        transformations contain linear transformations to map model parameters to this interested space
        '''
        means, covars = self.to_transformed_features(transformations)

        #do gmr based on these transformed parameters
        if self.output_mdl is None:
            self.output_mdl = GMM(   n_components=self.n_components, priors=self.priors, means=means, covariances=covars,
                                    covariance_type=self.covariance_type)
        else:
            self.output_mdl.means = means
            self.output_mdl.covariances = covars

        Y, Y_Sigma = self.output_mdl.predict_with_covariance(indices, X)
        return Y, Y_Sigma

    def initialize_KMeans(self, X):
        #data must be already merged
        kmeans = KMeans(n_clusters=self.n_components, n_init=5)
        kmeans.fit(X)
        self.priors = [None] * self.n_components
        self.means = [[None] * self.n_components for n_feat_len in self.n_feature_lens]
        self.covariances = [[None] * self.n_components for n_feat_len in self.n_feature_lens]
        for label in range(self.n_components):
            indices = np.where(kmeans.labels_ == label)
            self.priors[label] = float(len(indices))
            for idx, n_feat_len in enumerate(self.n_feature_lens):
                self.means[idx][label] = np.mean(X[indices][:, sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len)], axis=0)
                if self.covariance_type == 'full':
                    self.covariances[idx][label] = np.cov(X[indices][:, sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len)].T) + np.eye(n_feat_len) * self.min_covar
                elif self.covariance_type == 'diag':
                    self.covariances[idx][label] = np.diag(np.mean((X[indices][:, sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len)] - self.means[idx][label]) ** 2, axis=0))  + np.eye(n_feat_len) * self.min_covar
        self.priors = self.priors / np.sum(self.priors)

        return

    def initialize_time_segmentation(self, X):
        #data must be already merged and the first dimension should be time index
        time_segmentation = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), self.n_components+1)
        self.priors = [None] * self.n_components
        self.means = [[None] * self.n_components for n_feat_len in self.n_feature_lens]
        self.covariances = [[None] * self.n_components for n_feat_len in self.n_feature_lens]
        for label in range(self.n_components):
            indices = np.where((X[:, 0] >= time_segmentation[label]) & (X[:, 0] <= time_segmentation[label+1]))
            self.priors[label] = float(len(indices))

            tmp_mean = np.mean(np.squeeze(X[indices, :]), axis=0)
            tmp_cov = np.cov(np.squeeze(X[indices, :]).T)
            for idx, n_feat_len in enumerate(self.n_feature_lens):
                self.means[idx][label] = tmp_mean[sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len)]
                self.covariances[idx][label] = tmp_cov[ sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len),
                                                        sum(self.n_feature_lens[0:idx]):(sum(self.n_feature_lens[0:idx])+n_feat_len)]
        self.priors = self.priors / np.sum(self.priors)

        return
