"""
gmr
===

Gaussian Mixture Models (GMMs) for clustering and regression in Python.
"""

__version__ = "1.1"

__all__ = ['gmm', 'mvn', 'utils']

from .mvn import MVN, plot_error_ellipse, check_loglikelihood_grads
from .gmm import GMM, plot_error_ellipses, check_likelihood_grads
