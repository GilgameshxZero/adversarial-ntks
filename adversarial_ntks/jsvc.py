"""sklearn.svm.SVC in Jax"""
from typing import Callable

import numpy as np
from sklearn import svm

import jax
import jax.numpy as jnp


def gram(
    func: Callable,
    X1: jnp.ndarray,
    X2: jnp.ndarray,
) -> jnp.ndarray:
    """Computes the gram matrix.
    Taken from https://github.com/IPL-UV/jaxkern.

    Given a function `Callable` and some `params`, we can
    use the `jax.vmap` function to calculate the gram matrix
    as the function applied to each of the points.
    Parameters
    ----------
    func : Callable
        a callable function (kernel or distance)
    X1 : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    Y2 : jax.numpy.ndarray
        other input dataset (n_samples, n_features)
    Returns
    -------
    mat : jax.numpy.ndarray
        the gram matrix.
    """
    return jax.vmap(lambda x1: jax.vmap(lambda x2: func(x1, x2))(X2))(X1)


def sv_gram(clf: svm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """Get gram matrix between support vectors and X.
    X: shape (n_samples, n_features)
    return: shape (n_SV, n_samples)
    """
    kernel_type = clf.get_params()["kernel"]
    SV = jnp.array(clf.support_vectors_)  # of shape(n_SV, n_features)

    if kernel_type == "linear":
        return SV @ X.T
    elif kernel_type == "rbf":
        dists = gram(func=lambda x, y: jnp.sum((x - y)**2), X1=SV, X2=X)
        return jnp.exp(-clf._gamma * dists)
    elif kernel_type == "poly":
        # Poly-4.
        raise NotImplementedError()
    else:
        # Laplacian
        # TODO: Make custom kernels more configurable
        dists = gram(func=lambda x, y: jnp.sum(jnp.abs(x - y)), X1=SV, X2=X)
        return jnp.exp(-0.01 * clf._gamma * dists)


def decision_function(clf: svm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """
    X: shape (n_samples, n_features)
    return: shape (n_samples, )
    """
    gmat = sv_gram(clf, X)  # shape: (n_SV, n_samples)
    dual_coefs = jnp.array(clf.dual_coef_)  # shape: (n_classes - 1, n_SV)
    assert dual_coefs.shape[0] == 1
    return (dual_coefs @ gmat)[0] + clf.intercept_


def _decision_function_sum(clf: svm.SVC, X: jnp.ndarray) -> float:
    return decision_function(clf, X).sum()


grad_decision_function = jax.jit(
    jax.grad(_decision_function_sum, 1),
    static_argnums=0,
)


def predict(clf: svm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (n_features, )
    return: shape (n_features, )
    """
    return (decision_function(clf, X) > 0).astype(np.int)
