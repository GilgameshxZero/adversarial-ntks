"""
thundersvm.SVC in Jax
ThunderSVM is very jank.
"""
from typing import NamedTuple

import numpy as np
import thundersvm

import jax
import jax.numpy as jnp

from adversarial_ntks import kernel


class _JThunderParams(NamedTuple):
    """Everything here can be passed into jax as non static argument"""
    gamma: float
    coef0: float
    degree: int
    intercept: float

    SV: jnp.ndarray
    dual_coefs: jnp.ndarray


def _get_params(clf: thundersvm.SVC) -> _JThunderParams:
    assert clf.n_classes == 2

    jtp = _JThunderParams(
        gamma=clf._gamma,
        coef0=clf.coef0,
        degree=clf.degree,
        intercept=clf.intercept_[0],
        SV=jnp.array(clf.support_vectors_[:, 1:]),
        dual_coefs=jnp.array(clf.dual_coef_[0]),
    )

    assert jtp.SV.shape == (clf.n_support_.sum(), clf.n_features)
    assert jtp.dual_coefs.shape == (clf.n_support_.sum(), )

    return jtp


def _sv_gram(
    kernel_type: str,
    jtp: _JThunderParams,
    X: jnp.ndarray,
) -> jnp.ndarray:
    """
    Get gram matrix between support vectors and X.
    X: shape (n_samples, n_features)
    return: shape (n_SV, n_samples)

    Based off of formulas at
    https://scikit-learn.org/stable/modules/svm.html#svm-kernel_types
    """
    if kernel_type == "linear":
        return kernel.linear(jtp.SV, X)
    elif kernel_type == "polynomial":
        return kernel.poly(
            jtp.SV,
            X,
            gamma=jtp.gamma,
            coef0=jtp.coef0,
            degree=jtp.degree,
        )
    elif kernel_type == "rbf":
        return kernel.rbf(jtp.SV, X, gamma=jtp.gamma)
    else:
        raise NotImplementedError


def _norm2_naive(kernel_type: str, jtp: _JThunderParams) -> jnp.float64:
    """Uses more memory than necessary"""
    if kernel_type == "linear":
        w = jtp.dual_coefs @ jtp.SV
        return w @ w

    gmat = _sv_gram(kernel_type, jtp, jtp.SV)  # shape: (n_SV, n_SV)
    return (jtp.dual_coefs @ gmat @ jtp.dual_coefs)


def norm2_naive(clf: thundersvm.SVC) -> jnp.float64:
    kernel_type = clf.kernel
    jtp = _get_params(clf)
    return _norm2_naive(kernel_type, jtp)


def _norm2(kernel_type: str, jtp: _JThunderParams) -> jnp.float64:
    """Memory light version"""
    def f(x):
        kXx = _sv_gram(kernel_type, jtp, x.reshape(1, -1))  # shape: (n, 1)
        return (jtp.dual_coefs @ kXx)[0]

    return jtp.dual_coefs @ jax.lax.map(jax.jit(f), jtp.SV)


def norm2(clf: thundersvm.SVC) -> jnp.float64:
    kernel_type = clf.kernel
    jtp = _get_params(clf)
    return _norm2(kernel_type, jtp)


@jax.partial(jax.jit, static_argnums=0)
def _decision_function(
    kernel_type: str,
    jtp: _JThunderParams,
    X: jnp.ndarray,
) -> jnp.ndarray:
    """
    X: shape (n_samples, n_features)
    return: shape (n_samples, )
    """
    gmat = _sv_gram(kernel_type, jtp, X)  # shape: (n_SV, n_samples)

    # Need to do weird sign change so positive is class 1.
    return -jtp.dual_coefs @ gmat + jtp.intercept


def _decision_function_sum(
    kernel_type: str,
    jtp: _JThunderParams,
    X: jnp.ndarray,
) -> float:
    return _decision_function(kernel_type, jtp, X).sum()


_grad_decision_function = jax.jit(
    jax.grad(_decision_function_sum, 2),
    static_argnums=0,
)


def get_grad_decision_function(clf: thundersvm.SVC):
    """
    Written like this to prevent leaking memory.
    We want static_argnums to be as light as possible.
    https://github.com/google/jax/issues/282
    """
    kernel_type = clf.kernel
    jtp = _get_params(clf)

    def f(X):
        return _grad_decision_function(kernel_type, jtp, X)

    return f


def decision_function(clf: thundersvm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (n_features, )
    return: shape (n_features, )
    """
    kernel_type = clf.kernel
    jtp = _get_params(clf)
    return _decision_function(kernel_type, jtp, X)


def get_decision_function(clf: thundersvm.SVC):
    """
    Written like this to prevent leaking memory.
    We want static_argnums to be as light as possible.
    https://github.com/google/jax/issues/282
    """
    kernel_type = clf.kernel
    jtp = _get_params(clf)

    def f(X):
        return _decision_function(kernel_type, jtp, X)

    return f


def predict(clf: thundersvm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (n_features, )
    return: shape (n_features, )
    """
    return (decision_function(clf, X) > 0).astype(np.int)
