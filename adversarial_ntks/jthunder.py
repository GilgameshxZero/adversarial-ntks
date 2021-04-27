"""
thundersvm.SVC in Jax
ThunderSVM is very jank.
"""

import numpy as np
import thundersvm

import jax
import jax.numpy as jnp

from adversarial_ntks import kernel


def sv_gram(clf: thundersvm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """
    Get gram matrix between support vectors and X.
    X: shape (n_samples, n_features)
    return: shape (n_SV, n_samples)

    Based off of formulas at
    https://scikit-learn.org/stable/modules/svm.html#svm-kernels
    """
    SV = jnp.array(clf.support_vectors_)  # of shape(n_SV, n_features)

    # ThunderSVM sometimes gets shape wrong?!?!
    if SV.shape[1] == 1 + X.shape[1]:
        SV = SV[:, 1:]

    if clf.kernel == "linear":
        return kernel.linear(SV, X)
    elif clf.kernel == "polynomial":
        return kernel.poly(
            SV,
            X,
            gamma=clf._gamma,
            coef0=clf.coef0,
            degree=clf.degree,
        )
    elif clf.kernel == "rbf":
        return kernel.rbf(SV, X, gamma=clf._gamma)
    elif callable(clf.kernel):
        return clf.kernel(SV, X)
    else:
        raise NotImplementedError


def decision_function(clf: thundersvm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """
    X: shape (n_samples, n_features)
    return: shape (n_samples, )
    """
    gmat = sv_gram(clf, X)  # shape: (n_SV, n_samples)
    dual_coefs = jnp.array(clf.dual_coef_)  # shape: (n_classes - 1, n_SV)
    assert dual_coefs.shape[0] == 1

    # Need to do weird sign change so positive is class 1.
    return -(dual_coefs @ gmat)[0] + clf.intercept_


def _decision_function_sum(clf: thundersvm.SVC, X: jnp.ndarray) -> float:
    return decision_function(clf, X).sum()


grad_decision_function = jax.jit(
    jax.grad(_decision_function_sum, 1),
    static_argnums=0,
)


def predict(clf: thundersvm.SVC, X: jnp.ndarray) -> jnp.ndarray:
    """
    x: shape (n_features, )
    return: shape (n_features, )
    """
    return (decision_function(clf, X) > 0).astype(np.int)
