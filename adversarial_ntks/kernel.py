"""kernel primitives in Jax"""
from typing import Callable

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


@jax.jit
def linear(X1: jnp.ndarray, X2: jnp.ndarray):
    """X1.shape = (n1, d); X2.shape = (n2, d); return.shape = (n1, n2)"""
    return X1 @ X2.T


@jax.jit
def poly(
    X1: jnp.ndarray,
    X2: jnp.ndarray,
    gamma: float,
    coef0: float,
    degree: int,
):
    """X1.shape = (n1, d); X2.shape = (n2, d); return.shape = (n1, n2)"""
    return (gamma * X1 @ X2.T + coef0)**degree


@jax.jit
def rbf(X1: jnp.ndarray, X2: jnp.ndarray, gamma: float):
    """X1.shape = (n1, d); X2.shape = (n2, d); return.shape = (n1, n2)"""
    dists = gram(func=lambda x, y: jnp.sum((x - y)**2), X1=X1, X2=X2)
    return jnp.exp(-gamma * dists)


@jax.jit
def laplace(X1: jnp.ndarray, X2: jnp.ndarray, gamma: float):
    """X1.shape = (n1, d); X2.shape = (n2, d); return.shape = (n1, n2)"""
    dists = gram(func=lambda x, y: jnp.sum(jnp.abs(x - y)), X1=X1, X2=X2)
    return jnp.exp(-gamma * dists)
