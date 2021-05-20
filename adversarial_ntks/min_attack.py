"""
Computes approximate minimal adversarial examples
via linear interpolation and binary search.
"""

from typing import Callable, Optional, Union

import numpy as np
import sklearn.preprocessing
from tqdm.autonotebook import tqdm


def _pgd_early_stop(
    X: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    pred_fn: Callable[[np.ndarray], np.ndarray],  # Outputs pre-sgn activations
    eps: float,
    eps_norm: Union[int, np.float],
    num_steps: int,
    step_size: float,
    step_norm: Union[int, np.float],
) -> np.ndarray:
    """We operate with np.ndarrays because jax arrays are immutable."""
    base_preds = pred_fn(X) > 0

    Y = base_preds.astype(np.float64)
    retX = np.copy(X)
    for step in range(num_steps):
        grad = grad_fn(retX)
        grad_Y = grad * (Y * -2 + 1).reshape((-1, 1))

        # Compute step, scaled to have size step_size under norm step_norm
        if step_norm == np.inf:
            step = np.sign(grad_Y) * step_size
        else:
            step = step_size * sklearn.preprocessing.normalize(
                grad_Y, norm="l" + str(step_norm), axis=1)

        # Move a step
        tmpX = retX + step

        # Project to epsilon-ball
        if eps_norm == np.inf:
            tmpX = np.clip(tmpX, X - eps, X + eps)
        else:
            offset = tmpX - X
            offset_norms = np.linalg.norm(offset, axis=-1, ord=2)
            projection = X + eps * offset / offset_norms[:, np.newaxis]
            tmpX[offset_norms > eps] = projection[offset_norms > eps]
            
        cur_preds = pred_fn(retX) > 0
        retX[cur_preds == base_preds] = tmpX[cur_preds == base_preds]

    return retX


def pgd_early_stop(
    X: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    pred_fn: Callable[[np.ndarray], np.ndarray],  # Outputs pre-sgn activations
    eps: float,
    eps_norm: Union[int, float],
    num_steps: int,
    step_size: float,
    step_norm: Union[int, float],
    batch_size: Optional[int] = None,
    tqdm_leave: bool = True,
) -> np.ndarray:
    """
    Perform PGD on a dataset given a gradient function for the classifier.

    X: input images. Pixels in [0, 1]. Flattened.
    Y: binary labels.
    grad_fn: Takes images (X) and outputs gradient of the classifier at the images.
    eps: Size of epsilon-ball around each datapoint.
    eps_norm: Norm to compute ball sizes, e.g. 2 or np.inf.
    steps: Steps of PGD.
    step_size: Length to step per iteration
    step_norm: Norm for `step_size`.
    """
    num_samples = X.shape[0]

    if batch_size is None:
        batch_size = num_samples
    assert batch_size <= num_samples

    ret = np.zeros_like(X)
    for i in tqdm(range((num_samples - 1) // batch_size + 1),
                  leave=tqdm_leave):
        batch_lo = i * batch_size
        batch_hi = min(batch_lo + batch_size, num_samples)

        ret[batch_lo:batch_hi] = _pgd_early_stop(
            X=X[batch_lo:batch_hi],
            grad_fn=grad_fn,
            pred_fn=pred_fn,
            eps=eps,
            eps_norm=eps_norm,
            num_steps=num_steps,
            step_size=step_size,
            step_norm=step_norm,
        )

    return ret


def _min_interpolate(
    baseX: np.ndarray,  # (batch_dim, d)
    advX: np.ndarray,  # (batch_dim, d)
    pred_fn: Callable[[np.ndarray], np.ndarray],  # Outputs pre-sgn activations
    num_bin_search_iters: int,
) -> np.ndarray:
    base_preds = pred_fn(baseX) > 0
    adv_preds = pred_fn(advX) > 0
    assert not np.any(base_preds == adv_preds)

    lo = np.zeros(advX.shape[0])
    hi = np.ones(advX.shape[0])
    for _ in range(num_bin_search_iters):
        mid = (lo + hi) / 2.0

        curX = (1 - mid[:, np.newaxis]) * baseX + mid[:, np.newaxis] * advX
        cur_preds = pred_fn(curX) > 0

        lo[cur_preds != adv_preds] = mid[cur_preds != adv_preds]
        hi[cur_preds == adv_preds] = mid[cur_preds == adv_preds]

    return (1 - hi[:, np.newaxis]) * baseX + hi[:, np.newaxis] * advX


def min_interpolate(
    baseX: np.ndarray,  # (batch_dim, d)
    advX: np.ndarray,  # (batch_dim, d)
    pred_fn: Callable[[np.ndarray], np.ndarray],  # Outputs pre-sgn activations
    num_bin_search_iters: int,
    batch_size: Optional[int] = None,
    tqdm_leave: bool = True,
) -> np.ndarray:
    """
    Shrinks advX by as much as possible
    while still giving different predictions that baseX.
    """
    assert baseX.shape == advX.shape
    num_samples = baseX.shape[0]

    if batch_size is None:
        batch_size = num_samples
    assert batch_size <= num_samples

    ret = np.zeros_like(baseX)
    for i in tqdm(range((num_samples - 1) // batch_size + 1),
                  leave=tqdm_leave):
        batch_lo = i * batch_size
        batch_hi = min(batch_lo + batch_size, num_samples)

        ret[batch_lo:batch_hi] = _min_interpolate(
            baseX=baseX[batch_lo:batch_hi],
            advX=advX[batch_lo:batch_hi],
            pred_fn=pred_fn,
            num_bin_search_iters=num_bin_search_iters,
        )

    return ret


def compute_min_perturbation(
    X: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    pred_fn: Callable[[np.ndarray], np.ndarray],  # Outputs pre-sgn activations
    eps: float,
    eps_norm: Union[int, float],
    num_steps: int,
    step_size: float,
    step_norm: Union[int, float],
    num_bin_search_iters: int = 32,
    batch_size: Optional[int] = None,
    tqdm_leave: bool = True,
) -> np.ndarray:
    advX = pgd_early_stop(
        X=X,
        grad_fn=grad_fn,
        pred_fn=pred_fn,
        eps=eps,
        eps_norm=eps_norm,
        num_steps=num_steps,
        step_size=step_size,
        step_norm=step_norm,
        batch_size=batch_size,
        tqdm_leave=tqdm_leave,
    )

    retX = min_interpolate(
        baseX=X,
        advX=advX.copy(),
        pred_fn=pred_fn,
        num_bin_search_iters=num_bin_search_iters,
        batch_size=batch_size,
        tqdm_leave=tqdm_leave,
    )

    return retX, advX


def batch_predict(
    X: np.ndarray,
    pred_fn: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> np.ndarray:
    num_samples = X.shape[0]

    if batch_size is None:
        batch_size = num_samples
    assert batch_size <= num_samples

    ret = np.zeros(num_samples, dtype=X.dtype)
    for i in range((num_samples - 1) // batch_size + 1):
        batch_lo = i * batch_size
        batch_hi = min(batch_lo + batch_size, num_samples)

        ret[batch_lo:batch_hi] = pred_fn(X[batch_lo:batch_hi])

    return ret
