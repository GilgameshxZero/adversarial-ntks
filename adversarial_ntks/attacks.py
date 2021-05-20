from typing import Callable, Optional, Union

import numpy as np
import sklearn.preprocessing
from tqdm.autonotebook import tqdm


def _pgd(
    X: np.ndarray,
    Y: np.ndarray,
    grad_func: Callable[[np.ndarray], np.ndarray],
    eps: float,
    eps_norm: Union[int, np.float],
    num_steps: int,
    step_size: float,
    step_norm: Union[int, np.float],
    pixel_clip: bool,
) -> np.ndarray:
    """We operate with np.ndarrays because jax arrays are immutable."""
    res = np.copy(X)
    for step in range(num_steps):
        res_grad = grad_func(res)
        res_grad_Y = res_grad * (Y * -2 + 1).reshape((-1, 1))

        # Compute step, scaled to have size step_size under norm step_norm
        if step_norm == np.inf:
            step = np.sign(res_grad_Y) * step_size
        else:
            step = step_size * sklearn.preprocessing.normalize(
                res_grad_Y, norm="l" + str(step_norm), axis=1)

        # Move a step
        res += step

        # Project to epsilon-ball
        if eps_norm == np.inf:
            res = np.clip(res, X - eps, X + eps)
        else:
            offset = res - X
            offset_norms = np.linalg.norm(offset, axis=-1, ord=2)
            projection = X + eps * offset / offset_norms[:, np.newaxis]
            res[offset_norms > eps] = projection[offset_norms > eps]

        # Clip to valid pixel range
        if pixel_clip:
            res = np.clip(res, 0, 1)

    return res


def pgd(
    X: np.ndarray,
    Y: np.ndarray,
    grad_func: Callable[[np.ndarray], np.ndarray],
    eps: float,
    eps_norm: Union[int, float],
    num_steps: int,
    step_size: float,
    step_norm: Union[int, float],
    pixel_clip: bool,
    batch_size: Optional[int] = None,
    tqdm_leave: bool = True,
) -> np.ndarray:
    """
    Perform PGD on a dataset given a gradient function for the classifier.

    X: input images. Pixels in [0, 1]. Flattened.
    Y: binary labels.
    grad_func: Takes images (X) and outputs gradient of the classifier at the images.
    eps: Size of epsilon-ball around each datapoint.
    eps_norm: Norm to compute ball sizes, e.g. 2 or np.inf.
    steps: Steps of PGD.
    step_size: Length to step per iteration
    step_norm: Norm for `step_size`.
    """
    assert X.shape[0] == Y.shape[0]
    num_samples = X.shape[0]

    if batch_size is None:
        batch_size = num_samples
    assert batch_size <= num_samples

    ret = np.zeros_like(X)
    for i in tqdm(range((num_samples - 1) // batch_size + 1), leave=tqdm_leave):
        batch_lo = i * batch_size
        batch_hi = min(batch_lo + batch_size, num_samples)

        ret[batch_lo:batch_hi] = _pgd(
            X=X[batch_lo:batch_hi],
            Y=Y[batch_lo:batch_hi],
            grad_func=grad_func,
            eps=eps,
            eps_norm=eps_norm,
            num_steps=num_steps,
            step_size=step_size,
            step_norm=step_norm,
            pixel_clip=pixel_clip,
        )

    return ret
