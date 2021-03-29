import numpy as np
import sklearn.preprocessing


def pgd(
    X,
    Y,
    grad_func,
    eps,
    eps_norm,
    num_steps,
    step_size,
    step_norm,
    pixel_clip=True,
):
    """
    Perform PGD on a dataset given a gradient function for the classifier.

    X: np.array of input images. Pixels in [0, 1].
    Y: np.array of binary labels.
    grad_func: Function which takes a set of inputs (X), and returns an np.array
    of the gradient of the classifier at the image.
    eps: Size of epsilon-ball around each datapoint.
    eps_norm: Norm to compute ball sizes, e.g. 2 or np.inf.
    steps: Steps of PGD.
    step_size:
    step_norm: Norm for `step_size`.
    """
    assert X.shape[0] == Y.shape[0]

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
            res = X + eps * sklearn.preprocessing.normalize(
                offset, norm="l" + str(eps_norm), axis=1)

        # Clip to valid pixel range
        if pixel_clip:
            res = np.clip(res, 0, 1)

    return res
