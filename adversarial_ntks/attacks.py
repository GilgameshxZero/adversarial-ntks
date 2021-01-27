import numpy as np
import sklearn


def pgd(X, Y, grad_func, eps, eps_norm, steps, step_size, step_norm):
	"""
	Perform PGD on a dataset given a gradient function for the classifier.

	X: np.array of input images.
	Y: np.array of binary labels.
	grad_func: Function which takes a set of inputs (X), and returns an np.array
    of the gradient of the classifier at the image.
	eps: Size of epsilon-ball around each datapoint.
	eps_norm: Norm to compute ball sizes, e.g. 2 or np.inf.
	steps: Steps of PGD.
	step_size:
	step_norm: Norm for `step_size`.
	"""
	assert(X.shape[0] == Y.shape[0])

  res = np.copy(X)
  for step in range(steps):
    res_grad = grad_func(res)
    res_grad_Y = res_grad * (Y * -2 + 1).reshape((-1, 1))

    # Step.
    if step_norm == np.inf:
      res_delta = np.sign(res_grad_Y) * step_size
    else:
      res_delta = sklearn.preprocessing.normalize(
        res_grad_Y, norm="l" + str(step_norm), axis=1) * step_size
    res += res_delta

    # Project.
    if eps_norm == np.inf:
      res = np.clip(res, X - eps, X + eps)
    else:
      mask = res - X
      res = X + eps * sklearn.preprocessing.normalize(
        mask, norm="l" + str(eps_norm), axis=1)
    res = np.clip(res, 0, 1)
  
  return res
