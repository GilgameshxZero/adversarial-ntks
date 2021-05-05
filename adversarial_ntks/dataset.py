import operator
import os
from typing import NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    num_classes: int
    name: Optional[str] = None


def downsample_imgs(imgs, image_width):
    return tf.image.resize(
        tf.convert_to_tensor(imgs),
        size=[image_width, image_width],
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=True,
        antialias=False,
    ).numpy()


def get_np_data(
    name: str,  # "mnist" or "cifar10"
    split,
    agg_labels: Optional[Tuple[Tuple, ...]] = None,
    flatten: bool = True,
    image_width: Optional[int] = None,
    data_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")),
    dtype: np.dtype = np.float64,
) -> Dataset:
    """
    name: e.g. "mnist", "cifar10". See
          https://www.tensorflow.org/datasets/catalog/overview for more details.
    split: See https://www.tensorflow.org/datasets/splits
    """

    xs, ys = operator.itemgetter("image", "label")(tfds.as_numpy(
        tfds.load(
            name=name,
            split=split,
            batch_size=-1,
            data_dir=data_dir,
        )))

    xs = xs.astype(dtype)
    xs /= 255.0  # normalize

    if agg_labels is not None:
        for i, agg1 in enumerate(agg_labels):
            for agg2 in agg_labels[i + 1:]:
                assert set(agg1).intersection(set(agg2)) == set()

        for i, y in enumerate(ys):
            ys[i] = -1
            for j, agg in enumerate(agg_labels):
                if y in agg:
                    ys[i] = j
                    break

        xs = xs[ys >= 0]
        ys = ys[ys >= 0]

    if image_width is not None:
        xs = downsample_imgs(xs, image_width)

    if flatten:
        xs = xs.reshape((len(xs), -1))

    assert ys.min() == 0

    return Dataset(
        xs=xs,
        ys=ys,
        num_classes=ys.max() + 1,
        name=f"{name}-{split}",
    )


def plot_images(
    data: Dataset,
    flat: bool = True,
    channels: int = 1,
    num_to_plot: int = 10,
):
    # yapf: disable
    print(data.ys[:num_to_plot])
    for i, x in enumerate(data.xs[:num_to_plot]):
        plt.subplot(1, num_to_plot, i + 1)
        plt.imshow(np.clip(np.squeeze(x if not flat else
                                      np.reshape(x, (int(x.shape[0] ** 0.5), -1) if channels == 1 else
                                                 (int((x.shape[0] / channels) ** 0.5), -1, channels))), 0, 1))
        plt.axis("off")
    plt.show()
    # yapf: enable
