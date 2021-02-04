import operator
import os

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


def downsample_imgs(imgs, image_width):
    return tf.image.resize(
        tf.convert_to_tensor(imgs),
        size=[image_width, image_width],
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=True,
        antialias=False,
    ).numpy()


def get_np_data(
    name,  # "mnist" or "cifar10"
    split,
    binary_labels=False,
    flatten=True,
    image_width=None,
    data_dir=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data")),
    dtype=np.float64
):
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

    if binary_labels:
        n_classes = ys.max() + 1
        ys = (ys >= (n_classes // 2)).astype(np.int)

    if image_width is not None:
        xs = downsample_imgs(xs, image_width)

    if flatten:
        xs = xs.reshape((len(xs), -1))

    return xs, ys


def plot_sample_data(data, flat=True, channels=1):
    # yapf: disable
    for i, x in enumerate(data):
        plt.subplot(1, data.shape[0], i + 1)
        plt.imshow(np.clip(np.squeeze(x if not flat else
                                      np.reshape(x, (int(x.shape[0] ** 0.5), -1) if channels == 1 else
                                                 (int((x.shape[0] / channels) ** 0.5), -1, channels))), 0, 1))
        plt.axis("off")
    plt.show()
    # yapf: enable
