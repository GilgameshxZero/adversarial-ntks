from __future__ import annotations

import dataclasses
import operator
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class Dataset:
    xs: np.ndarray
    ys: np.ndarray
    num_classes: int
    one_hot: bool
    name: str = "UnknownDS"

    def __post_init__(self):
        assert self.xs.shape[0] == self.ys.shape[0]
        if self.one_hot:
            assert self.ys.shape == (self.xs.shape[0], self.num_classes)
        else:
            assert self.ys.shape == (self.xs.shape[0], )
            assert self.ys.max() < self.num_classes

    def to_one_hot(self) -> Dataset:
        if self.one_hot: return self
        return Dataset(
            xs=self.xs,
            ys=tf.keras.utils.to_categorical(
                self.ys,
                num_classes=self.num_classes,
            ),
            num_classes=self.num_classes,
            one_hot=True,
            name=self.name,
        )

    def prefix(self, sz: int) -> Dataset:
        return Dataset(
            xs=self.xs[:sz],
            ys=self.ys[:sz],
            num_classes=self.num_classes,
            one_hot=self.one_hot,
            name=f"{self.name}[:{sz}]",
        )

    def subsample(self, sz: int) -> Dataset:
        inds = np.random.choice(self.xs.shape[0], size=sz, replace=False)
        return Dataset(
            xs=self.xs[inds],
            ys=self.ys[inds],
            num_classes=self.num_classes,
            one_hot=self.one_hot,
            name=f"{self.name}[~{sz}]",
        )

    def replace_xs(self, new_xs: np.ndarray) -> Dataset:
        return Dataset(
            xs=new_xs,
            ys=self.ys,
            num_classes=self.num_classes,
            one_hot=self.one_hot,
            name=f"self.name(replaced_xs)",
        )


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
        one_hot=False,
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
