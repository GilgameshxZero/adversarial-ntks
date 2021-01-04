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
):
    """
    name: e.g. "mnist", "cifar10". See
          https://www.tensorflow.org/datasets/catalog/overview for more details.
    split: See https://www.tensorflow.org/datasets/splits
    """

    xs, ys = operator.itemgetter('image', 'label')(
        tfds.as_numpy(
            tfds.load(
                name=name,
                split=split,
                batch_size=-1,
                data_dir="./data",
            )
        )
    )

    xs = xs.astype(np.float64)
    xs /= 255.0  # normalize

    if binary_labels:
        n_classes = ys.max() + 1
        ys = (ys >= (n_classes // 2)).astype(np.int)

    if image_width is not None:
        xs = downsample_imgs(xs, image_width)

    if flatten:
        xs = xs.reshape((len(xs), -1))

    return xs, ys


def plot_sample_data(data, max_i, flat=True, channels=1):
    for i, x in enumerate(data[:max_i]):
        plt.subplot(1, max_i, i + 1)
        plt.imshow(np.clip(np.squeeze(x if not flat else
                                      jnp.reshape(x, (int(x.shape[0] ** 0.5), -1) if channels == 1 else
                                                  (int((x.shape[0] / channels) ** 0.5), -1, channels))), 0, 1))
        plt.axis("off")
    plt.show()