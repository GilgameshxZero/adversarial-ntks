import unittest

import numpy as np
import jax  # This hides tensorflow logging

from adversarial_ntks.dataset import Dataset, get_np_data


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.ds = get_np_data(
            name="mnist",
            split="train[:128]",
            image_width=14,
            flatten=False,
        )

    def test_post_init(self):
        with self.assertRaises(Exception):
            # Not matching shapes
            Dataset(xs=np.arange(6),
                    ys=np.arange(7),
                    one_hot=False,
                    num_classes=10)

        with self.assertRaises(Exception):
            # Bad num_classes
            Dataset(xs=np.arange(7),
                    ys=np.arange(7),
                    one_hot=True,
                    num_classes=6)

        with self.assertRaises(Exception):
            # Bad one hot
            Dataset(xs=np.arange(7),
                    ys=np.arange(7),
                    one_hot=True,
                    num_classes=10)

    def test_mnist(self):
        ds = self.ds
        self.assertEqual(ds.xs.shape, (128, 14, 14, 1))
        self.assertEqual(ds.xs.shape[0], ds.ys.shape[0])
        self.assertEqual(ds.num_classes, 10)
        self.assertEqual(ds.one_hot, False)

    def test_agg_labels(self):
        ds = get_np_data(
            name="mnist",
            split="train[:128]",
            agg_labels=((1, ), (0, 7), (9, )),
        )
        self.assertEqual(ds.ys.min(), 0)
        self.assertEqual(ds.ys.max(), 2)
        self.assertEqual(ds.num_classes, 3)

        with self.assertRaises(Exception):
            get_np_data(
                name="mnist",
                split="train[:128]",
                agg_labels=((1, ), (0, 7), (9, 2, 3, 1)),
            )

    def test_one_hot(self):
        ds = self.ds
        self.assertEqual(ds.one_hot, False)
        self.assertEqual(ds.to_one_hot().one_hot, True)
        self.assertEqual(ds.to_one_hot().to_one_hot().one_hot, True)

    def test_prefix(self):
        ds = self.ds.prefix(17)
        self.assertEqual(ds.xs.shape[0], 17)

    def test_subsample(self):
        ds = self.ds.subsample(17)
        self.assertEqual(ds.xs.shape[0], 17)