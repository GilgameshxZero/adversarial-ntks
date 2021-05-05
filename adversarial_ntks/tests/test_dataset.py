import unittest

import jax  # This hides tensorflow logging

from adversarial_ntks.dataset import get_np_data


class TestDataset(unittest.TestCase):
    def test_mnist(self):
        ds = get_np_data(
            name="mnist",
            split="train[:128]",
            image_width=14,
            flatten=False,
        )
        self.assertEqual(ds.xs.shape, (128, 14, 14, 1))
        self.assertEqual(ds.xs.shape[0], ds.ys.shape[0])

    def test_agg_labels(self):
        ds = get_np_data(
            name="mnist",
            split="train[:128]",
            agg_labels=((1, ), (0, 7), (9, )),
        )
        self.assertEqual(ds.ys.min(), 0)
        self.assertEqual(ds.ys.max(), 2)

        with self.assertRaises(Exception):
            get_np_data(
                name="mnist",
                split="train[:128]",
                agg_labels=((1, ), (0, 7), (9, 2, 3, 1)),
            )