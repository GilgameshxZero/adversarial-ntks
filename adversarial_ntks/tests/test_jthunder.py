import unittest

import numpy as np
import thundersvm

import jax
import jax.numpy as jnp

from adversarial_ntks import jthunder
from adversarial_ntks.dataset import get_np_data

jax.config.update("jax_enable_x64", True)


class TestJThunder(unittest.TestCase):
    def setUp(self):
        self.ds_train = get_np_data(
            name="mnist",
            split="train[:128]",
            binary_labels=True,
            image_width=14,
        )

        self.ds_test = get_np_data(
            name="mnist",
            split="test[:32]",
            binary_labels=True,
            image_width=14,
        )
        self.jtest_xs = jnp.array(self.ds_test.xs)

    def check_jthunder(self, clf: thundersvm.SVC):
        self.assertTrue(
            np.allclose(
                clf.decision_function(self.ds_test.xs).flatten(),
                -jthunder.decision_function(clf, self.jtest_xs),
                rtol=1e-4,
                atol=1e-4,
            ))

        self.assertTrue(
            all(
                clf.predict(self.ds_test.xs).astype(np.int) ==
                jthunder.predict(clf, self.jtest_xs)))

        self.assertTrue(
            np.allclose(jthunder.norm2(clf), jthunder.norm2_naive(clf)))

    def test_linear(self):
        self.check_jthunder(
            thundersvm.SVC(C=3, kernel="linear").fit(
                X=self.ds_train.xs,
                y=self.ds_train.ys,
            ))

    def test_poly(self):
        self.check_jthunder(
            thundersvm.SVC(C=3, kernel="polynomial", coef0=np.pi,
                           degree=2).fit(
                               X=self.ds_train.xs,
                               y=self.ds_train.ys,
                           ))

    def test_rbf(self):
        self.check_jthunder(
            thundersvm.SVC(C=3, kernel="rbf").fit(
                X=self.ds_train.xs,
                y=self.ds_train.ys,
            ))