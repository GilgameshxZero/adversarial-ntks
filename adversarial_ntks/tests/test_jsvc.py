import unittest

import numpy as np
from sklearn import svm

import jax
import jax.numpy as jnp

from adversarial_ntks import jsvc
from adversarial_ntks.dataset import get_np_data

jax.config.update("jax_enable_x64", True)


class TestJSVC(unittest.TestCase):
    def setUp(self):
        self.train_xs, self.train_ys = get_np_data(
            name="mnist",
            split="train[:128]",
            binary_labels=True,
            image_width=14,
        )

        self.test_xs, _ = get_np_data(
            name="mnist",
            split="test[:32]",
            binary_labels=True,
            image_width=14,
        )
        self.jtest_xs = jnp.array(self.test_xs)

    def check_jsvc(self, clf: svm.SVC):
        self.assertTrue(
            np.allclose(
                clf.decision_function(self.test_xs),
                jsvc.decision_function(clf, self.jtest_xs),
            ))
        self.assertTrue(
            all(clf.predict(self.test_xs) == jsvc.predict(clf, self.jtest_xs)))

    def test_linear(self):
        self.check_jsvc(
            svm.SVC(C=3, kernel="linear").fit(
                X=self.train_xs,
                y=self.train_ys,
            ))

    def test_poly(self):
        self.check_jsvc(
            svm.SVC(C=3, kernel="poly", coef0=np.pi, degree=2).fit(
                X=self.train_xs,
                y=self.train_ys,
            ))

    def test_rbf(self):
        self.check_jsvc(
            svm.SVC(C=3, kernel="rbf").fit(
                X=self.train_xs,
                y=self.train_ys,
            ))
