import unittest
import numpy as np
from pyexpt.datasets import Blobs3
from sklearn.base import clone

class Blobs3TestCases(unittest.TestCase):
    def test_n_samples(self):
        dataset = Blobs3(n_samples=10)
        X = dataset.X
        self.assertEqual(len(X), 10)

    def test_n_features(self):
        dataset = Blobs3(n_features=3)
        X = dataset.X
        n, d = X.shape
        self.assertEqual(d, 3)

    def test_clone_WO_random_state(self):
        dataset = Blobs3()
        data1 = dataset.X
        dataset = clone(dataset)
        data2 = dataset.X
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, data1, data2)

    def test_clone_W_random_state(self):
        dataset = Blobs3(random_state=0)
        data1 = dataset.X
        dataset = clone(dataset)
        data2 = dataset.X
        np.testing.assert_array_equal(data1, data2)

    def test_set_params(self):
        dataset = Blobs3(random_state=0)
        dataset = clone(dataset).set_params(n_samples=10, n_features=3)
        self.assertTupleEqual((dataset.n_samples, dataset.n_features), (10, 3))

    def test_set_params_dict(self):
        dataset = Blobs3(random_state=0)
        dataset = clone(dataset).set_params(**{'n_samples':10, 'n_features':3})
        self.assertTupleEqual((dataset.n_samples, dataset.n_features), (10, 3))
