import unittest
from pyexpt import expts
from pyexpt.datasets import Blobs3
from sklearn.cluster import KMeans

class ExptTestCases(unittest.TestCase):
    def test_1dataset_only(self):
        expt = expts.Expt(alg_list = [None], data_list=[Blobs3()])
        expt.run()
        self.assertEqual(expt.results.shape, (1, 4))

    def test_data_list_only(self):
        n = 4
        expt = expts.Expt(alg_list = [None])
        expt.data_list=[Blobs3(cluster_std=0.2*i) for i in range(1,n+1)]
        expt.run()
        self.assertEqual(expt.results.shape, (n, 4))

    def test_data_list_with_params(self):
        n = 4
        expt = expts.Expt(alg_list = [None])
        expt.data_list=[Blobs3(cluster_std=0.2*i) for i in range(1,n+1)]
        expt.data_params = {'n_samples':[8**i for i in range(1,n+1)]}
        expt.run()
        self.assertEqual(expt.results.shape, (n*n, 4+len(expt.data_params)))

    def test_1algorithm_only(self):
        expt = expts.Expt(alg_list = [KMeans()], data_list = [None])
        expt.run()
        self.assertEqual(expt.results.shape, (1, 4))
