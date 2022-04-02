from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import math
import numpy as np
from sklearn import datasets

class Blobs3(BaseEstimator):
    """An example of a BaseDataset class."""
    def __init__(self, n_samples=100, n_features=2, *, 
            cluster_std=1.0, random_state=None,) -> None:
        super().__init__()
        self.n_samples    = n_samples
        self.n_features   = n_features
        self.cluster_std  = cluster_std
        self.random_state = random_state
        self.centers = np.zeros((3,self.n_features))
        self.centers[:,:2] = np.array([[-1, 0], [0, math.sqrt(3)], [1, 0]])
        # generate
        self.X, self.y = datasets.make_blobs(self.n_samples, 
            centers=self.centers, cluster_std=self.cluster_std, 
            random_state=self.random_state)
    
    def __str__(self):
        return f"3 blobs {self.cluster_std}"
