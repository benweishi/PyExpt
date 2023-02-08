import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone, BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
import time

class Expt():
    """A class to run experiments.
    Product run all algorithm in `alg_list` with all dataset in `data_list`, 
    try all params in `alg_params` and `data_params`, repeat `n_repeat` times.

    Parameters
    ----------
    run_func : function(algorithm, dataset), optional
        A function to run the algorithm with the dataset. The default is None.

    alg_list : list of sklearn.base.BaseEstimator, optional
        A list of algorithms to run. The default is [].
    
    data_list : list of sklearn.base.BaseEstimator, optional
        A list of datasets to run. The default is [].
    
    measure_func : function(algorithm, dataset), optional
        A function to measure the performance of the algorithm with the dataset. The default is None.
        The function should return a dictionary of results.
    
    alg_params : dict, optional
        A dictionary of parameters for the algorithms. The default is {}.
    
    data_params : dict, optional
        A dictionary of parameters for the datasets. The default is {}.
    
    n_repeat : int, optional
        Number of times to repeat the experiment, also is used as a part of the random_state for the dataset and the algorithm. The default is 1.
    
    random_state : int, optional
        Random seed. The default is None, which means to randomly generate one for each run.
        The actural random seed is random_state + round.
    """
    def __init__(self, run_func=None, alg_list=[], data_list=[], measure_func=None, alg_params={}, data_params={}, n_repeat=1, random_state=None) -> None:
        self.run_func = run_func
        self.alg_list = alg_list
        self.data_list = data_list
        self.measure_func = measure_func
        self.alg_params = alg_params
        self.data_params = data_params
        self.n_repeat = n_repeat
        self.random_state = random_state
        self.results = None

    def run(self, n_repeat=None, random_state=None):
        """Run the experiments and store the results in self.results as a `pandas.DataFrame`,
        1 row per experiment, with columns: 
        ['round','alg','data','time'] + alg_params.keys() + data_params.keys() + measure_func.keys().

        Parameters
        ----------
        n_repeat : int, optional
            Number of times to repeat the experiment. The default is None, which means to use the value in self.n_repeat.
        
        random_state : int, optional
            Random seed. The default is None, which means to use the value in self.random_state.
        """
        if n_repeat is None:
            n_repeat = self.n_repeat
        if random_state is None:
            random_state = np.random.randint(2**30) if self.random_state is None else self.random_state
        self.results = pd.DataFrame(columns=['round'])
        for r in range(n_repeat):
            result = {'round':r}
            for data, data_params in product(self.data_list, ParameterGrid(self.data_params)):
                if data is None:
                    result |= {'data':'None'}
                else:
                    start_time = time.time()
                    dataset = clone(data).set_params(**data_params, random_state=random_state+r)
                    result |= {'data':str(dataset), 'make_time':time.time()-start_time} | data_params
                for alg, alg_params in product(self.alg_list, ParameterGrid(self.alg_params)):
                    if alg is None:
                        result |= {'alg':'None'}
                    else:
                        start_time = time.time()
                        algorithm = clone(alg).set_params(**alg_params)
                        result |= {'alg':str(algorithm), 'init_time':time.time()-start_time} | alg_params
                        if self.run_func is not None:
                            start_time = time.time()
                            self.run_func(algorithm, dataset)
                            result |= {'run_time':time.time()-start_time}
                    if self.measure_func is not None:
                        start_time = time.time()
                        result |= self.measure_func(algorithm, dataset) | {'measure_time':time.time()-start_time}
                    self.results = pd.concat([self.results, pd.DataFrame([result])], ignore_index=True)
        return self

    def plot(self, x_list, y_list, group):
        """Plot the results."""
        n_row = len(y_list)
        n_col = len(x_list)
        fig, axs = plt.subplots(n_row, n_col, squeeze=False, sharex=True, figsize=(0.5+5.5*n_col,0.5+3.5*n_row))
        for r in range(n_row):
            for c in range(n_col):
                sns.lineplot(data=self.results, x=x_list[c], y=y_list[r], hue="alg", ax=axs[r, c])