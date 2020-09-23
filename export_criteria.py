import numpy as np
import sys


if __name__ == '__main__':
    methods = ['metric_exp/sup_tree', 'relevance_test_exp/True', 'metric_exp/stab', 'metric_exp/rsla', 'metric_exp/tsla']
    datasets = sys.argv[1:]
    for dataset in datasets:
        print(dataset, ":")
        res = list()
        for method in methods:
            base_path = 'out/' + method + '/' + dataset
            res.append(np.loadtxt(base_path + '/acc.txt').mean())
        print(res)
        print()
