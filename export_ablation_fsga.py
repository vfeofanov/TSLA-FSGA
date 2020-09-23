import numpy as np
import sys


if __name__ == '__main__':
    methods = ['mate_type_exp/standard', 'mate_type_exp/weighted', 'max_num_mut_exp/1-2', 'relevance_test_exp/True']
    datasets = sys.argv[1:]
    for dataset in datasets:
        print(dataset, ":")
        res = list()
        for method in methods:
            base_path = 'out/' + method + '/' + dataset
            res.append(np.loadtxt(base_path + '/acc.txt').mean())
        print(res)
        print()
