import numpy as np
import sys


if __name__ == '__main__':
    methods = ['sota_exp/rlsr', 'sota_exp/sfs', 'sota_exp/ssls', 'sota_exp/co_train_fss', 'sota_exp/tsla_fss',
               'metric_exp/tsla']
    datasets = sys.argv[1:]
    for dataset in datasets:
        print(dataset, ":")
        res = list()
        for method in methods:
            base_path = 'out/' + method + '/' + dataset
            res.append(np.loadtxt(base_path + '/acc.txt').mean())
        print(res)
        print()
