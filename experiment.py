from feature_selection_genetic_algorithm import FeatureSelectionGeneticAlgorithm
from co_train_forward_feature_selection import CoTrainForwardWrapperSelection
from rescaled_linear_square_regression import RescaledLinearSquareRegression
from semi_supervised_laplacian_score import SemiSupervisedLaplacianScore
from forward_feature_selection import ForwardWrapperSelection
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from semi_fisher_score import SemiFisherScore
from self_learning import SelfLearning
from data_setter import DataSetter
from selection_metrics import *
from copy import deepcopy
from time import time
import numpy as np
import json
import sys
import os


def param_values_converter(param_name, param_values):
    # params, which values are strings
    string_params = ['mate_type', 'metric']
    if param_name in string_params:
        return param_values
    if param_name == 'max_num_mut':
        converted_param_values = list()
        for param_value in param_values:
            if param_value in ['1-3', '1-2', '2-3']:
                converted_param_values.append(param_value)
            else:
                converted_param_values.append(int(param_value))
        return converted_param_values
    if param_name == 'w_threshold':
        converted_param_values = list()
        for param_value in param_values:
            if param_value in ['1-d', '1-5d']:
                converted_param_values.append(param_value)
            else:
                converted_param_values.append(float(param_value))
        return converted_param_values
    if param_name in ['relevance_test']:
        converted_param_values = list(map(lambda value: bool(value), param_values))
        return converted_param_values


def export_json(dictionary, path):
    with open(path, 'w') as f:
        json.dump(dictionary, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _create_ga_experiment(param_name, values):
    params = {
        'num_generations': 20,
        'num_candidates': 40,
        'num_parents': 8,
        'rand_cross_point': True,
        'prob_mut': 1,
        'max_num_mut': '1-2',
        'w_threshold': '1-5d',
        'metric': 'sup_rf',
        'report': 0,
        'relevance_test': True,
        'mate_type': 'weighted',
        'relevance_threshold': 1e-04,
        'n_feat': None,
    }
    if param_name == 'mate_type':
        params['prob_mut'] = 0
        params['max_num_mut'] = 0
        params['relevance_test'] = False
    if param_name == 'max_num_mut':
        params['relevance_test'] = False
     
    path = 'out/' + param_name + '_exp/'
    if not os.path.isdir(path):
        os.makedirs(path)
    for value in values:
        if not os.path.isdir(path + str(value)):
            os.makedirs(path + str(value))
        params[param_name] = value
        export_json(params, path + str(value) + '/ga_config.json')


def convert_metric(params, random_state):
    if params['metric'] is not None:
        if params['metric'] in ['tsla', 'sup_rf', 'rsla']:
            params['metric'] = OOBFeatureSelectionMetric(random_state=random_state)
        if params['metric'] == 'sup_tree':
            learner = DecisionTreeClassifier(min_samples_leaf=5, random_state=random_state)
            params['metric'] = CVFeatureSelectionMetric(learner=learner, random_state=random_state)
        if params['metric'] == 'stab':
            params['metric'] = StabFeatureSelectionMetric(random_state=random_state)
    return params


class Experiment:
    def __init__(self, db_name, n_jobs=4, n_exp=20):
        self.db_name = db_name
        self.n_jobs = n_jobs
        self.n_exp = n_exp
        ds = DataSetter(self.db_name)
        self.x = ds.x
        self.y = ds.y
        self.unlab_size = ds.unlab_size

    def ga_experiment(self, param_name, values):
        _create_ga_experiment(param_name, values)
        for value in values:
            base_path = 'out/' + param_name + '_exp/' + str(value)
            params = load_json(base_path + '/ga_config.json')
            params['n_jobs'] = self.n_jobs
            acc = list()
            n_feat_res = list()
            subset_res = list()
            time_res = list()
            if params['relevance_test']:
                n_feat_removed_res = list()
            else:
                n_feat_removed_res = None
            for rand_state in range(self.n_exp):
                # split data
                x_l, x_u, y_l, y_u = train_test_split(self.x, self.y, test_size=self.unlab_size, 
                                                      stratify=self.y, random_state=rand_state)
                params_now = deepcopy(params)
                params_now['random_state'] = rand_state
                if self.db_name == 'synthetic':
                    params_now['n_feat'] = 8
                metric_name = params['metric']
                params_now = convert_metric(params_now, rand_state)

                # feature selection
                fs_model = FeatureSelectionGeneticAlgorithm(**params_now)
                t0 = time()
                if 'sla' in metric_name:
                    if 'rsla' in metric_name:
                        model = SelfLearning(policy='rsla', random_state=rand_state)
                    else:
                        model = SelfLearning(policy='tsla', random_state=rand_state)
                    model.fit(x_l, y_l, x_u)
                    x_pl = model.x_pl
                    y_pl = model.y_pl
                    fs_model.fit(x_pl, y_pl)
                if 'sup' in metric_name:
                    fs_model.fit(x_l, y_l)
                if 'stab' in metric_name:
                    fs_model.fit([x_l, x_u], y_l)
                subset = fs_model.subset
                t1 = time()

                # learning on the final subset
                model = SelfLearning(policy='tsla', random_state=rand_state)
                model.fit(x_l[:, subset], y_l, x_u[:, subset])
                y_u_pred = model.predict(x_u[:, subset])
                acc.append(accuracy_score(y_u, y_u_pred))
                n_feat_res.append(subset.sum())
                subset_res.append(subset)
                time_res.append(t1-t0)
                if n_feat_removed_res is not None:
                    n_feat_removed_res.append(fs_model.n_feat_removed)
                print(value, ':', rand_state)
            save_output(base_path, self.db_name, acc, n_feat_res, subset_res, time_res, n_feat_removed_res)
    
    def sota_experiment(self, methods):
        check_path_existence('out/sota_exp/')
        if self.db_name == 'synthetic':
            n_feat = 8
        else:
            n_feat = int(np.sqrt(self.x.shape[1]))
        for method in methods:
            base_path = 'out/sota_exp/' + method
            check_path_existence(base_path)
            acc = list()
            n_feat_res = [n_feat] * self.n_exp
            subset_res = list()
            time_res = list()
            n_feat_removed_res = None
            for rand_state in range(self.n_exp):
                # split data
                x_l, x_u, y_l, y_u = train_test_split(self.x, self.y, test_size=self.unlab_size,
                                                      stratify=self.y, random_state=rand_state)

                # feature selection
                fs_model = initialize_fs_model(method, n_feat, rand_state, self.n_jobs)
                t0 = time()
                if method == 'tsla_fss':
                    model = SelfLearning(policy='tsla', random_state=rand_state)
                    model.fit(x_l, y_l, x_u)
                    x_pl = model.x_pl
                    y_pl = model.y_pl
                    fs_model.fit(x_pl, y_pl)
                else:
                    fs_model.fit(x_l, y_l, x_u)
                subset = fs_model.select_features(n_feat)
                t1 = time()
                if t1-t0 >= 3600:
                    print('The time limit is exceeded!')

                # learning on the final subset
                model = SelfLearning(policy='tsla', random_state=rand_state)
                model.fit(x_l[:, subset], y_l, x_u[:, subset])
                y_u_pred = model.predict(x_u[:, subset])
                acc.append(accuracy_score(y_u, y_u_pred))
                subset_res.append(subset)
                time_res.append(t1-t0)
                print(method, ':', rand_state)
            save_output(base_path, self.db_name, acc, n_feat_res, subset_res, time_res, n_feat_removed_res)

    def no_selection_experiment(self, methods):
        check_path_existence('out/no_sel_exp/')
        for method in methods:
            base_path = 'out/no_sel_exp/' + method
            check_path_existence(base_path)
            acc = list()
            time_res = list()
            for rand_state in range(self.n_exp):
                x_l, x_u, y_l, y_u = train_test_split(self.x, self.y, test_size=self.unlab_size, 
                                                      stratify=self.y, random_state=rand_state)
                t0 = time()
                if method == 'tsla':
                    model = SelfLearning(policy='tsla', random_state=rand_state)
                if method == 'rsla':
                    model = SelfLearning(policy='rsla', random_state=rand_state)
                model.fit(x_l, y_l, x_u)
                y_u_pred = model.predict(x_u)
                acc.append(accuracy_score(y_u, y_u_pred))
                t1 = time()
                time_res.append(t1-t0)
                print(method, ':', rand_state)
            path = base_path + '/' + self.db_name
            check_path_existence(path)
            np.savetxt(path + '/acc.txt', acc)
            np.savetxt(path + '/time.txt', time_res)
 

def check_path_existence(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_output(base_path, db_name, acc, n_feat_res, subset_res, time_res, n_feat_removed_res=None):
    path = base_path + '/' + db_name
    check_path_existence(path)
    np.savetxt(path + '/acc.txt', acc)
    np.savetxt(path + '/n_feat.txt', n_feat_res)
    np.savetxt(path + '/subset.txt', subset_res)
    np.savetxt(path + '/time.txt', time_res)
    if n_feat_removed_res is not None:
        np.savetxt(path + '/n_feat_removed.txt', n_feat_removed_res)


def initialize_fs_model(method_name, n_feat, random_state, n_jobs):
    if method_name == 'rlsr':
        fs_model = RescaledLinearSquareRegression(gamma=0.1)
    if method_name == 'ssls':
        fs_model = SemiSupervisedLaplacianScore()
    if method_name == 'sfs':
        fs_model = SemiFisherScore()
    if method_name == 'co_train_fss':
        fs_model = CoTrainForwardWrapperSelection(n_feat=n_feat, random_state=random_state, n_jobs=n_jobs)
    if method_name == 'tsla_fss':
        metric = OOBFeatureSelectionMetric(random_state=random_state)
        fs_model = ForwardWrapperSelection(metric=metric, random_state=random_state, n_jobs=n_jobs)
    return fs_model


if __name__ == '__main__':
    data_name = sys.argv[1]
    num_jobs = int(sys.argv[2])
    num_exp = int(sys.argv[3])
    exp = Experiment(data_name, num_jobs, num_exp)
    arg_name = sys.argv[4]
    arg_values = sys.argv[5:]
    if arg_name == 'sota':
        exp.sota_experiment(arg_values)
    elif arg_name == 'no_selection':
        exp.no_selection_experiment(arg_values)
    else:
        arg_values = param_values_converter(arg_name, arg_values)
        exp.ga_experiment(arg_name, arg_values)
