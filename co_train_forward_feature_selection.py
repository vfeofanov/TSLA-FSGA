import numpy as np
from joblib import Parallel, delayed
from aux_functions import learn_candidate, compute_oob_score
from selection_metrics import CVFeatureSelectionMetric
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy


def compute_score(model, x, y, candidate):
    model.fit(x, y, candidate)
    model.compute_value(x, y, candidate)
    return model


def _supervised_forward_stepwise_fs(x, y, metric, random_state, n_add=None, n_jobs=4):
    d = x.shape[1]
    # final_scores = []
    all_variables = np.arange(d)
    variables = []
    if n_add is None:
        n_add = int(0.1 * d)
    b = True
    while b:
        candidate_variables = np.setdiff1d(all_variables, variables)
        d_now = len(variables)
        d_candidates = len(candidate_variables)
        if d_candidates <= n_add:
            variables.extend(candidate_variables)
            b = False
            continue
        candidates = np.array([[False] * d] * d_candidates)
        rows = np.repeat(np.arange(d_candidates), d_now).astype(np.int)
        cols = np.array(variables * d_candidates).astype(np.int)
        candidates[(rows, cols)] = True
        candidates[np.arange(d_candidates), candidate_variables] = True

        models = list(map(lambda i: deepcopy(metric), range(len(candidates))))
        models = Parallel(n_jobs=n_jobs)(delayed(compute_score)(models[i], x, y, candidates[i]) for i in range(len(candidates)))
        scores = np.array(list(map(lambda model: model.value, models)))

        # models = Parallel(n_jobs=n_jobs)(delayed(learn_candidate)(x[:, candidate], y, random_state)
        #                                  for candidate in candidates)
        # scores = np.array(Parallel(n_jobs=n_jobs)(delayed(compute_oob_score)(models[jj]) for jj in range(d_candidates)))
        to_add = np.argsort(scores)[::-1][:n_add]
        variables.extend(candidate_variables[to_add])
    return np.array(variables)


def _co_train_forward_stepwise(x_l, y_l, x_u, metric, n_feat, n_add, n_add_start, pseudo_label_size, sampling_size,
                               random_state, n_jobs):

    d = x_l.shape[1]
    u = x_u.shape[0]
    if n_feat is None:
        n_feat = d
    if n_add is None:
        n_add = int(0.1 * d)
    if pseudo_label_size is None:
        pseudo_label_size = int(0.5 * u)

    all_variables = np.arange(d)
    variables = []
    # initialization of a feature subset
    variables.extend(np.random.choice(all_variables, n_add_start, replace=False))
    b = True
    while b:
        candidate_variables = np.setdiff1d(all_variables, variables)
        d_now = len(variables)
        d_candidates = len(candidate_variables)
        if d_now >= n_feat:
            variables.extend(candidate_variables)
            b = False
            continue
        if d_candidates < n_add:
            variables.extend(candidate_variables)
            b = False
            continue
        # model = learn_candidate(x_l[:, variables], y_l, random_state=random_state)
        # y_pred_u = model.predict(x_u[:, variables])
        model = metric.learner_
        model.fit(x_l[:, variables], y_l)
        y_pred_u = model.predict(x_u[:, variables])
        pseudo_selected_variables = []
        for s in range(sampling_size):
            idx = np.random.choice(np.arange(u), pseudo_label_size, replace=False)
            x_train = np.concatenate((x_l, x_u[idx, :]))
            y_train = np.concatenate((y_l, y_pred_u[idx]))
            ranking = _supervised_forward_stepwise_fs(x_train, y_train, metric, random_state, n_add=None, n_jobs=n_jobs)
            pseudo_selected_variables.extend(ranking[~np.in1d(ranking, variables)][:n_add])
        pseudo_selected_variables = np.array(pseudo_selected_variables)
        final_candidates = np.unique(pseudo_selected_variables)
        counts = np.bincount(pseudo_selected_variables)
        counts = counts[counts != 0]
        variables.extend(
            final_candidates[counts.argsort()[::-1]][:n_add]
        )
    return np.array(variables)


class CoTrainForwardWrapperSelection:
    """
    to be done
    """

    def __init__(self, metric=None, n_feat=None, n_add=None, n_add_start=5, pseudo_label_size=None, sampling_size=5, random_state=0,
                 n_jobs=4):
        if metric is None:
            learner = DecisionTreeClassifier(min_samples_leaf=5, random_state=random_state)
            self.metric = CVFeatureSelectionMetric(learner=learner, random_state=random_state)
        self.n_feat = n_feat
        self.n_add = n_add
        self.n_add_start = n_add_start
        self.pseudo_label_size = pseudo_label_size
        self.sampling_size = sampling_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.ranking = None

    def fit(self, x_l, y_l, x_u):
        ranking = _co_train_forward_stepwise(x_l, y_l, x_u, self.metric, self.n_feat, self.n_add, self.n_add_start,
                                             self.pseudo_label_size, self.sampling_size, self.random_state, self.n_jobs)
        self.ranking = ranking
        return self

    def select_features(self, num_feat):
        subset = self.ranking[:num_feat]
        return subset
