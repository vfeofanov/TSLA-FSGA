import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from selection_metrics import CVFeatureSelectionMetric


def compute_score(model, x, y, candidate):
    model.fit(x, y, candidate)
    model.compute_value(x, y, candidate)
    return model


def _forward_stepwise_fs(x, y, metric, random_state, n_add=None, n_jobs=4):
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

        models = list(map(lambda i: deepcopy(metric), range(d_candidates)))
        models = Parallel(n_jobs=n_jobs)(delayed(compute_score)(models[i], x, y, candidates[i]) for i in range(d_candidates))
        scores = np.array(list(map(lambda model: model.value, models)))
        to_add = np.argsort(scores)[::-1][:n_add]
        variables.extend(candidate_variables[to_add])
        # final_scores.append(scores[to_delete])
    return np.array(variables)


class ForwardWrapperSelection:
    """
    to be done
    """

    def __init__(self, metric=None, random_state=0, n_add=None, n_jobs=4):
        if metric is None:
            learner = DecisionTreeClassifier(min_samples_leaf=5, random_state=random_state)
            self.metric = CVFeatureSelectionMetric(learner=learner, random_state=random_state)
        else:
            self.metric = metric
        self.random_state = random_state
        self.n_add = n_add
        self.n_jobs = n_jobs
        self.ranking = None

    def fit(self, x, y):
        ranking = _forward_stepwise_fs(x, y, self.metric, self.random_state, self.n_add, self.n_jobs)
        self.ranking = ranking
        return self

    def select_features(self, num_feat):
        subset = self.ranking[:num_feat]
        return subset
