from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from copy import deepcopy
import numpy as np


class FeatureSelectionMetric:
    def __init__(self, learner=None, random_state=None):
        self.learner = learner
        self.random_state = random_state
        self.value = None
        if learner is None:
            self.learner_ = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                                   random_state=self.random_state)
        else:
            self.learner_ = learner
            self.learner_.random_state = self.random_state
    
    def fit(self, x, y, candidate):
        self.learner_.fit(x[:, candidate], y)


class OOBFeatureSelectionMetric(FeatureSelectionMetric):
    def __init__(self, learner=None, random_state=None):
        super().__init__(learner=learner,
                         random_state=random_state)
        
    def compute_value(self, x, y, candidate):
        self.value = self.learner_.oob_score_
        return self.value


class StabFeatureSelectionMetric(FeatureSelectionMetric):
    def __init__(self, learner=None, random_state=None):
        super().__init__(learner=learner,
                         random_state=random_state)
    
    def fit(self, x, y, candidate):
        # x is a list of x_l, x_u
        x_l = x[0]
        y_l = y
        self.learner_.fit(x_l[:, candidate], y_l)
        
    def compute_value(self, x, y, candidate):
        x_u = x[1]
        max_vote_u = self.learner_.predict_proba(x_u[:, candidate]).max(axis=1)
        self.value = self.learner_.oob_score_ + max_vote_u.mean()
        return self.value


class CVFeatureSelectionMetric(FeatureSelectionMetric):
    def __init__(self, learner=None, n_splits=5, n_jobs=5, random_state=None):
        super().__init__(learner=learner,
                         random_state=random_state)
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.scores = None
    
    def compute_value(self, x, y, candidate):
        kf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        splits = list(kf.split(x, y))
        models = list(map(lambda i: deepcopy(self.learner_), range(self.n_splits)))
        self.scores = np.array(Parallel(n_jobs=self.n_jobs)(delayed(_one_fold_fit)(
            models[i], x[:, candidate], y, splits[i][0], splits[i][1]) for i in range(self.n_splits)))
        self.value = self.scores.mean()
        return self.value


def _one_fold_fit(model, x, y, train_index, test_index):
    model.fit(x[train_index, :], y[train_index])
    y_pred_test = model.predict(x[test_index, :])
    return accuracy_score(y[test_index], y_pred_test)
