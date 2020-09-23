import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
import self_learning_cython as slc


def joint_bayes_risk(margin, pred, i, j, theta, samplingRate=50):
    # li = \sum_{x\in X_U} \I{y=i} =approx.= \sum_{x\in X_U} m_Q(x,i)
    li = np.sum(margin[:, i])
    margins = margin[:, j]
    # gammas = sorted(list(set(margins[margins > theta])))
    gammas = theta + (1 - theta) * (np.arange(samplingRate) + 1) / samplingRate
    infimum = 1e+05
    upperBounds = []
    # for gamma in gammas:
    for n in range(np.size(gammas)):
        gamma = gammas[n]
        I_ij = np.sum(margin[np.array((margins < gamma) & (margins >= theta)), i]) / li
        K_ij = np.dot(margin[:, i], np.array(pred == j) * margins) / li
        # M-less of gamma
        Mg_ij = np.dot(margin[:, i], np.array(margins < gamma) * margins) / li
        # M-less of theta
        Mt_ij = np.dot(margin[:, i], np.array(margins < theta) * margins) / li
        A = K_ij + Mt_ij - Mg_ij
        upperBound = I_ij + (A * (A > 0)) / gamma
        upperBounds.append(upperBound)
        if upperBound < infimum:
            infimum = upperBound
        if n > 3:
            if upperBounds[-1] > upperBounds[-2] and upperBounds[-2] >= upperBounds[-3]:
                break
    return infimum


def optimal_threshold_vector(margin, pred, K, samplingRate=50):
    theta = []

    def Reduction(matrix, margin):
        K = margin.shape[1]
        u = margin.shape[0]
        countClass = np.array([np.sum(margin[:, j]) for j in range(K)])
        return (1 / u) * np.dot(countClass, np.sum(matrix, axis=1))

    u = margin.shape[0]
    for k in range(K):
        # A set of possible thetas:
        theta_min = np.min(margin[:, k])
        theta_max = np.max(margin[:, k])
        thetas = theta_min + np.arange(samplingRate) * (theta_max - theta_min) / samplingRate
        JBR = []
        BE = []
        for n in range(samplingRate):
            matrix = np.zeros((K, K))
            for i in range(K):
                if i == k:
                    continue
                else:
                    matrix[i, k] = joint_bayes_risk(margin, pred, i, k, thetas[n])
                    if (i == 0) and (k == 1):
                        JBR.append(matrix[i, k])

            pbl = (1 / u) * np.sum((margin[:, k] >= thetas[n]) & (pred == k))
            if pbl == 0:
                pbl = 1e-15
            BE.append(Reduction(matrix, margin)/pbl)
            if n > 3:
                if BE[-1] > BE[-2] and BE[-2] >= BE[-3]:
                    break
        BE = np.array(BE)
        num = np.argmin(BE)
        if type(num) is list:
            num = num[0]
        theta.append(thetas[num])
    return np.array(theta)


def tsla(x_l, y_l, x_u, cython=True, base_classifier=None, **kwargs):
    """
    Based on the following paper:
    Feofanov, Vasilii, Emilie Devijver, and Massih-Reza Amini.
    "Transductive Bounds for the Multi-Class Majority Vote Classifier."
    In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, pp. 3566-3573. 2019.
    :param x_l: Labeled observations.
    :param y_l: Labels.
    :param x_u:  Unlabeled data. Will be used for learning.
    :param cython:  Whether or not to use cython code, which gives speedup in computation. The default value is True.
    :param base_classifier: The base classifier. By default, it is a Random Forest with 100 trees.
    :return: The final classification model H that has been trained on (x_l, y_l)
    and pseudo-labeled (x_u, hat{y}_u).
    """

    if 'random_state' not in kwargs:
        rand_state = None
    else:
        rand_state = kwargs['random_state']

    if base_classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=rand_state)
    else:
        classifier = base_classifier

    l = x_l.shape[0]
    sample_distr = np.repeat(1 / l, l)
    K = np.unique(y_l).shape[0]
    b = True
    thetas = []
    while b:
        u = x_u.shape[0]
        # Learn a classifier
        classifier.fit(x_l, y_l, sample_weight=sample_distr)
        if len(thetas) == 0:
            classifier_0 = classifier
        vote_u = classifier.predict_proba(x_u)
        pred_u = np.argmax(vote_u, axis=1)

        # Find a threshold minimizing Bayes conditional error
        if cython:
            theta = slc.c_optimal_threshold_vector(vote_u, pred_u, K)
        else:
            theta = optimal_threshold_vector(vote_u, pred_u, K)
        thetas.append(theta)

        # Select observations with argmax vote more than corresponding theta
        selection = np.array(vote_u[np.arange(u), pred_u] >= theta[pred_u])
        x_s = x_u[selection, :]
        y_s = pred_u[selection]
        # Stop if there is no anything to add:
        if x_s.shape[0] == 0:
            b = False
            continue

        # Move them from the unlabeled set to the train one
        x_l = np.concatenate((x_l, x_s))
        y_l = np.concatenate((y_l, y_s))
        x_u = np.delete(x_u, np.where(selection), axis=0)
        s = x_l.shape[0] - l
        sample_distr = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / s, s)))

        # Stop criterion
        if x_u.shape[0] == 0:
            b = False
    classifier.fit(x_l, y_l, sample_weight=sample_distr)
    return classifier, x_l, y_l, thetas, classifier_0


def fsla(x_l, y_l, x_u, theta, max_iter, base_classifier=None, **kwargs):
    """
    A policy for self-learning to pseudo-label examples with prediction vote above a threshold.
    :param x_l: Labeled observations.
    :param y_l: Labels.
    :param x_u:  Unlabeled data. Will be used for learning.
    :param theta: Threshold on the prediction vote. When it's equal to 'mean', it is set to the mean prediction vote.
    :param max_iter: A maximum number of iterations that self-learning does.
    :param base_classifier: The base classifier. By default, it is a Random Forest with 100 trees.
    :return: The final classification model H that has been trained on (x_l, y_l)
    and pseudo-labeled (x_u, hat{y}_u).
    """

    if 'random_state' not in kwargs:
        rand_state = None
    else:
        rand_state = kwargs['random_state']

    if base_classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=rand_state)
    else:
        classifier = base_classifier

    l = x_l.shape[0]
    sample_distr = np.repeat(1 / l, l)
    n = 1
    b = True
    while b:
        u = x_u.shape[0]
        # Learn a classifier
        classifier.fit(x_l, y_l, sample_weight=sample_distr)
        if n == 1:
            classifier_0 = classifier
        vote_u = classifier.predict_proba(x_u)
        pred_u = np.argmax(vote_u, axis=1)
        pred_vote = np.max(vote_u, axis=1)
        if theta is 'mean':
            theta_ = pred_vote.mean()
        else:
            theta_ = theta

        # Select observations with argmax vote more than corresponding theta
        selection = np.array(vote_u[np.arange(u), pred_u] >= theta_)
        x_s = x_u[selection, :]
        y_s = pred_u[selection]

        # Move them from the unlabeled set to the train one
        x_l = np.concatenate((x_l, x_s))
        y_l = np.concatenate((y_l, y_s))
        x_u = np.delete(x_u, np.where(selection), axis=0)

        s = x_l.shape[0] - l
        if x_s.shape[0] == 0:
            b = False
            continue
        sample_distr = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / s, s)))

        # Stop criterion
        if x_u.shape[0] == 0:
            b = False
        n += 1
        if n == max_iter:
            b = False
    classifier.fit(x_l, y_l, sample_weight=sample_distr)
    return classifier, x_l, y_l, classifier_0


def rsla(x_l, y_l, x_u, prop, base_classifier=None, **kwargs):
    """
    A self-learning algorithm where a random portion of unlabeled examples are pseudo-labeled at each iteration.
    :param x_l: Labeled observations.
    :param y_l: Labels.
    :param x_u:  Unlabeled data. Will be used for learning.
    :param prop: A proportion of unlabeled examples taken at each iteration.
    :param base_classifier: The base classifier. By default, it is a Random Forest with 100 trees.
    :return: The final classification model H that has been trained on (x_l, y_l)
    and pseudo-labeled (x_u, hat{y}_u).
    """

    if 'random_state' not in kwargs:
        rand_state = None
    else:
        rand_state = kwargs['random_state']

    if base_classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=rand_state)
    else:
        classifier = base_classifier

    np.random.seed(rand_state)
    l = x_l.shape[0]
    u = x_u.shape[0]
    pl = 0
    n_take = np.int(prop * u)

    sample_distr = np.repeat(1 / l, l)
    n = 1
    b = True
    while b:
        # Learn a classifier
        classifier.fit(x_l, y_l, sample_weight=sample_distr)
        if n == 1:
            classifier_0 = classifier
        vote_u = classifier.predict_proba(x_u)
        pred_u = np.argmax(vote_u, axis=1)
        if u < n_take:
            n_take = u
        # Select observations with argmax vote more than corresponding theta
        selection = np.random.choice(np.arange(u), n_take, replace=False)
        x_s = x_u[selection, :]
        y_s = pred_u[selection]
        # Move them from the unlabeled set to the train one
        x_l = np.concatenate((x_l, x_s))
        y_l = np.concatenate((y_l, y_s))
        x_u = np.delete(x_u, np.where(selection), axis=0)
        s = selection.size
        pl += s
        u -= s
        if x_s.shape[0] == 0:
            b = False
            continue
        sample_distr = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / pl, pl)))
        # Stop criterion
        if x_u.shape[0] == 0:
            b = False
    classifier.fit(x_l, y_l, sample_weight=sample_distr)
    return classifier, x_l, y_l, classifier_0


class SelfLearning:
    def __init__(self, base_classifier=None, policy='tsla', fixed_theta=None, max_iter=None, cython=True, prop=0.1,
                 random_state=None):

        if base_classifier is None:
            self.base_classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                                          random_state=random_state)
        self.policy = policy
        if policy == 'tsla':
            self.cython = cython
            self.thetas = None
        if policy == 'fsla':
            self.fixed_theta = fixed_theta
            if max_iter is None:
                self.max_iter = 10
            else:
                self.max_iter = max_iter
        if policy == 'rsla':
            if prop is None:
                self.prop = 0.1
            else:
                self.prop = prop
        self.random_state = random_state
        self.oob_decision_function_ = None
        self.feature_importances_ = None
        self.x_l = None
        self.y_l = None
        self.x_u = None
        self.x_pl = None
        self.y_pl = None
        self.final_model = None
        self.init_model = None

    def fit(self, x_l, y_l, x_u):
        self.x_l = x_l
        self.y_l = y_l
        self.x_u = x_u
        if self.policy == 'tsla':
            self.final_model, self.x_pl, self.y_pl, self.thetas, self.init_model = tsla(x_l, y_l, x_u,
                                                                                        cython=self.cython,
                                                                                        base_classifier=self.base_classifier)
        if self.policy == 'fsla':
            self.final_model, self.x_pl, self.y_pl, self.init_model = fsla(x_l, y_l, x_u, theta=self.fixed_theta,
                                                                           max_iter=self.max_iter,
                                                                           base_classifier=self.base_classifier)

        if self.policy == 'rsla':
            self.final_model, self.x_pl, self.y_pl, self.init_model = rsla(x_l, y_l, x_u, prop=self.prop,
                                                                           base_classifier=self.base_classifier)

        if self.final_model.__class__.__name__ == 'RandomForestClassifier':
            self.oob_decision_function_ = self.final_model.oob_decision_function_
        self.feature_importances_ = self.final_model.feature_importances_

    def predict(self, x):
        return self.final_model.predict(x)

    def predict_proba(self, x, init_model=False):
        if init_model:
            return self.init_model.predict_proba(x)
        else:
            return self.final_model.predict_proba(x)