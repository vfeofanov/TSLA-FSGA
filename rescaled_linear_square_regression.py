import numpy as np
from aux_functions import squared_frobenius_norm, l2p_norm, projection_simplex_sort, my_label_binarizer
from sklearn.metrics import accuracy_score


def _rlsr(x_l, y_l, x_u, gamma, max_iter_algo_loop, max_iter_solver, epsilon):
        """
        Input
        -----
        x_l: {numpy array}, shape {n labeled samples, n_features}
        y_l: {numpy array}, shape {n labeled samples, n_clusters}
        x_u: {numpy array}, shape {n unlabeled samples, n_features}
        gama: {float}, regular value
        max_iter_algo_loop is maximum number of iterations in main loop
        max_iter_solver maximum number of iterations for solving W
        notes
        -----
        the bigger score for a feature, the more important it is.
        Output
        ------
        theta: feature scores
        Y : result matrix Y
        """

        X_l = x_l.T
        X_u = x_u.T
        Y_l = my_label_binarizer(y_l)

        l = X_l.shape[1]
        n = l + X_u.shape[1]
        d = X_l.shape[0]
        c = Y_l.shape[1]
        X = np.concatenate((x_l, x_u)).T
        Y = np.ones((n, c)) / c
        Y[:l, :] = Y_l

        H = np.eye(n) - np.ones(n) / n
        Q = np.eye(d) / d

        obj = np.zeros((max_iter_algo_loop, 1))

        XHX = np.dot(X, H.dot(X.T))
        for it in range(max_iter_algo_loop):
            XHY = np.dot(X, H.dot(Y))
            # update W with fixed b and Y
            for it2 in range(max_iter_solver):
                W = np.dot(np.linalg.inv(XHX + gamma*Q), XHY)
                temp = ((W*W).sum(axis=1) + epsilon) ** 0.5
                Q = np.diag(np.sum(temp)/temp)
                obj_w = squared_frobenius_norm(H.dot(np.dot(X.T, W)) - H.dot(Y)) +\
                gamma * np.matrix.trace(np.dot(W.T, Q.dot(W)))
                if it2 > 0:
                    change = np.abs(obj_w_old-obj_w) / obj_w_old
                    if change < 1e-7:
                        break
                obj_w_old = obj_w
            b = ((Y.sum(axis=0)).T - (np.dot(W.T, X)).sum(axis=1)) / n

            # update Yu
            for i in range(l, n):
                Y[i, :] = np.dot(X[:, i].T, W) + b.T
                Y[i, :] = projection_simplex_sort(Y[i, :])

            obj[it] = squared_frobenius_norm(np.dot(X.T, W) + np.broadcast_to(b, (n, c)) - Y) + gamma * (l2p_norm(W, 1) ** 2)

            if it == 0:
                minObj = obj[it]
                bestW = W
            else:
                if not np.isnan(obj[it]) and obj[it] <= minObj:
                    minObj = obj[it]
                    bestW = W

            if it > 0:
                change = np.abs((obj[it-1] - obj[it]) / obj[it])
                if change < 1e-8:
                    break

        W = bestW
        theta = (np.sum(W*W, axis=1)) ** 0.5
        theta = theta / theta.sum()
        return theta, Y


class RescaledLinearSquareRegression:
    """
    Based on the following paper:
    Xiaojun Chen, Guowen Yuan, Feiping Nie, Joshua ZheX_ue Huang: Semi-supervised Feature Selection via Rescaled Linear Regression. IJCAI 2017: 1525-1531
    """
    
    def __init__(self, gamma=0.1, max_iter_algo_loop=100, max_iter_solver=10, epsilon=1e-5):
        self.gamma = gamma
        self.max_iter_algo_loop = max_iter_algo_loop
        self.max_iter_solver = max_iter_solver
        self.epsilon = epsilon
        self.fscores = None
        self.y_u_predicted = None
        self.transductive_score = None
    
    def fit(self, x_l, y_l, x_u, y_u=None):
        fscores, Y = _rlsr(x_l, y_l, x_u, self.gamma, self.max_iter_algo_loop, self.max_iter_solver, self.epsilon)
        self.fscores = fscores
        Y_u = Y[x_l.shape[0]:, :]
        # if y_u is provided, the transductive_score is computed
        if y_u is not None:
            self.y_u_predicted = Y_u.argmax(axis=1)
            self.transductive_score = accuracy_score(y_u, self.y_u_predicted)
        return self
    
    def select_features(self, num_feat):
        subset = np.argsort(self.fscores)[::-1][:num_feat]
        return subset
