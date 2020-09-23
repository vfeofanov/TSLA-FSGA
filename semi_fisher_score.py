import numpy as np
from aux_functions import laplacian


def _semi_fisher_score(x_l, y_l, x_u, knn, lam, delta, normalize):
    x = np.concatenate((x_l, x_u))
    d = x_l.shape[1]
    n = x.shape[0]
    c = np.unique(y_l).size
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    mean_by_class = np.array([np.mean(x_l[y_l == i], axis=0) for i in range(c)])
    var_by_class = np.array([np.var(x_l[y_l == i], axis=0) for i in range(c)])
    l_by_class = np.array([np.sum(y_l == i) for i in range(c)])
    l_by_class_matrix = np.broadcast_to(l_by_class, (d, c)).T
    S_between = np.sum(l_by_class_matrix * ((mean_by_class - mean)**2), axis=0)
    S_within = np.sum(l_by_class_matrix * var_by_class, axis=0)
    numerator = S_between + delta * var
    # computing Laplacian matrix L
    GX = np.dot(x, x.T)
    DX = (np.diag(GX) - GX) + (np.diag(GX) - GX).T
    DX[np.diag_indices(n)] = 1e+15
    # affinity matrix: 1 if one ex. is knn for the other; 0 otherwise
    IDX = np.argsort(DX, axis=1)
    inds = (np.array([[i]*knn for i in range(n)]).ravel(), IDX[:, :knn].ravel())
    S = np.zeros((n, n))
    S[inds] = 1
    S = np.logical_or(S, S.T).astype(np.int)
    L = laplacian(S, zero_diag=False, normalize=normalize)
    # computing J(f_r) = 2 f_r^T L f_r
    J = np.array(list(map(lambda feature: 2 * np.dot(feature, np.matmul(L, feature)), x.T)))
    denominator = S_within + lam * J
    return numerator / denominator


class SemiFisherScore:
    """
    Based on the following paper:
    M. Yang, Y. Chen, G. Ji, Semi_fisher score: a semi-supervised method for feature selection, in:
    Int. Conf. Mach. Learn. Cybern., 2010: pp. 527-532.
    """
    def __init__(self, knn=20, lam=1, delta=1, normalize=False):
        self.knn = knn
        self.lam = lam
        self.delta = delta
        self.normalize = normalize
        self.fscores = None
    
    def fit(self, x_l, y_l, x_u):
        fscores = _semi_fisher_score(x_l, y_l, x_u, self.knn, self.lam, self.delta, self.normalize)
        self.fscores = fscores
        return self
    
    def select_features(self, num_feat):
        subset = np.argsort(self.fscores)[::-1][:num_feat]
        return subset
