import numpy as np
import math
from aux_functions import laplacian


def _laplacian_between(y_l, n, zero_diag, normalize):
    W = np.zeros((n, n))
    l = y_l.shape[0]
    Y_l = np.broadcast_to(y_l, (l, l))
    W[:l, :l] = (Y_l!=Y_l.T).astype(np.int)
    if zero_diag:
        W[np.diag_indices(W.shape[0])] = 0
    d = W.sum(axis=1)
    D = np.diag(d)
    L = D - W
    if normalize:
        D_i = np.diag(d ** -0.5)
        return np.matmul(np.matmul(D_i, L), D_i)
    else:
        return L


def _compute_semisup_laplacian_score(L_within, L_between, X):
    numerator = np.array(list(map(lambda feature: np.dot(feature, np.matmul(L_within, feature)), X.T)))
    denominator = np.array(list(map(lambda feature: np.dot(feature, np.matmul(L_between, feature)), X.T)))
    return numerator/denominator


def _semisup_laplacian_score(X_l, y_l, X_u, knn, sigma, zero_diag, normalize):
    l = X_l.shape[0]
    X = np.concatenate((X_l, X_u))
    n = X.shape[0]
    GX = np.dot(X, X.T)
    KX = (np.diag(GX) - GX) + (np.diag(GX) - GX).T
    
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
        
    # construct a weight matrix
    DX = KX.copy()
    DX[np.diag_indices(n)] = 1e+15
    # the mask for lab/unlab, unlabel/unlab examples: 1 if one ex. is knn for the other; 0 otherwise
    IDX = np.argsort(DX, axis=1)
    inds = (np.array([[i]*knn for i in range(n)]).ravel(), IDX[:, :knn].ravel())
    Mask = np.zeros((n, n))
    Mask[inds] = 1
    Mask = np.logical_or(Mask, Mask.T).astype(np.int)
    # Mask_unsup = Mask
    # the mask for labeled examples: 1 if share the same label; 0 otherwise 
    Y_l = np.broadcast_to(y_l, (l, l))
    Mask_semisup = Mask.copy()
    Mask_semisup[:l, :l] = (Y_l == Y_l.T).astype(np.int)
    KX = (-1) * KX * 0.5 / sigma / sigma
    KX = np.exp(KX, KX)
    W_semisup = KX * Mask_semisup
    # W_unsup = KX * Mask_unsup
    
    L_within = laplacian(W_semisup, zero_diag, normalize)
    L_between = _laplacian_between(y_l, n, zero_diag, normalize)
    
    # D_unsup = np.diag(W_unsup.sum(axis=1))
    # D_unsup_sum = D_unsup.sum()
    # D_unsup_col_sum = D_unsup.sum(axis=1)
    # if feat_norm:
    #     X = np.array(list(map(lambda feature: feature-(np.dot(feature, D_unsup_col_sum)/D_unsup_sum), X.T))).T
    return _compute_semisup_laplacian_score(L_within, L_between, X)


class SemiSupervisedLaplacianScore:
    """
    Based on the following paper:
    J. Zhao, K. Lu, X. He, Locality sensitive semi-supervised feature selection, Neurocomputing. 71
    (2008) 1842-1849. doi:10.1016/j.neucom.2007.06.014.
    """
    
    def __init__(self, knn=20, sigma=None, zero_diag=False, normalize=False):
        self.knn = knn
        self.sigma = sigma
        self.zero_diag = zero_diag
        self.normalize = normalize
        self.fscores = None
    
    def fit(self, x_l, y_l, x_u):
        fscores = _semisup_laplacian_score(x_l, y_l, x_u, self.knn, self.sigma, self.zero_diag, self.normalize)
        self.fscores = fscores
        return self
    
    def select_features(self, num_feat):
        subset = np.argsort(self.fscores)[:num_feat]
        return subset
