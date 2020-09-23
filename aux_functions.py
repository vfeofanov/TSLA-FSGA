import numpy as np
from sklearn.preprocessing import LabelBinarizer


def squared_frobenius_norm(A):
    return (A * A).sum()


def l2p_norm(A, p, epsilon=0):
    return np.sum((np.sum(A*A, axis=1)+epsilon)**(p/2))**(1.0/p)


def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def my_label_binarizer(y):
    if np.unique(y).size > 2:
        lb = LabelBinarizer()
        Y = lb.fit_transform(y)
        return Y
    else:
        Y = np.array(list(map(lambda y_i: [0, 1] if y_i == 1 else [1, 0], y)))
        return Y


def laplacian(W, zero_diag=False, normalize=False):
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

        
def compute_oob_score(model):
    return model.oob_score_


def learn_candidate(x, y, random_state):
    model = RandomForestClassifier(oob_score=True, n_estimators=100, n_jobs=-1, random_state=random_state)
    model.fit(x, y)
    return model
