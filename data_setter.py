from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat
import scipy.sparse as sp
import pandas as pd
import numpy as np


def _read_mnist(hog_feat=False):
    if hog_feat:
        x = sp.load_npz('data/mnist-hog-x.npz')
        x = x.toarray().view(type=np.ndarray)
        y = np.loadtxt('data/mnist-hog-y.txt', dtype=np.int)
    else:
        (mnist_train, y_train), (mnist_test, y_test) = mnist.load_data()
        mnist_train = (mnist_train > 0).reshape(60000, 28, 28).astype(np.uint8) * 255
        x_train = mnist_train.reshape((60000, 28*28), order='F')
        x, x_rest, y, y_rest = train_test_split(x_train, y_train, test_size=50000, random_state=0)
    return x, y


def _read_mice_protein():
    df = pd.read_csv("data/mice_protein_expression.csv", index_col=0)
    x = (df.iloc[:, :77]).values
    le = LabelEncoder()
    y = le.fit_transform((df.iloc[:, 77]).values)
    return x, y


def _read_fashion():
    (mnist_train, y_train), (mnist_test, y_test) = fashion_mnist.load_data()
    mnist_train = (mnist_train > 0).reshape(60000, 28, 28).astype(np.uint8) * 255
    x_train = mnist_train.reshape((60000, 28*28), order='F')
    x, x_rest, y, y_rest = train_test_split(x_train, y_train, test_size=50000, random_state=0, stratify=y_train)
    return x, y


class DataSetter:
    def __init__(self, db_name):
        self.db_name = db_name
        if self.db_name == "coil20":
            mat = loadmat("data/COIL20.mat")
            self.x = mat['X']
            self.y = mat['Y'].ravel()
            self.y -= 1
            self.unlab_size = 0.9
        if self.db_name == "mnist":
            self.x, self.y = _read_mnist()
            self.unlab_size = 0.99
        if self.db_name == "fashion":
            self.x, self.y = _read_fashion()
            self.unlab_size = 0.99
        if self.db_name == "protein":
            self.x, self.y = _read_mice_protein()
            self.unlab_size = 0.9
        if self.db_name == "madelon":
            mat = loadmat("data/madelon.mat")
            self.x = mat['X']
            self.y = mat['Y'].ravel()
            self.y[self.y == -1] = 0
            self.unlab_size = 0.9
        if self.db_name == "gisette":
            mat = loadmat("data/gisette.mat")
            self.x = mat['X']
            self.y = mat['Y'].ravel()
            self.y[self.y == -1] = 0
            self.unlab_size = 0.99
        if self.db_name == "pcmac":
            mat = loadmat("data/PCMAC.mat")
            self.x = mat['X']
            self.y = mat['Y'].ravel()
            self.y -= 1
            self.unlab_size = 0.9
        if self.db_name == "relathe":
            mat = loadmat("data/RELATHE.mat")
            self.x = mat['X']
            self.y = mat['Y'].ravel()
            self.y -= 1
            self.unlab_size = 0.9
        if self.db_name == "basehock":
            mat = loadmat("data/BASEHOCK.mat")
            self.x = mat['X']
            self.y = mat['Y'].ravel()
            self.y -= 1
            self.unlab_size = 0.9
        if self.db_name == "isolet":
            mat = loadmat("data/Isolet.mat")
            self.x = mat['X']
            self.y = mat['Y'].ravel()
            self.y -= 1
            self.unlab_size = 0.9
        if self.db_name == "synthetic":
            self.x, self.y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=8,
                                                 n_redundant=6, n_repeated=0, class_sep=2, random_state=0,
                                                 shuffle=False)
            self.unlab_size = 900
