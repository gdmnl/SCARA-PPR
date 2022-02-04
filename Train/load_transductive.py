import gc
import numpy as np
import scipy.sparse
import sklearn
import torch

from data_processor import DataProcess


np.set_printoptions(linewidth=150, edgeitems=5,
                    formatter=dict(float=lambda x: "% 9.3e" % x))


def matnorm_adj(m, deg, norm, idx):
    """PPR matrix normalization
    """
    if norm == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = m.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        m.data = deg_sqrt[idx[row]] * m.data * deg_inv_sqrt[col]
    elif norm == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = m.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        m.data = deg[idx[row]] * m.data * deg_inv[col]
    elif norm == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {norm}")
    return m


def matnorm_inf(m, axis=0):
    """L_inf normalization of matrix, scale to [0, 1] and sum of column is 1
    """
    m = m - m.min(axis=axis)
    m = m / m.sum(axis=axis)
    return m


def matstd(m):
    if scipy.sparse.issparse(m):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    else:
        scaler = sklearn.preprocessing.StandardScaler()
    m = scaler.fit_transform(m)
    return m


def matstd_clip(m):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(m[1:, :])
    mean = scaler.mean_
    std = scaler.scale_
    k = 3
    m = np.clip(m, a_min=mean-k*std, a_max=mean+k*std)
    m = scaler.transform(m)
    return m


def diag_sp(diag):
    """scipy sparse diagonal matrix"""
    n = len(diag)
    return scipy.sparse.dia_matrix((diag, [0]), shape=(n, n))


def reserr_d2rmax(drmax, deg, rrz):
    """
    Convert absolute error drmax to rmax in GBP.
    Here use the max degree as d(s).

    Args:
        drmax (float): absolute error, i.e. d(s)^rrz * rmax
        deg (np.array): degree array
        rrz (float): exponent r
    Returns:
        rmax (float): absolute error
    """
    d_max = np.max(deg)
    d_max = np.power(d_max, rrz)
    rmax = drmax / d_max
    return rmax


# ====================
def load_transductive_data(algo, datastr, alpha, eps, rrz, seed=0):
    print('-' * 20)
    print("Start loading...")
    processor = DataProcess(datastr, rrz=rrz, seed=seed)
    processor.input(['deg', 'labels'])
    processor.calculate(['idx_train'])
    # Get graph property
    n, m = processor.n, processor.m
    deg = processor.deg
    labels = torch.LongTensor(processor.labels)
    # Get index
    idx_train = torch.LongTensor(processor.idx_train)
    idx_val = torch.LongTensor(processor.idx_val)
    idx_test = torch.LongTensor(processor.idx_test)

    # Get topk P
    ppr_file = f'../save/{datastr}/feat/{seed}'
    file_weights = f'{ppr_file}/score_{eps:g}.npy'
    weights = np.load(file_weights)
    features = weights
    features = features.transpose()             # shape [n, F]

    # Compute features
    deg_pow = np.power(np.maximum(deg, 1e-12), rrz - 1)
    deg_pow = diag_sp(deg_pow)
    features = deg_pow @ features               # shape [n, F]
    print('-' * 20)
    print(features[:5, :])
    print(features[idx_train, :])
    features = matstd_clip(features)
    # print(features[:5, :])
    # print(features[idx_train, :])
    features = torch.FloatTensor(features)

    print("n={}, m={}, F={}".format(n, m, features.size()))
    print(labels.size(), labels)
    print(idx_train.size(), idx_val.size(), idx_test.size(), idx_train[:10])
    return features, labels, idx_train, idx_val, idx_test


def load_paper(algo, datastr, alpha, eps, rrz, seed=0):
    def diag_mul(m, diag):
        row = m.shape[0]
        for i in range(row):
            m[i] *= diag[i]
        return m

    def lmatstd(m):
        rowh = m.shape[0] // 2
        std = np.std(m[:rowh], axis=0)
        m[:rowh] /= std
        m[rowh:] /= std
        gc.collect()
        return m

    def lmatstd_clip(m, idx):
        std = np.std(m[idx[2]], axis=0)
        k = 5
        for i in range(m.shape[0]):
            m[i] = np.clip(m[i], a_min=-k*std, a_max=k*std) / std
        gc.collect()
        return m

    print('-' * 20)
    print("Start loading...")
    processor = DataProcess(datastr, rrz=rrz, seed=seed)
    processor.input(['deg', 'labels'])
    processor.calculate(['idx_train'])
    # Get graph property
    n, m = processor.n, processor.m
    deg = processor.deg
    labels = torch.LongTensor(processor.labels)
    # Get index
    idx_train = torch.LongTensor(processor.idx_train)
    idx_val = torch.LongTensor(processor.idx_val)
    idx_test = torch.LongTensor(processor.idx_test)
    idx_all = np.concatenate((idx_train, idx_val, idx_test))
    idx_all = np.sort(idx_all)
    idx_all = torch.LongTensor(idx_all)

    # Get topk P
    ppr_file = f'../save/{datastr}/feat/{seed}'
    file_weights = f'{ppr_file}/score_{eps:g}.npy'
    weights = np.load(file_weights)
    features = scipy.sparse.csr_matrix((128, n), dtype=np.float32)
    features[idx_all] = weights
    features = features.transpose()             # shape [n, F]
    del weights
    gc.collect()

    # Compute features
    deg_pow = np.power(np.maximum(deg, 1e-12), rrz - 1)
    features = diag_mul(lmatstd(features), deg_pow)
    print('-' * 20)
    print(features[:5, :])
    print(features[idx_train, :])
    features = lmatstd_clip(features, idx_all)
    # print(features[:5, :])
    # print(features[idx_train, :])
    features = torch.FloatTensor(features)

    print("n={}, m={}, F={}".format(n, m, features.size()))
    print(labels.size(), labels)
    print(idx_train.size(), idx_val.size(), idx_test.size(), idx_train[:10])
    return features, labels, idx_train, idx_val, idx_test
