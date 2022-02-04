import numpy as np
import torch

from load_transductive import diag_sp, matstd_clip
from data_processor import DataProcess


np.set_printoptions(linewidth=150, edgeitems=5, threshold=10,
                    formatter=dict(float=lambda x: "% 9.3e" % x))

# ====================
def load_inductive_data(algo, datastr, alpha, eps, rrz, seed=0):
    print('-' * 20)
    print("Start loading...")
    processor = DataProcess(datastr, rrz=rrz, seed=seed)
    processor.input(['deg', 'labels', 'idx_train', 'idx_val', 'idx_test'])
    processor.calculate(['labels_oh'])
    # Get graph property
    n, m = processor.n, processor.m
    deg = processor.deg
    labels = torch.LongTensor(processor.labels_oh)
    # Get index
    idx_train = torch.LongTensor(processor.idx_train)
    idx_val = torch.LongTensor(processor.idx_val)
    idx_test = torch.LongTensor(processor.idx_test)

    print('-' * 20)
    def precompute(ds, idx=None):
        # Get topk P
        ppr_file = f'../save/{datastr}/{ds}/{seed}'
        file_weights = f'{ppr_file}/score_{eps:g}.npy'
        weights = np.load(file_weights)
        features = weights
        features = features.transpose()             # shape [n, F]
        if idx is None:
            degree = deg
        else:
            features = features[idx]
            degree = deg[idx]

        # Compute features
        deg_pow = np.power(np.maximum(degree, 1e-12), rrz - 1)
        deg_pow = diag_sp(deg_pow)
        features = deg_pow @ features               # shape [n, F]
        features = matstd_clip(features)
        return features

    features = precompute('feat', idx=None)
    features_train = precompute('feat_train', idx=idx_train)
    print(features_train[:5, :])
    print(features[idx_train, :])
    print(features[-5:, :])
    features = torch.FloatTensor(features)
    features_train = torch.FloatTensor(features_train)

    print("n={}, m={}, F_t={}, F={}".format(n, m, features_train.size(), features.size()))
    print(labels.size(), labels)
    print(idx_train.size(), idx_val.size(), idx_test.size(), idx_train[:10])
    return features_train, features, labels, idx_train, idx_val, idx_test
