import gc
import numpy as np
# import scipy.sparse as sp
import sklearn
import torch

from data_processor import DataProcess, diag_sp


np.set_printoptions(linewidth=160, edgeitems=5,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)


def lmatstd(m):
    """Large matrix standardization"""
    rowh = m.shape[0] // 2
    std = np.std(m[:rowh], axis=0)
    m[:rowh] /= std
    m[rowh:] /= std
    gc.collect()
    return m


def lmatstd_clip(m, with_mean=False):
    """Large matrix standardization with clipping"""
    rowh = m.shape[0] // 10
    scaler = sklearn.preprocessing.StandardScaler(with_mean=with_mean)
    scaler.fit(m[rowh:2*rowh, :])
    mean = scaler.mean_
    std = scaler.scale_
    k = 3
    m = np.clip(m, a_min=mean-k*std, a_max=mean+k*std)
    m = scaler.transform(m)
    return m


def diag_mul(diag, m):
    """Diagonal matrix multiplication"""
    row = m.shape[0]
    for i in range(row):
        m[i] *= diag[i]
    return m


# ====================
def load_data(algo: str, datastr: str, datapath: str,
              inductive: bool, multil: bool, spt: int,
              alpha: float, eps: float, rrz: float, seed: int=0):
    print('-' * 20)
    # print("Start loading...")
    # Get degree and label
    processor = DataProcess(datastr, path=datapath, rrz=rrz, seed=seed)
    processor.input(['deg', 'labels'])
    deg = processor.deg
    if multil:
        processor.calculate(['labels_oh'])
        labels = torch.LongTensor(processor.labels_oh)
        labels = labels.float()
    else:
        labels = torch.LongTensor(processor.labels)
    # Get index
    if inductive:
        processor.input(['idx_train', 'idx_val', 'idx_test'])
    else:
        processor.calculate(['idx_train'])
    idx = {'train': torch.LongTensor(processor.idx_train),
           'val': torch.LongTensor(processor.idx_val),
           'test': torch.LongTensor(processor.idx_test)}
    # Get graph property
    n, m = processor.n, processor.m

    # Precompute integration
    def precompute(algo_i):
        # Load embedding
        est_dir = f'../save/{datastr}/{algo_i}/{seed}'
        if spt == 1:
            feat_file = f'{est_dir}/score_{alpha:g}_{eps:g}.npy'
            features = np.load(feat_file)
        else:
            # features = sp.lil_matrix((200, n), dtype=np.float32)
            features = None
            for i in range(spt):
                feat_file = f'{est_dir}/score_{alpha:g}_{eps:g}_{i}.npy'
                features_spt = np.load(feat_file)
                if features is None:
                    features = features_spt.astype(np.float32)
                else:
                    np.concatenate((features, features_spt), axis=0, out=features, dtype=np.float32)
                print(f'  Split {i} loaded, now shape: {features.shape}')
        features = features.transpose()                 # shape [n, F]
        print('all head no norm')
        print(features[:5, :])
        print('train head no norm ', idx['train'][:5])
        print(features[idx['train'][:5], :])

        # Process degree
        if algo_i.endswith('_train'):
            processor_i = DataProcess(datastr+'_train', path=datapath, rrz=rrz, seed=seed)
            processor_i.input(['deg'])
            deg_i = processor_i.deg
        else:
            deg_i = deg
        idx_zero = np.where(deg_i == 0)[0]
        if len(idx_zero) > 0:
            print(f"Warning: {len(idx_zero)} isolated nodes found: {idx_zero}!")
        deg_pow = np.power(np.maximum(deg_i, 1e-12), rrz - 1)
        deg_pow[idx_zero] = 0

        # Normalize embedding by degree
        if spt == 1:
            deg_pow = diag_sp(deg_pow)
            features = deg_pow @ features               # shape [n, F]
            features = lmatstd_clip(features)
        else:
            features = diag_mul(deg_pow, lmatstd(features))
            features = lmatstd_clip(features)
        return features

    # Assign features
    features = precompute(f'{algo}')
    print('all head ')
    print(features[:5, :])
    feat = {'val': torch.FloatTensor(features[idx['val']]),
            'test': torch.FloatTensor(features[idx['test']])}
    if inductive:
        features_train = precompute(f'{algo}_train')
        feat['train'] = torch.FloatTensor(features_train)
        del features, features_train
    else:
        feat['train'] = torch.FloatTensor(features[idx['train']])
        del features
    gc.collect()

    print('train head ', idx['train'][:5])
    print(feat['train'][:5, :])
    # print('test head ', idx['test'][:5])
    # print(feat['test'][:5, :])
    # print(labels.size(), labels)
    print(f"n={n}, m={m}, F_t={feat['train'].size()}")
    print(f"n_train={idx['train'].size()}, n_val={idx['val'].size()}, n_test={idx['test'].size()}")
    return feat, labels, idx
