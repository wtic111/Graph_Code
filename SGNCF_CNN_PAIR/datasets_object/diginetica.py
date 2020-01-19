import pickle
import os
import numpy as np
import torch
import scipy.sparse as sp
import time
from utils import spmx_1_normalize, spmx2torch_sparse_tensor, spmx_sym_normalize


# PATH = '/mnt/lizechao/10218_yanzhao/datasets/diginetica'
PATH = '/home/wuwu/datasets/diginetica'

class DigineticaData(object):
    def __init__(self):
        print('----------------------------------')
        print('Get the Diginetica data:')
        # root
        self.root = PATH
        self.d, self.train_size, self.test_size = self.get_all_d()
        self.item_size = len(self.get_item_uniques(self.d))
        print('----------------------------------')

    def get_all_d(self):
        train_d = self.get_train_d()
        test_d = self.get_test_d(idx=len(train_d))

        d = self.union_d(train_d, test_d)
        item_uniques = self.get_item_uniques(d)

        iid2iidx = {iid: iidx for iidx, iid in enumerate(item_uniques)}
        d = self.re_index_d(d, v2vidx=iid2iidx)

        return d, len(train_d), len(test_d)

    def get_train_d(self):
        # train path
        path_train = os.path.join(self.root, 'train.txt')

        # read tuple train
        with open(path_train, 'rb') as f:
            train_t = pickle.load(f)

        # transform tuple to my data structure
        train_d = {i: {'items': train_t[0][i], 'label': train_t[1][i]}for i in range(len(train_t[0]))}

        return train_d

    def get_test_d(self, idx):
        # test path
        path_test = os.path.join(self.root, 'test.txt')

        # read tuple test
        with open(path_test, 'rb') as f:
            test_t = pickle.load(f)

        # transform tuple to my data structure
        test_d = {i+idx: {'items': test_t[0][i], 'label': test_t[1][i]} for i in range(len(test_t[0]))}

        return test_d

    def union_d(self, d1, d2):
        assert isinstance(d1, dict)
        assert isinstance(d2, dict)
        d = d1.copy()
        d.update(d2)

        return d

    def re_index_d(self, d, v2vidx, k2kidx=None):
        assert isinstance(d, dict)
        if k2kidx is not None:
            re_index_d = {k2kidx[k]: {'items': [v2vidx[v] for v in d[k]['items']], 'label': v2vidx[d[k]['label']]}
                          for k in d.keys()}
        else:
            re_index_d = {k: {'items': [v2vidx[v] for v in d[k]['items']], 'label': v2vidx[d[k]['label']]}
                          for k in d.keys()}

        return re_index_d

    def get_item_uniques(self, d):
        assert isinstance(d, dict)
        seq_uniques = set.union(*[set(d[k]['items']) for k in d.keys()])
        label_uniques = set([d[k]['label'] for k in d.keys()])
        uniques = seq_uniques.union(label_uniques)

        return list(uniques)

    def get_decay_adj(self, d, tail, alpha):
        print('----------------------------------')
        print('Use decay adj, have self loop:')
        if tail is not None:
            tail_seq = {k: d[k]['items'][-tail:] for k in d.keys()}
        else:
            tail_seq = {k: d[k]['items'] for k in d.keys()}

        row = np.array([sidx for sidx in tail_seq.keys() for iidx in tail_seq[sidx][::-1]], dtype=np.int32)
        col = np.array([iidx for sidx in tail_seq.keys() for iidx in tail_seq[sidx][::-1]], dtype=np.int32)
        data = np.array([np.power(alpha, i) for sidx in tail_seq.keys() for i, iidx in enumerate(tail_seq[sidx][::-1])],
                        dtype=np.float32)
        mx = sp.csr_matrix((data, (row, col)), shape=(len(d), self.item_size))

        adj = sp.bmat([[None, mx], [mx.transpose(), None]]).tocsr()
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
        print('----------------------------------')

        return adj

    def get_session_adj(self, d, alpha):
        print('----------------------------------')
        print('Use sessoin adj cpu sparse tensor, row 1:')
        assert isinstance(alpha, float)
        row = np.array([sidx for sidx in d.keys() for iidx in d[sidx]['items'][::-1]], dtype=np.int32)
        col = np.array([iidx for sidx in d.keys() for iidx in d[sidx]['items'][::-1]], dtype=np.int32)
        data = np.array([np.power(alpha, i) for sidx in d.keys() for i, iidx in enumerate(d[sidx]['items'][::-1])],
                        dtype=np.float32)
        mx = sp.csr_matrix((data, (row, col)), shape=(len(d), self.item_size))
        mx = spmx_1_normalize(mx)
        print('----------------------------------')

        return spmx2torch_sparse_tensor(mx)

    def get_labels(self, d):
        assert isinstance(d, dict)
        labels = np.array([d[k]['label'] for k in d.keys()])

        return torch.from_numpy(labels).long()

    def get_indexes(self):
        train_idxes = torch.arange(self.train_size)
        test_idxes = torch.arange(self.test_size) + self.train_size
        item_idxes = torch.arange(self.item_size) + self.train_size + self.test_size

        return train_idxes, test_idxes, item_idxes