import pandas as pd
import numpy as np
import torch
import os
import pickle
import scipy.sparse as sp
import time
from utils import spmx_1_normalize, spmx2torch_sparse_tensor, spmx_sym_normalize




PATH = '/mnt/lizechao/10218_yanzhao/datasets/yoochoose_slices'
# PATH = '/home/wuwu/datasets/yoochoose_slices'




class YoochooseData(object):
    def __init__(self, dataset):
        print('----------------------------------')
        print('Get the yoochoose data:')
        self.start_time = time.time()
        self.root = os.path.join(PATH, dataset)
        self.d, self.train_size, self.test_size = self.get_all_d()
        self.item_size = len(self.get_item_uniques(self.d))
        print('----------------------------------')

    def get_all_d(self):
        train_df = self.get_df(train=True)
        test_df = self.get_df(train=False)
        train_d = self.get_dict(train_df, 0)
        test_d = self.get_dict(test_df, len(train_d))

        d = self.union_d(train_d, test_d)
        item_uniques = self.get_item_uniques(d)

        iid2iidx = {iid: iidx for iidx, iid in enumerate(item_uniques)}
        d = self.re_index_d(d, v2vidx=iid2iidx)

        return d, len(train_d), len(test_d)

    def get_df(self, train):
        path_df = os.path.join(self.root, 'train.csv') if train else os.path.join(self.root, 'test.csv')
        df = pd.read_csv(path_df,
                         header=None,
                         usecols=[0, 1, 2],
                         names=['sessionid', 'itemid', 'timestamp'],
                         dtype={'sessionid': np.int32, 'timestamp': str, 'itemid': np.int64})
        df.sort_values(by=['sessionid', 'timestamp'], inplace=True)

        return df

    def get_dict(self, df, d_idx0):
        assert isinstance(df, pd.DataFrame)
        d = dict()
        items = list()
        new_sidx = d_idx0
        for e in range(len(df) - 1):
            sid = df.sessionid.values[e]
            next_sid = df.sessionid.values[e + 1]
            if sid == next_sid:
                iid = df.itemid.values[e]
                next_iid = df.itemid.values[e + 1]
                items.append(iid)
                d[new_sidx] = {'items': items[:], 'label': next_iid}
                new_sidx += 1
            else:
                items.clear()

        return d

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

    def get_adj(self, d, tail):
        if tail is not None:
            tail_seq = {k: d[k]['items'][-tail:] for k in d.keys()}
        else:
            tail_seq = {k: d[k]['items'] for k in d.keys()}

        row = np.array([sidx for sidx in tail_seq.keys() for iidx in set(tail_seq[sidx])], dtype=np.int32)
        col = np.array([iidx for sidx in tail_seq.keys() for iidx in set(tail_seq[sidx])], dtype=np.int32)
        data = np.ones_like(row, dtype=np.float32)
        mx = sp.csr_matrix((data, (row, col)), shape=(len(d), self.item_size))

        adj = sp.bmat([[None, mx], [mx.transpose(), None]]).tocsr()
        adj = spmx_1_normalize(adj)

        return spmx2torch_sparse_tensor(adj)

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

    def get_gcn_adj(self, d):
        print('----------------------------------')
        print('Use gcn adj:')
        row = np.array([sidx for sidx in d.keys() for iidx in set(d[sidx]['items'])], dtype=np.int32)
        col = np.array([iidx for sidx in d.keys() for iidx in set(d[sidx]['items'])], dtype=np.int32)
        data = np.ones_like(row)
        mx = sp.csr_matrix((data, (row, col)), shape=(len(d), self.item_size))
        A = sp.bmat([[None, mx], [mx.transpose(), None]]).tocsr()
        A_hat = A + sp.eye(A.shape[0], dtype=np.float32)
        print('----------------------------------')

        return A_hat

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

    def get_session_last_item(self, d):
        print('----------------------------------')
        print('Use sessoin last item cpu LongTensor:')
        assert isinstance(d, dict)

        # get train and test session the last item, then transform to cpu tensor
        session_last_item = [d[k]['items'][-1] for k in d.keys()]
        session_last_item = torch.LongTensor(session_last_item)
        print('----------------------------------')

        return session_last_item

    def get_feature(self, d):
        row = np.array([sidx for sidx in d.keys() for iidx in d[sidx]['items']], dtype=np.int32)
        col = np.array([iidx for sidx in d.keys() for iidx in d[sidx]['items']], dtype=np.int32)
        data = np.ones_like(row)
        mx = sp.csr_matrix((data, (row, col)), shape=(len(d), self.item_size))
        A = sp.bmat([[None, mx], [mx.transpose(), None]]).tocsr()
        A_hat = A + sp.eye(A.shape[0], dtype=np.float32)
        feature = spmx_1_normalize(A_hat)

        return spmx2torch_sparse_tensor(feature)

    def get_labels(self, d):
        assert isinstance(d, dict)
        labels = np.array([d[k]['label'] for k in d.keys()])

        return torch.from_numpy(labels).long()

    def get_local(self, d):
        local_feature = [d[k]['items'][-1] for k in d.keys()]

        return torch.LongTensor(local_feature)

    def get_indexes(self):
        train_idxes = torch.arange(self.train_size)
        test_idxes = torch.arange(self.test_size) + self.train_size
        item_idxes = torch.arange(self.item_size) + self.train_size + self.test_size

        return train_idxes, test_idxes, item_idxes




