import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt


def get_session_len(session2item):
    session_len = np.zeros(len(session2item))
    for sid in session2item.keys():
        session_len[sid] = len(session2item[sid])

    return session_len


def build_session_spmx(dataset, samples=500):
    sessions = dataset.sessions
    item2session = dataset.item2session
    session2item = dataset.session2item
    session_len = get_session_len(session2item)
    row = list()
    col = list()
    sims = list()
    for sid in range(sessions):
        sarray = np.zeros(sessions)
        for iid in session2item[sid]:
            i_sids = np.array(list(item2session[iid]))
            sarray[i_sids] += 1
        sarray[sid] = 0
        norm = np.power(session_len[sid], 0.5) * np.power(session_len, 0.5)
        norm[norm == 0] = 1
        sarray = sarray / norm
        sid_edges = len(sarray[sarray != 0])
        if sid_edges >= samples:
            indices = np.argsort(sarray)[-1:-1 - samples:-1]
        else:
            indices = np.argsort(sarray)[-1:-1 - sid_edges:-1]
        #         print(len(indices))
        for ind in indices:
            row.append(sid)
            col.append(ind)
            sims.append(sarray[ind])
    #             print(sid, ind, sarray[ind])
    row = np.array(row, dtype=np.int32)
    col = np.array(col, dtype=np.int32)
    sims = np.array(sims, dtype=np.float32)
    session_spmx = sp.csr_matrix((sims, (row, col)), shape=(sessions, sessions))
    #     session_spmx = session_spmx + sp.eye(sessions, sessions)

    return session_spmx


def build_item2session_spmx(item2session, items, sessions):
    row = list()
    col = list()
    for iidx in item2session.keys():
        for sidx in item2session[iidx]:
            row.append(iidx)
            col.append(sidx)
    row = np.array(row, dtype=np.int32)
    col = np.array(col, dtype=np.int32)
    data = np.ones_like(row, dtype=np.float32)
    item2session_spmx = sp.csr_matrix((data, (row, col)), shape=(items, sessions))

    return item2session_spmx


def test(sample, session_spmx, item2session_spmx):
    for sid in sample.data.sessionIdx.unique():
        for sid2 in sample.data.sessionIdx.unique():
            if sid == sid2: continue
            # print(sid, sid2)
            intersection = sample.session2item[sid] & sample.session2item[sid2]
            if len(intersection) != 0:
                if session_spmx.toarray()[sid, sid2] == 0: print('false')
            else:
                if session_spmx.toarray()[sid, sid2] != 0: print('false')
    for sid in sample.session2item.keys():
        for iid in sample.session2item[sid]:
            # print(sid, iid)
            if item2session_spmx.toarray()[iid, sid] != 1: print('false')

    return


# def test_dataloader(dataset):
#     data = dataset.data
#     flag = True
#     supp = dict()
#     for inputid, lableid, _, _ in YoochooseDataloader(dataset, batch_size=10):
#         if len(np.flatnonzero(data.sessionIdx.values[inputid] - data.sessionIdx.values[lableid] != 0)) != 0:
#             flag = False
#             break
#     batch_supp = np.zeros(10, dtype=np.int32)
#     for inputid, lableid, batch_items, batch_itemadj in YoochooseDataloader(dataset, batch_size=10):
#         # print(batch_supp, batch_items)
#         for i, item in enumerate(batch_items):
#             print(i, inputid[i], batch_supp[i])
#             item_sessionIdxs = dataset.item2session[item]
#             item_sessionIdxs = list(item_sessionIdxs)
#             item_sessionIdxs = np.array(item_sessionIdxs)
#             if data.sessionIdx.values[inputid[i]] == data.sessionIdx.values[batch_supp[i]] or len(
#                     np.flatnonzero(batch_supp)) == 0:
#                 if supp.get(i) is None:
#                     supp.update({i: item_sessionIdxs})
#                 else:
#                     supp.update({i: np.union1d(supp[i], item_sessionIdxs)})
#             else:
#                 supp.update({i: item_sessionIdxs})
#             print(item, item_sessionIdxs, np.flatnonzero(batch_itemadj[i, :]), supp[i])
#             print(len(np.intersect1d(np.flatnonzero(batch_itemadj[i, :]), supp[i])) == len(supp[i]))
#             if len(np.intersect1d(np.flatnonzero(batch_itemadj[i, :]), supp[i])) == len(supp[i]) == False:
#                 flag = False
#                 break
#         if flag == False: break
#         batch_supp = inputid
#     if flag:
#         print('Dataloader is true')
#     else:
#         print('Dataloader is wrong')
#     return


def spmx_1_normalize(spmx):
    """
    reture a csr_matrix: dtype=np.float32
    """
    rowsum = np.array(spmx.sum(axis=1), dtype=np.float32)
    rowsum[rowsum == 0] = 1.0
    supp = np.power(rowsum, -1).flatten()
    sum_inverse_diags = sp.diags(supp).tocsr()
    spmx = sum_inverse_diags.dot(spmx)
    spmx = spmx.tocsr().astype(np.float32)

    return spmx


def spmx_sym_normalize(spmx):
    rowsum = np.array(spmx.sum(axis=1), dtype=np.float32)
    supp = np.power(rowsum, -0.5).flatten()
    d = sp.diags(supp).tocsr()
    spmx = d.dot(spmx)
    spmx = spmx.dot(d)
    spmx = spmx.tocsr().astype(np.float32)

    return spmx


def tensor_normalize(x):
    rowsum = torch.sum(x, dim=1)
    rowsum[rowsum == 0] = 1.0
    rowsum = rowsum.view(-1, 1)
    x = x / rowsum

    return x


def numpy_cos_normalize(x):
    row_sqrsum = np.sum(np.power(x, 2), axis=1)
    norm = np.power(row_sqrsum, 0.5).reshape(-1, 1)
    norm[norm == 0] = 1.0

    return x / norm


def numpy_1_normalize(x):
    row_sum = np.sum(x, axis=1).reshape(-1, 1)
    row_sum[row_sum == 0] = 1.0

    return x / row_sum


def spmx_cos_normalize(spmx):
    sqr_colsum = np.array((spmx.power(2)).sum(1)).flatten()
    sqr_colsum[sqr_colsum == 0] = 1.0
    norm = np.power(sqr_colsum, -0.5)
    sum_inverse_diags = sp.diags(norm).tocsr()
    spmx = sum_inverse_diags.dot(spmx)
    spmx = spmx.tocsr().astype(np.float32)

    return spmx


def get_spmx_rowsum_diag(x):
    rowsum = np.array(x.sum(1)).flatten()
    rowsum[rowsum == 0] = 1.0
    norm = np.power(rowsum, -1.0)
    rowsum_diag = sp.diags(norm).tocsr()

    return rowsum_diag


def spmx2torch_sparse_tensor(spmx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    spmx: csr_matrix
    i: dtype:torch.int64
    v: dtype:torch.float32
    """
    i = torch.LongTensor(np.vstack((spmx.tocoo().row, spmx.tocoo().col)))
    v = torch.FloatTensor(spmx.data)

    return torch.sparse.FloatTensor(i, v, torch.Size(spmx.shape))


def get_session_offsets(data):
    supp = data.groupby('sessionId').size()
    session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
    session_offsets[1:] = supp.cumsum()

    return session_offsets


def get_recall(out, lable, k=20):
    k_item = out.topk(k)[1]
    lable = lable.view(-1, 1).expand_as(k_item)
    hits = len((k_item == lable).nonzero())

    return hits / out.shape[0]


def get_norm(model, type):
    norm_2 = 0
    params = []
    if type == 'emb':
        for name, param in model.named_parameters():
            if name == 'item_emb.weight' or name == 'session_emb.weight':
                params.append(param)

    elif type == 'gcn':
        for name, param in model.named_parameters():
            if name != 'item_emb.weight' and name != 'session_emb.weight':
                params.append(param)

    for param in params:
        norm_2 += torch.norm(param, p=2)

    return norm_2

def plt_evalution(epochs, recalls, mrrs, k, alpha, lr_emb, l2_emb, lr_gcn, l2_gcn, model_type):
    fig = plt.figure()

    # choose a style
    plt.style.use('seaborn-whitegrid')

    # plot recalls and mrrs
    plt.plot(epochs, recalls, c='red', label='recall@'+str(k))
    plt.plot(epochs, mrrs, c='green', label='mrr@'+str(k))

    # plot the legend
    plt.legend(loc='best')

    # plot xlabel and ylabel
    plt.xlabel('epochs')
    plt.ylabel('rate')

    # plot title
    plt.title(model_type+'-Alpha'+str(alpha)+'_'+str(k)+'_lr_emb'+str(lr_emb)+'_l2_emb'+str(l2_emb)+'_lr_gcn'+str(lr_gcn)+'_l2_gcn'+str(l2_gcn))

    # save
    plt.savefig(model_type+'-Alpha'+str(alpha)+'_'+str(k)+'_lr_emb'+str(lr_emb)+'_l2_emb'+str(l2_emb)+'_lr_gcn'+str(lr_gcn)+'_l2_gcn'+str(l2_gcn)+'.png')

    # close
    plt.close(fig)

def plt_norm(epochs, emb_norms, gcn_norms, alpha, lr_emb, l2_emb, lr_gcn, l2_gcn, model_type):
    fig = plt.figure()

    # choose a style
    plt.style.use('seaborn-whitegrid')

    # plot norm_embs and norm_gcns
    plt.plot(epochs, emb_norms, c='red', label='emb_norm')
    plt.plot(epochs, gcn_norms, c='green', label='gcn_norm')

    # plot the legend
    plt.legend(loc='best')

    # plot xlabel and ylabel
    plt.xlabel('epochs')
    plt.ylabel('value')

    # plot title
    plt.title(model_type+'-Norm_Alpha' + str(alpha)+'_lr_emb'+str(lr_emb)+'_l2_emb'+str(l2_emb)+'_lr_gcn'+str(lr_gcn)+'_l2_gcn'+str(l2_gcn))

    # save
    plt.savefig(model_type+'-Norm_Alpha' + str(alpha) + '_lr_emb'+str(lr_emb)+'_l2_emb'+str(l2_emb)+'_lr_gcn'+str(lr_gcn)+'_l2_gcn'+str(l2_gcn)+'.png')

    # close
    plt.close(fig)