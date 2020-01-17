import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from utils import tensor_normalize

import math


class GCNconv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(GCNconv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # original init parameters
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

        # new init parameters
        self.weight.data.normal_(0, 0.01)
        if self.bias is not None:
            self.bias.data.normal_(0, 0.01)

    def forward(self, x, adj):
        # print(self.weight, self.bias)
        if x.is_sparse:
            x = torch.spmm(x, self.weight)
        else:
            x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x)

        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class sgncf1(nn.Module):
    def __init__(self, dataset_nums, item_nums, item_emb_dim, hid_dim1):
        super(sgncf1, self).__init__()
        print('----------------------------------')
        print('Use sgncf1 Model:')
        # dataset_nums = train_size + test_size
        self.dataset_nums = dataset_nums

        self.item_emb = nn.Embedding(item_nums, item_emb_dim)

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)

        print('----------------------------------')

    def forward(self, batch_sidxes, batch_iidxes, A, SI):
        # get session emb and item emb
        x_item = self.item_emb.weight
        x_session = torch.spmm(SI, x_item)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 2 gcn layer
        h1 = self.gconv1(x, A)

        # transform batch_iidxes, batch_iidxes += train_size + test_size
        batch_iidxes += self.dataset_nums

        # predict the scores
        # predict: (sidx, iidx) -> Vec_sidx * Vec_iidx
        out = torch.matmul(h1[batch_sidxes], h1[batch_iidxes].transpose(1, 0))

        return out


class sgncf2(nn.Module):
    def __init__(self, dataset_nums, item_nums, item_emb_dim, hid_dim1, hid_dim2):
        super(sgncf2, self).__init__()
        print('----------------------------------')
        print('Use sgncf2 Model:')
        # dataset_nums = train_size + test_size
        self.dataset_nums = dataset_nums

        # define item_emb layer
        self.item_emb = nn.Embedding(item_nums, item_emb_dim)

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)

        print('----------------------------------')

    def forward(self, batch_sidxes, batch_iidxes, A, SI):
        # get session emb and item emb
        x_item = self.item_emb.weight
        x_session = torch.spmm(SI, x_item)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 2 gcn layer
        h1 = F.relu(self.gconv1(x, A))
        h2 = self.gconv2(h1, A)

        # transform batch_iidxes, batch_iidxes += train_size + test_size
        batch_iidxes += self.dataset_nums

        # predict the scores
        # predict: (sidx, iidx) -> Vec_sidx * Vec_iidx
        out = h2[batch_sidxes] * h2[batch_iidxes]
        out = out.sum(dim=1)

        return out


