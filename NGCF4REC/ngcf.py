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


class GCNpredict(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(GCNpredict, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # for param in self.parameters():
        #     nn.init.xavier_uniform_(param)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, batch_x, adj):
        x = torch.spmm(adj, batch_x)
        x = torch.mm(x, self.weight)
        if self.bias is not None:
            return x + self.bias
        else:
            return x



class ngcf2(nn.Module):
    def __init__(self, item_nums, item_emb_dim, session_num, hid_dim1, hid_dim2,
                 pretrained_item_emb):
        super(ngcf2, self).__init__()
        print('----------------------------------')
        print('Use ngcf2 Model:')
        # define item_emb layer
        if pretrained_item_emb is not None:
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
        else:
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)

        # define session_emb layer
        self.session_emb = nn.Embedding.from_pretrained(torch.zeros(session_num, item_emb_dim), freeze=False)

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)
        print('----------------------------------')

    def forward(self, batch_idxes, A, item_idxes, session_emb_idxes, item_emb_idxes):
        # get session emb and item emb
        x_session = self.session_emb(session_emb_idxes)
        x_item = self.item_emb(item_emb_idxes)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 2 gcn layer
        h1 = F.relu(self.gconv1(x, A))
        print(h1[item_idxes])
        h2 = self.gconv2(h1, A)
        print(h2[batch_idxes])

        # predict the scores
        out = torch.matmul(h2[batch_idxes], h2[item_idxes].transpose(1, 0))

        return out

class ngcf3(nn.Module):
    def __init__(self, item_nums, item_emb_dim, hid_dim1, hid_dim2, hid_dim3,
                 pretrained_item_emb, F_normalize):
        super(ngcf3, self).__init__()
        # define item_emb layer
        if pretrained_item_emb is not None:
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
        else:
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)
        self.gconv3 = GCNconv(in_dim=hid_dim2, out_dim=hid_dim3)

        # init the flags
        self.F_normalize = F_normalize

    def forward(self, batch_idxes, A, session_adj, item_idxes):
        # get the session_embedding
        session_emb = torch.spmm(session_adj, self.item_emb.weight)

        # get x
        x = torch.cat((session_emb, self.item_emb.weight), dim=0)

        # 3 gcn layer
        h1 = F.relu(self.gconv1(x, A))
        if self.F_normalize == 'True':
            h1 = F.normalize(h1, dim=1)
        h2 = F.relu(self.gconv2(h1, A))
        if self.F_normalize == 'True':
            h2 = F.normalize(h2, dim=1)
        h3 = self.gconv3(h2, A)

        # predict the scores
        out = torch.matmul(h3[batch_idxes], h3[item_idxes].transpose(1, 0))

        return out

class ngcf2_fc(nn.Module):
    def __init__(self, item_nums, item_emb_dim, session_num, hid_dim1, hid_dim2,
                 pretrained_item_emb):
        super(ngcf2_fc, self).__init__()
        print('----------------------------------')
        print('Use ngcf2 fc Model:')
        # define item_emb layer
        if pretrained_item_emb is not None:
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
        else:
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)

        # define session_emb layer
        self.session_emb = nn.Embedding.from_pretrained(torch.zeros(session_num, item_emb_dim), freeze=False)

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)

        # define fc layer
        self.fc = nn.Linear(in_features=hid_dim2, out_features=item_nums)
        print('----------------------------------')

    def forward(self, batch_idxes, A, item_idxes, session_emb_idxes, item_emb_idxes):
        # get session emb and item emb
        x_session = self.session_emb(session_emb_idxes)
        x_item = self.item_emb(item_emb_idxes)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 2 gcn layer
        h1 = F.relu(self.gconv1(x, A))
        h2 = self.gconv2(h1, A)

        # predict the scores
        out = self.fc(h2[batch_idxes])

        return out

class ngcf2_gcnpre(nn.Module):
    def __init__(self, item_nums, item_emb_dim, session_num, hid_dim1, hid_dim2,
                 pretrained_item_emb):
        super(ngcf2_gcnpre, self).__init__()
        print('----------------------------------')
        print('Use ngcf2 GCNpredict Model:')
        # define item_emb layer
        if pretrained_item_emb is not None:
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
        else:
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)

        # define session_emb layer
        self.session_emb = nn.Embedding.from_pretrained(torch.zeros(session_num, item_emb_dim), freeze=False)

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)
        self.gpred = GCNpredict(in_dim=hid_dim2, out_dim=item_nums)

        print('----------------------------------')

    def forward(self, batch_idxes, A, item_idxes, session_emb_idxes, item_emb_idxes):
        # get session emb and item emb
        x_session = self.session_emb(session_emb_idxes)
        x_item = self.item_emb(item_emb_idxes)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 2 gcn layer
        h1 = F.relu(self.gconv1(x, A))
        h2 = self.gconv2(h1, A)

        # predict the scores
        out = self.gpred(h2[batch_idxes], A)

        return out

class ngcf2_session_last_item(nn.Module):
    def __init__(self, item_nums, item_emb_dim, hid_dim1, hid_dim2,
                 pretrained_item_emb):
        super(ngcf2_session_last_item, self).__init__()
        print('----------------------------------')
        print('Use ngcf2 session last item Model:')
        # define item_emb layer
        if pretrained_item_emb is not None:
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
        else:
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)

        print('----------------------------------')

    def forward(self, batch_idxes, A, item_idxes, session_last_item_idxes, item_emb_idxes):
        # get session emb and item emb
        x_session = self.item_emb(session_last_item_idxes)
        x_item = self.item_emb(item_emb_idxes)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 2 gcn layer
        h1 = F.relu(self.gconv1(x, A))
        h2 = self.gconv2(h1, A)

        # predict the scores
        out = torch.matmul(h2[batch_idxes], h2[item_idxes].transpose(1, 0))

        return out

class ngcf1_session_hot_items(nn.Module):
    def __init__(self, item_nums, item_emb_dim, hid_dim1,
                 pretrained_item_emb):
        super(ngcf1_session_hot_items, self).__init__()
        print('----------------------------------')
        print('Use ngcf1 session hot items Model:')
        print('----------------------------------')
        # define item_emb layer
        if pretrained_item_emb is not None:
            print('----------------------------------')
            print('Model use pretrained item embedding:')
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
            print('----------------------------------')
        else:
            print('----------------------------------')
            print('Model init the item embedding:')
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)
            print('----------------------------------')

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)

        print('----------------------------------')

    def forward(self, batch_idxes, A, item_idxes, session_adj, item_emb_idxes):
        # get session emb and item emb
        x_item = self.item_emb(item_emb_idxes)
        x_session = torch.spmm(session_adj, x_item)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 1 gcn layer
        h1 = self.gconv1(x, A)

        # predict the scores
        out = torch.matmul(h1[batch_idxes], h1[item_idxes].transpose(1, 0))

        return out

class ngcf2_session_hot_items(nn.Module):
    def __init__(self, item_nums, item_emb_dim, hid_dim1, hid_dim2,
                 pretrained_item_emb):
        super(ngcf2_session_hot_items, self).__init__()
        print('----------------------------------')
        print('Use ngcf2 session hot items Model:')
        print('----------------------------------')
        # define item_emb layer
        if pretrained_item_emb is not None:
            print('----------------------------------')
            print('Model use pretrained item embedding:')
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
            print('----------------------------------')
        else:
            print('----------------------------------')
            print('Model init the item embedding:')
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)
            print('----------------------------------')

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)

        print('----------------------------------')

    def forward(self, batch_idxes, A, item_idxes, session_adj, item_emb_idxes):
        # get session emb and item emb
        x_item = self.item_emb(item_emb_idxes)
        x_session = torch.spmm(session_adj, x_item)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 2 gcn layer
        x = F.relu(self.gconv1(x, A))
        x = self.gconv2(x, A)

        # predict the scores
        out = torch.matmul(x[batch_idxes], x[item_idxes].transpose(1, 0))

        return out

class ngcf3_session_hot_items(nn.Module):
    def __init__(self, item_nums, item_emb_dim, hid_dim1, hid_dim2,
                 hid_dim3, pretrained_item_emb):
        super(ngcf3_session_hot_items, self).__init__()
        print('----------------------------------')
        print('Use ngcf3 session hot items Model:')
        print('----------------------------------')

        # define item_emb layer
        if pretrained_item_emb is not None:
            print('----------------------------------')
            print('Model use pretrained item embedding:')
            self.item_emb = nn.Embedding.from_pretrained(pretrained_item_emb, freeze=False)
            print('----------------------------------')
        else:
            print('----------------------------------')
            print('Model init the item embedding:')
            self.item_emb = nn.Embedding(item_nums, item_emb_dim)
            print('----------------------------------')

        # define gcn layers
        self.gconv1 = GCNconv(in_dim=item_emb_dim, out_dim=hid_dim1)
        self.gconv2 = GCNconv(in_dim=hid_dim1, out_dim=hid_dim2)
        self.gconv3 = GCNconv(in_dim=hid_dim2, out_dim=hid_dim3)

        print('----------------------------------')

    def forward(self, batch_idxes, A, item_idxes, session_adj, item_emb_idxes):
        # get session emb and item emb
        x_item = self.item_emb(item_emb_idxes)
        x_session = torch.spmm(session_adj, x_item)

        # get x
        x = torch.cat((x_session, x_item), dim=0)

        # 3 gcn layer
        h1 = F.relu(self.gconv1(x, A))
        h2 = F.relu(self.gconv2(h1, A))
        h3 = self.gconv3(h2, A)

        # predict the scores
        out = torch.matmul(h3[batch_idxes], h3[item_idxes].transpose(1, 0))

        return out