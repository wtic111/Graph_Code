import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from baseline_data.yoochoose_ import YoochooseData
from baseline_data.last_fm import LastfmData
from baseline_data.diginetica import DigineticaData
from ngcf import ngcf1_session_hot_items, ngcf2_session_hot_items, ngcf3_session_hot_items, ngcf2_session_last_item
from evaluation import Evaluation
from session_dataset import SessionDataset
from torch.utils.data import DataLoader
from utils import spmx_1_normalize, spmx2torch_sparse_tensor, spmx_sym_normalize, get_norm, plt_evalution, plt_norm


import time


def train(dataset, alpha, A_type, normalize_type, session_type, pretrained_item_emb, model_type, batch_size, shuffle,
          item_emb_dim, hid_dim1, hid_dim2, hid_dim3, lr_emb, lr_gcn, l2_emb, l2_gcn, epochs):
    # init
    if dataset == 'LastFM':
        # use LastFM dataset
        data_obj = LastfmData()
    elif dataset == 'Diginetica':
        # use Diginetica dataset
        data_obj = DigineticaData()
    else:
        # use yoochoose1_64 dataset
        data_obj = YoochooseData(dataset=dataset)

    # gpu device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init A
    # A: type=scipy.sparse
    A = data_obj.get_decay_adj(data_obj.d, tail=None, alpha=alpha) if A_type == 'decay' else data_obj.get_gcn_adj(data_obj.d)
    # normalize the adj, type = 'ramdom_walk'(row 1) or type = 'symmetric'
    if normalize_type == 'random_walk':
        print('----------------------------------')
        print('Normalize_type is random_walk:')
        A = spmx_1_normalize(A)
        print('----------------------------------')
    else:
        print('----------------------------------')
        print('Normalize_type is symmetric:')
        A = spmx_sym_normalize(A)
        print('----------------------------------')
    # transform the adj to a sparse cpu tensor
    A = spmx2torch_sparse_tensor(A)

    # get cpu tensor: labels
    labels = data_obj.get_labels(data_obj.d)

    # get cpu tensor: item_idxes
    _, _, item_idxes = data_obj.get_indexes()

    if session_type == 'session_hot_items':
        # get cpu sparse tensor: session adj
        session_adj = data_obj.get_session_adj(data_obj.d, alpha=alpha)
    else:
        # if not use session adj, then session_adj = None
        session_adj = None

    if session_type == 'session_last_item':
        # get cpu LongTensor: session_last_item
        session_last_item = data_obj.get_session_last_item(data_obj.d).long()
    else:
        # if not use session_last_item, then session_last_item = None
        session_last_item = None

    # get pretrained_item_emb
    if pretrained_item_emb == 'True' and alpha != 0.0:
        print('----------------------------------')
        if dataset == 'yoochoose1_64':
            print('Use yoochoose1_64 pretrained item embedding: '+'pretrained_emb'+str(alpha)+'.pkl')
            pretrained_item_emb = torch.load('./yoo1_64_pretrained_item_emb/pretrained_emb'+str(alpha)+'.pkl')['item_emb.weight']
        elif dataset == 'yoochoose1_8':
            print('Use yoochoose1_8 pretrained item embedding: '+'pretrained_emb'+str(alpha)+'.pkl')
            pretrained_item_emb = torch.load('./yoo1_8_pretrained_item_emb/pretrained_emb'+str(alpha)+'.pkl')['item_emb.weight']
        elif dataset == 'LastFM':
            print('Use LastFM pretrained item embedding: ' + 'pretrained_emb' + str(alpha) + '.pkl')
            pretrained_item_emb = torch.load('./lastfm_pretrained_item_emb/pretrained_emb' + str(alpha) + '.pkl')['item_emb.weight']
        else:
            print('Use Diginetica pretrained item embedding: ' + 'pretrained_emb' + str(alpha) + '.pkl')
            pretrained_item_emb = torch.load('./dig_pretrained_item_emb/pretrained_emb' + str(alpha) + '.pkl')['item_emb.weight']
        print('----------------------------------')
    else:
        print('----------------------------------')
        print('Not use pretrained item embedding:')
        pretrained_item_emb = None
        print('----------------------------------')

    # get cpu LongTensor: item_emb_idxes
    item_emb_idxes = torch.arange(data_obj.item_size).long()

    # transform all tensor to cuda
    A = A.to(device)
    labels = labels.to(device)
    item_idxes = item_idxes.to(device)
    item_emb_idxes = item_emb_idxes.to(device)
    if session_last_item is not None:
        session_last_item = session_last_item.to(device)
    if session_adj is not None:
        session_adj = session_adj.to(device)

    # define the evalution object
    evalution5 = Evaluation(k=5)
    evalution10 = Evaluation(k=10)
    evalution15 = Evaluation(k=15)
    evalution20 = Evaluation(k=20)

    # define yoochoose data object
    trainset = SessionDataset(train_size=data_obj.train_size,
                              test_size=data_obj.test_size,
                              train=True,
                              labels=labels)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    testset = SessionDataset(train_size=data_obj.train_size,
                             test_size=data_obj.test_size,
                             train=False,
                             labels=labels)
    testloader = DataLoader(dataset=testset,
                            batch_size=batch_size,
                            shuffle=False)

    # define model, then transform to cuda
    if model_type == 'ngcf1_session_hot_items':
        # use ngcf1_session_hot_items model:
        model = ngcf1_session_hot_items(item_nums=data_obj.item_size,
                                        item_emb_dim=item_emb_dim,
                                        hid_dim1=hid_dim1,
                                        pretrained_item_emb=pretrained_item_emb)
    elif model_type == 'ngcf2_session_hot_items':
        # use ngcf2_session_hot_items model:
        model = ngcf2_session_hot_items(item_nums=data_obj.item_size,
                                        item_emb_dim=item_emb_dim,
                                        hid_dim1=hid_dim1,
                                        hid_dim2=hid_dim2,
                                        pretrained_item_emb=pretrained_item_emb)
    elif model_type == 'ngcf3_session_hot_items':
        # use ngcf3_session_hot_items model:
        model = ngcf3_session_hot_items(item_nums=data_obj.item_size,
                                        item_emb_dim=item_emb_dim,
                                        hid_dim1=hid_dim1,
                                        hid_dim2=hid_dim2,
                                        hid_dim3=hid_dim3,
                                        pretrained_item_emb=pretrained_item_emb)
    else:
        # use ngcf2_session_last_item model:
        model = ngcf2_session_last_item(item_nums=data_obj.item_size,
                                        item_emb_dim=item_emb_dim,
                                        hid_dim1=hid_dim1,
                                        hid_dim2=hid_dim2,
                                        pretrained_item_emb=pretrained_item_emb)
    model.to(device)

    # define loss and optim
    criterion = nn.CrossEntropyLoss()
    if model_type == 'ngcf1_session_hot_items':
        # use ngcf1_session_hot_items model parameters:
        optim_emb = optim.Adagrad([{'params': model.item_emb.parameters()}], lr=lr_emb, weight_decay=l2_emb)
        optim_gcn = optim.Adam([{'params': model.gconv1.parameters()}], lr=lr_gcn, weight_decay=l2_gcn)

    elif model_type == 'ngcf2_session_hot_items':
        # use ngcf2_session_hot_items model parameters:
        optim_emb = optim.Adagrad([{'params': model.item_emb.parameters()}], lr=lr_emb, weight_decay=l2_emb)
        optim_gcn = optim.Adam([{'params': model.gconv1.parameters()},
                                {'params': model.gconv2.parameters()}], lr=lr_gcn, weight_decay=l2_gcn)

    elif model_type == 'ngcf3_session_hot_items':
        # use ngcf3_session_hot_items model parameters:
        optim_emb = optim.Adagrad([{'params': model.item_emb.parameters()}], lr=lr_emb, weight_decay=l2_emb)
        optim_gcn = optim.Adam([{'params': model.gconv1.parameters()},
                                {'params': model.gconv2.parameters()},
                                {'params': model.gconv3.parameters()}], lr=lr_gcn, weight_decay=l2_gcn)

    else:
        # use ngcf2_session_last_item model parameters:
        optim_emb = optim.Adagrad([{'params': model.item_emb.parameters()}], lr=lr_emb, weight_decay=l2_emb)
        optim_gcn = optim.Adam([{'params': model.gconv1.parameters()},
                                {'params': model.gconv2.parameters()}], lr=lr_gcn, weight_decay=l2_gcn)

    # figure recall mrr norm
    fig_recalls = []
    fig_mrrs = []
    fig_emb_norms = []
    fig_gcn_norms = []
    fig_epochs = []

    # train epochs
    for epoch in range(epochs):
        # model training
        start = time.time()

        # train evalution dict
        recall = {'5': [], '10': [], '15': [], '20': []}
        mrr = {'5': [], '10': [], '15': [], '20': []}

        # test evalution dict
        r = {'5': [], '10': [], '15': [], '20': []}
        m = {'5': [], '10': [], '15': [], '20': []}

        # loss list
        losses = []

        model.train()
        for i, data in enumerate(trainloader):
            # zero optim
            optim_emb.zero_grad()
            optim_gcn.zero_grad()

            # batch inputs
            batch_idxes, batch_labels = data[0].long().to(device), data[1].long().to(device)

            # predicting
            if model_type == 'ngcf1_session_hot_items':
                # use ngcf1_session_hot_items model to predict
                outs = model(batch_idxes, A, item_idxes, session_adj, item_emb_idxes)
            elif model_type == 'ngcf2_session_hot_items':
                # use ngcf2_session_hot_items model to predict
                outs = model(batch_idxes, A, item_idxes, session_adj, item_emb_idxes)
            elif model_type == 'ngcf3_session_hot_items':
                # use ngcf3_session_hot_items model to predict
                outs = model(batch_idxes, A, item_idxes, session_adj, item_emb_idxes)
            else:
                # use ngcf2_session_last_item model to predict
                outs = model(batch_idxes, A, item_idxes, session_last_item, item_emb_idxes)

            # loss
            loss = criterion(outs, batch_labels)

            # backward
            loss.backward()

            # optim step
            optim_emb.step()
            optim_gcn.step()

            # evalution, k=5, 10, 15, 20
            recall['5'].append(evalution5.evaluate(outs, batch_labels)[0])
            recall['10'].append(evalution10.evaluate(outs, batch_labels)[0])
            recall['15'].append(evalution15.evaluate(outs, batch_labels)[0])
            recall['20'].append(evalution20.evaluate(outs, batch_labels)[0])
            mrr['5'].append(evalution5.evaluate(outs, batch_labels)[1])
            mrr['10'].append(evalution10.evaluate(outs, batch_labels)[1])
            mrr['15'].append(evalution15.evaluate(outs, batch_labels)[1])
            mrr['20'].append(evalution20.evaluate(outs, batch_labels)[1])

            # losses
            losses.append(loss.item())

            # print loss, recall, mrr
            if i % 50 == 49:
                print('[{0: 2d}, {1:5d}]  loss:{2:.4f}'.format(epoch + 1,
                                                               i + 1,
                                                               np.mean(losses)))
                print('[recall@5 ]:{0:.4f}  [mrr@5 ]:{1:.4f}'.format(np.mean(recall['5']),
                                                                     np.mean(mrr['5'])))
                print('[recall@10]:{0:.4f}  [mrr@10]:{1:.4f}'.format(np.mean(recall['10']),
                                                                     np.mean(mrr['10'])))
                print('[recall@15]:{0:.4f}  [mrr@15]:{1:.4f}'.format(np.mean(recall['15']),
                                                                     np.mean(mrr['15'])))
                print('[recall@20]:{0:.4f}  [mrr@20]:{1:.4f}'.format(np.mean(recall['20']),
                                                                     np.mean(mrr['20'])))

        # print gcn_norm, emb_norm
        emb_norm = get_norm(model, 'emb')
        gcn_norm = get_norm(model, 'gcn')
        fig_emb_norms.append(emb_norm)
        fig_gcn_norms.append(gcn_norm)
        print('[gcn_norm]:{0:.4f}  [emb_norm]:{1:.4f}'.format(gcn_norm,
                                                          emb_norm))

        # epoch time
        print('[epoch time]:{0:.4f}'.format(time.time() - start))

        # save model
        if epoch % 10 == 9:
            torch.save(model.state_dict(), 'params' + model_type+'-Alpha'+str(alpha)+'_'+'_lr_emb'+str(lr_emb)+'_l2_emb'+str(l2_emb)+'_lr_gcn'+str(lr_gcn)+'_l2_gcn'+str(l2_gcn) + '.pkl')

        # model eval
        model.eval()
        with torch.no_grad():
            for j, d in enumerate(testloader):
                # test batch inputs
                b_idxes, b_labels = d[0].long().to(device), d[1].long().to(device)

                # predicting
                if model_type == 'ngcf1_session_hot_items':
                    # use ngcf1_session_hot_items model to predict
                    o = model(b_idxes, A, item_idxes, session_adj, item_emb_idxes)
                elif model_type == 'ngcf2_session_hot_items':
                    # use ngcf2_session_hot_items model to predict
                    o = model(b_idxes, A, item_idxes, session_adj, item_emb_idxes)
                elif model_type == 'ngcf3_session_hot_items':
                    # use ngcf3_session_hot_items model to predict
                    o = model(b_idxes, A, item_idxes, session_adj, item_emb_idxes)
                else:
                    # use ngcf2_session_last_item model to predict
                    o = model(b_idxes, A, item_idxes, session_last_item, item_emb_idxes)

                # evalution, k=5, 10, 15, 20
                r['5'].append(evalution5.evaluate(o, b_labels)[0])
                r['10'].append(evalution10.evaluate(o, b_labels)[0])
                r['15'].append(evalution15.evaluate(o, b_labels)[0])
                r['20'].append(evalution20.evaluate(o, b_labels)[0])
                m['5'].append(evalution5.evaluate(o, b_labels)[1])
                m['10'].append(evalution10.evaluate(o, b_labels)[1])
                m['15'].append(evalution15.evaluate(o, b_labels)[1])
                m['20'].append(evalution20.evaluate(o, b_labels)[1])

            # print test recall mrr
            print('[{0: 2d}]'.format(epoch + 1))
            print('[recall@5 ]:{0:.4f}  [mrr@5 ]:{1:.4f}'.format(np.mean(r['5']),
                                                                 np.mean(m['5'])))
            print('[recall@10]:{0:.4f}  [mrr@10]:{1:.4f}'.format(np.mean(r['10']),
                                                                 np.mean(m['10'])))
            print('[recall@15]:{0:.4f}  [mrr@15]:{1:.4f}'.format(np.mean(r['15']),
                                                                 np.mean(m['15'])))
            print('[recall@20]:{0:.4f}  [mrr@20]:{1:.4f}'.format(np.mean(r['20']),
                                                                 np.mean(m['20'])))

            # plt recall and mrr and norm
            fig_epochs.append(epoch)
            fig_recalls.append(np.mean(r['20']))
            fig_mrrs.append(np.mean(m['20']))
            plt_evalution(fig_epochs, fig_recalls, fig_mrrs, k=20, alpha=alpha, lr_emb=lr_emb,
                          l2_emb=l2_emb, lr_gcn=lr_gcn, l2_gcn=l2_gcn, model_type=model_type)
            plt_norm(fig_epochs, fig_emb_norms, fig_gcn_norms, alpha=alpha, lr_emb=lr_emb,
                     l2_emb=l2_emb, lr_gcn=lr_gcn, l2_gcn=l2_gcn, model_type=model_type)