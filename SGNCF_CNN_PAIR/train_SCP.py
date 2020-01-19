import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datasets_object.yoochoose_ import YoochooseData
from datasets_object.last_fm import LastfmData
from datasets_object.diginetica import DigineticaData
from sgncf import sgncf1_cnn, sgncf2_cnn
from evaluation import Evaluation
from session_dataloader import SessionDataloader
from torch.utils.data import DataLoader
from utils import spmx_1_normalize, spmx2torch_sparse_tensor, spmx_sym_normalize, get_norm, plt_evalution, plt_norm


import time


def train(dataset, alpha, A_type, normalize_type, model_pretrained_params,
          model_type, batch_size, test_batch_size, negative_nums, item_emb_dim, hid_dim1, hid_dim2,
          hid_dim3, lr_emb, l2_emb, lr_gcn, l2_gcn, lr_cnn, l2_cnn,
          epochs, params_file_name):
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

    # get cpu sparse tensor: session adj
    SI = data_obj.get_session_adj(data_obj.d, alpha=alpha)


    # load model pretrained params
    if model_pretrained_params == 'True':
        print('----------------------------------')
        if dataset == 'LastFM':
            # use LastFM params
            print('Use LastFM model pretraned params: ' + params_file_name + '.pkl')
            pretrained_state_dict = torch.load('./lastfm_pretrained_params/'+params_file_name+'.pkl')
        elif dataset == 'Diginetica':
            # use Diginetica params
            print('Use Diginetica model pretraned params: ' + params_file_name + '.pkl')
            pretrained_state_dict = torch.load('./dig_pretrained_params/'+params_file_name+'.pkl')
        else:
            # use yoochoose1_64 params
            print('Use yoochoose1_64 model pretraned params: ' + params_file_name + '.pkl')
            pretrained_state_dict = torch.load('./yoo1_64_pretrained_params/'+params_file_name+'.pkl')
        print('----------------------------------')
    else:
        pretrained_state_dict = None


    # transform all tensor to cuda
    A = A.to(device)
    labels = labels.to(device)
    SI = SI.to(device)

    # define the evalution object
    evalution5 = Evaluation(k=5)
    evalution10 = Evaluation(k=10)
    evalution15 = Evaluation(k=15)
    evalution20 = Evaluation(k=20)

    # define yoochoose data object
    trainloader = SessionDataloader(train_size=data_obj.train_size,
                                    test_size=data_obj.test_size,
                                    item_size=data_obj.item_size,
                                    labels=labels,
                                    batch_size=batch_size,
                                    train=True,
                                    negative_nums=negative_nums,
                                    shuffle=True)
    testloader = SessionDataloader(train_size=data_obj.train_size,
                                    test_size=data_obj.test_size,
                                    item_size=data_obj.item_size,
                                    labels=labels,
                                    batch_size=test_batch_size*data_obj.item_size,
                                    train=False,
                                    negative_nums=negative_nums,
                                    shuffle=False)

    # define model, then transform to cuda
    if model_type == 'sgncf1_cnn':
        # use sgncf1_cnn model:
        model = sgncf1_cnn(dataset_nums=data_obj.train_size+data_obj.test_size,
                           item_nums=data_obj.item_size,
                           item_emb_dim=item_emb_dim,
                           hid_dim1=hid_dim1)
    else:
        # use sgncf2_cnn model:
        model = sgncf2_cnn(dataset_nums=data_obj.train_size+data_obj.test_size,
                           item_nums=data_obj.item_size,
                           item_emb_dim=item_emb_dim,
                           hid_dim1=hid_dim1,
                           hid_dim2=hid_dim2)
    model.to(device)

    # update model_state_dict
    if pretrained_state_dict is not None:
        model_state_dict = model.state_dict()
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)

    # define loss and optim
    criterion = nn.BCEWithLogitsLoss()
    if model_type == 'sgncf1_cnn':
        # use sgncf1 model parameters:
        optim_emb = optim.Adagrad([{'params': model.item_emb.parameters()}], lr=lr_emb, weight_decay=l2_emb)
        optim_gcn = optim.Adam([{'params': model.gconv1.parameters()}], lr=lr_gcn, weight_decay=l2_gcn)
        optim_cnn = optim.Adam([{'params': model.cnn_1d.parameters()},
                                {'params': model.fc.parameters()}], lr=lr_cnn, weight_decay=l2_cnn)
    else:
        # use sgncf2 model parameters:
        optim_emb = optim.Adagrad([{'params': model.item_emb.parameters()}], lr=lr_emb, weight_decay=l2_emb)
        optim_gcn = optim.Adam([{'params': model.gconv1.parameters()},
                                {'params': model.gconv2.parameters()}], lr=lr_gcn, weight_decay=l2_gcn)
        optim_cnn = optim.Adam([{'params': model.cnn_1d.parameters()},
                                {'params': model.fc.parameters()}], lr=lr_cnn, weight_decay=l2_cnn)

    # figure recall mrr norm
    fig_recalls = []
    fig_mrrs = []
    fig_emb_norms = []
    fig_gcn_norms = []
    fig_cnn_norms = []
    fig_epochs = []

    # train epochs
    for epoch in range(epochs):
        # model training
        start = time.time()

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
            optim_cnn.zero_grad()

            # batch inputs
            batch_sidxes, batch_iidxes, batch_labels = data[:, 0].long().to(device), data[:, 1].long().to(device), data[:, 2].float().to(device)

            # predicting
            outs = model(batch_sidxes, batch_iidxes, A, SI)

            # loss
            loss = criterion(outs, batch_labels)

            # backward
            loss.backward()

            # optim step
            optim_emb.step()
            optim_gcn.step()
            optim_cnn.step()

            # losses
            losses.append(loss.item())

            # print loss, recall, mrr
            if i % 20 == 19:
                print('[{0: 2d}, {1:5d}, {2: 7d}], loss:{3:.4f}'.format(epoch + 1,
                                                                        int(i*(batch_size/(negative_nums+1))),
                                                                        data_obj.train_size,
                                                                        np.mean(losses)))

        # print gcn_norm, emb_norm
        emb_norm = get_norm(model, 'emb')
        gcn_norm = get_norm(model, 'gcn')
        cnn_norm = get_norm(model, 'cnn')
        fig_emb_norms.append(emb_norm)
        fig_gcn_norms.append(gcn_norm)
        fig_cnn_norms.append(gcn_norm)
        print('[gcn_norm]:{0:.4f}  [emb_norm]:{1:.4f}  [cnn_norm]:{2:.4f}'.format(gcn_norm,
                                                                                  emb_norm,
                                                                                  cnn_norm))

        # epoch time
        print('[epoch time]:{0:.4f}'.format(time.time() - start))

        # model eval
        model.eval()
        with torch.no_grad():
            for j, d in enumerate(testloader):
                # test batch inputs
                b_sidxes, b_iidxes, b_labels = d[0][:, 0].long().to(device), d[0][:, 1].long().to(device), d[1].to(device)

                # predicting
                o = model(b_sidxes, b_iidxes, A, SI)
                o = o.view(-1, data_obj.item_size)

                # evalution, k=5, 10, 15, 20
                r['5'].append(evalution5.evaluate(o, b_labels)[0])
                r['10'].append(evalution10.evaluate(o, b_labels)[0])
                r['15'].append(evalution15.evaluate(o, b_labels)[0])
                r['20'].append(evalution20.evaluate(o, b_labels)[0])
                m['5'].append(evalution5.evaluate(o, b_labels)[1])
                m['10'].append(evalution10.evaluate(o, b_labels)[1])
                m['15'].append(evalution15.evaluate(o, b_labels)[1])
                m['20'].append(evalution20.evaluate(o, b_labels)[1])

                # print test inf
                # print('[{0: 2d}, {1: 5d}, {2: 7d}]'.format(epoch+1,
                #                                            j * test_batch_size,
                #                                            data_obj.test_size))

            # print test recall mrr
            print('[{0: 2d}]'.format(epoch + 1))
            print('[recall@5 ]:{0:.4f}  [mrr@5 ]:{1:.4f}'.format(np.sum(r['5'])/data_obj.test_size,
                                                                 np.sum(m['5'])/data_obj.test_size))
            print('[recall@10]:{0:.4f}  [mrr@10]:{1:.4f}'.format(np.sum(r['10'])/data_obj.test_size,
                                                                 np.sum(m['10'])/data_obj.test_size))
            print('[recall@15]:{0:.4f}  [mrr@15]:{1:.4f}'.format(np.sum(r['15'])/data_obj.test_size,
                                                                 np.sum(m['15'])/data_obj.test_size))
            print('[recall@20]:{0:.4f}  [mrr@20]:{1:.4f}'.format(np.sum(r['20'])/data_obj.test_size,
                                                                 np.sum(m['20'])/data_obj.test_size))

            # plt recall and mrr and norm
            fig_epochs.append(epoch)
            fig_recalls.append(np.sum(r['20'])/data_obj.test_size)
            fig_mrrs.append(np.sum(m['20'])/data_obj.test_size)
            plt_evalution(fig_epochs, fig_recalls, fig_mrrs, k=20, alpha=alpha, lr_emb=lr_emb,
                          l2_emb=l2_emb, lr_gcn=lr_gcn, l2_gcn=l2_gcn, model_type=model_type,
                          lr_cnn=lr_cnn, l2_cnn=l2_cnn)
            plt_norm(fig_epochs, fig_emb_norms, fig_gcn_norms, fig_cnn_norms, alpha=alpha, lr_emb=lr_emb,
                     l2_emb=l2_emb, lr_gcn=lr_gcn, l2_gcn=l2_gcn, model_type=model_type,
                     lr_cnn=lr_cnn, l2_cnn=l2_cnn)