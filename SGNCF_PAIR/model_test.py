import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datasets_object.yoochoose_ import YoochooseData
from datasets_object.last_fm import LastfmData
from datasets_object.diginetica import DigineticaData
from sgncf import sgncf1, sgncf2
from evaluation import Evaluation
from session_dataloader import SessionDataloader
from torch.utils.data import DataLoader
from utils import spmx_1_normalize, spmx2torch_sparse_tensor, spmx_sym_normalize, get_norm, plt_evalution, plt_norm


import time


def model_test(dataset, alpha, A_type, normalize_type,
          model_type, negative_nums, item_emb_dim, hid_dim1, hid_dim2,
          model_pretrained_params, params_file_name):
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
    testloader = SessionDataloader(train_size=data_obj.train_size,
                                    test_size=data_obj.test_size,
                                    item_size=data_obj.item_size,
                                    labels=labels,
                                    batch_size=50*data_obj.item_size,
                                    train=False,
                                    negative_nums=negative_nums,
                                    shuffle=False)

    # define model, then transform to cuda
    if model_type == 'sgncf1':
        # use sgncf1_cnn model:
        model = sgncf1(dataset_nums=data_obj.train_size+data_obj.test_size,
                       item_nums=data_obj.item_size,
                       item_emb_dim=item_emb_dim,
                       hid_dim1=hid_dim1,
                       pretrained_item_emb=None)
    else:
        # use sgncf2_cnn model:
        model = sgncf2(dataset_nums=data_obj.train_size+data_obj.test_size,
                       item_nums=data_obj.item_size,
                       item_emb_dim=item_emb_dim,
                       hid_dim1=hid_dim1,
                       hid_dim2=hid_dim2,
                       pretrained_item_emb=None)
    model.to(device)

    # update model_state_dict
    if pretrained_state_dict is not None:
        model_state_dict = model.state_dict()
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)

    # test evalution dict
    r = {'5': [], '10': [], '15': [], '20': []}
    m = {'5': [], '10': [], '15': [], '20': []}

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
            print('[{0: 2d}, {1: 4d}]'.format(j*50,
                                              data_obj.test_size))

        # print test recall mrr
        print('[recall@5 ]:{0:.4f}  [mrr@5 ]:{1:.4f}'.format(np.sum(r['5']) / data_obj.test_size,
                                                             np.sum(m['5']) / data_obj.test_size))
        print('[recall@10]:{0:.4f}  [mrr@10]:{1:.4f}'.format(np.sum(r['10']) / data_obj.test_size,
                                                             np.sum(m['10']) / data_obj.test_size))
        print('[recall@15]:{0:.4f}  [mrr@15]:{1:.4f}'.format(np.sum(r['15']) / data_obj.test_size,
                                                             np.sum(m['15']) / data_obj.test_size))
        print('[recall@20]:{0:.4f}  [mrr@20]:{1:.4f}'.format(np.sum(r['20']) / data_obj.test_size,
                                                             np.sum(m['20']) / data_obj.test_size))



model_test(dataset='yoochoose1_64',
           alpha=0.4,
           A_type='decay',
           normalize_type='random_walk',
           model_type='sgncf2',
           negative_nums=3,
           item_emb_dim=150,
           hid_dim1=150,
           hid_dim2=150,
           model_pretrained_params='True',
           params_file_name='params-Alpha0.4__lr_emb0.001_l2_emb0.0_lr_gcn0.001_l2_gcn1e-05')