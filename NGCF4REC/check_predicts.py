import torch
from baseline_data.yoochoose_ import YoochooseData
from ngcf import ngcf2
from evaluation import Evaluation
from session_dataset import YoochooseDataset
from torch.utils.data import DataLoader
from utils import spmx_1_normalize, spmx2torch_sparse_tensor, spmx_sym_normalize


def check(dataset, A_type, normalize_type, model_type, F_normalize, batch_size, shuffle, alpha, pretrained_item_emb,
          item_emb_dim, hid_dim1, hid_dim2, hid_dim3, lr_emb, lr_gcn, l2_emb, l2_gcn, epoch):
    # init
    yooch = YoochooseData(dataset=dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init A
    # A: type=scipy.sparse
    A = yooch.get_decay_adj(yooch.d, tail=None, alpha=alpha) if A_type == 'decay' else yooch.get_gcn_adj(yooch.d)
    # normalize the adj, type = 'ramdom_walk'(row 1) or type = 'symmetric'
    A = spmx_1_normalize(A) if normalize_type == 'ramdom_walk' else spmx_sym_normalize(A)
    # transform the adj to a sparse cpu tensor
    A = spmx2torch_sparse_tensor(A)

    # get cpu tensor: lables
    lables = yooch.get_lables(yooch.d)

    # get cpu sparse tensor: session_adj
    session_adj = yooch.get_session_adj(yooch.d, alpha=alpha)

    # get cpu tensor: item_idxes
    _, _, item_idxes = yooch.get_indexes()

    # if session_emb use the local then use the code below
    session_local_idxes = yooch.get_local(yooch.d)

    # get pretrained_item_emb
    pretrained_item_emb = torch.load('pretrained_emb'+str(int(alpha*10))+'.pkl')['item_emb.weight'] if pretrained_item_emb == 'True' else None


    # transform all tensor to cuda
    A = A.to(device)
    lables = lables.to(device)
    session_adj = session_adj.to(device)
    item_idxes = item_idxes.to(device)

    # if session_emb use the local then use the code below
    session_local_idxes = session_local_idxes.to(device)

    # define the evalution object
    evalution5 = Evaluation(k=5)
    evalution10 = Evaluation(k=10)
    evalution15 = Evaluation(k=15)
    evalution20 = Evaluation(k=20)
    max_epoch = {'5': -1, '10': -1, '15': -1, '20': -1}
    max_recall = {'5': -1, '10': -1, '15': -1, '20': -1}
    max_mrr = {'5': -1, '10': -1, '15': -1, '20': -1}

    # define yoochoose data object
    trainset = YoochooseDataset(train_size=yooch.train_size,
                                test_size=yooch.test_size,
                                train=True,
                                lables=lables)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    testset = YoochooseDataset(train_size=yooch.train_size,
                                test_size=yooch.test_size,
                                train=False,
                                lables=lables)
    testloader = DataLoader(dataset=testset,
                            batch_size=batch_size,
                            shuffle=False)

    model = ngcf2(item_nums=yooch.item_size,
                  item_emb_dim=item_emb_dim,
                  hid_dim1=hid_dim1,
                  hid_dim2=hid_dim2,
                  pretrained_item_emb=pretrained_item_emb,
                  F_normalize=F_normalize)
    model.to(device)
    model.load_state_dict(torch.load('/home/wuwu/下载/ngcf2alpha0.3params.pkl'))

    model.eval()
    outs = model(yooch.train_size+1, A, session_adj, item_idxes)

    a = 1


check(dataset='yoochoose1_64',
      A_type='decay',
      model_type='ngcf2',
      normalize_type='random_walk',
      F_normalize='False',
      batch_size=1024,
      shuffle=True,
      alpha=0.3,
      pretrained_item_emb='True',
      item_emb_dim=150,
      hid_dim1=150,
      hid_dim2=150,
      hid_dim3=150,
      lr_emb=0.05,
      lr_gcn=0.01,
      l2_emb=0,
      l2_gcn=0,
      epoch=25)




