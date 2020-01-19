import argparse
from train_SCP import train

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name')
parser.add_argument('--alpha', type=float, default=0.4, help='hot decay')
parser.add_argument('--A_type', type=str, default='decay', help='A type')
parser.add_argument('--normalize_type', type=str, default='random_walk', help='normalize_type')
parser.add_argument('--model_pretrained_params', type=str, default='True', help='model_pretrained_params')
parser.add_argument('--model_type', type=str, default='sgncf2', help='model_type')
parser.add_argument('--batch_size', type=int, default=4*8192, help='input batch size')
parser.add_argument('--test_batch_size', type=int, default=500, help='test_batch_size')
parser.add_argument('--negative_nums', type=int, default=3, help='negative_nums')
parser.add_argument('--item_emb_dim', type=int, default=150, help='item_emb_dim')
parser.add_argument('--hid_dim1', type=int, default=150, help='the dim of hidden1')
parser.add_argument('--hid_dim2', type=int, default=150, help='the dim of hidden2')
parser.add_argument('--hid_dim3', type=int, default=150, help='the dim of hidden3')
parser.add_argument('--lr_emb', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_gcn', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_cnn', type=float, default=0.01, help='learning rate')
parser.add_argument('--l2_emb', type=float, default=0.0, help='l2')
parser.add_argument('--l2_gcn', type=float, default=1e-6, help='l2')
parser.add_argument('--l2_cnn', type=float, default=1e-6, help='l2')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--params_file_name', type=str, default='params-Alpha0.4__lr_emb0.001_l2_emb0.0_lr_gcn0.001_l2_gcn1e-05', help='params_file_name')
args = parser.parse_args()
print(args)


def main():
    train(dataset=args.dataset,
          alpha=args.alpha,
          A_type=args.A_type,
          normalize_type=args.normalize_type,
          model_pretrained_params=args.model_pretrained_params,
          model_type=args.model_type,
          batch_size=args.batch_size,
          test_batch_size=args.test_batch_size,
          negative_nums=args.negative_nums,
          item_emb_dim=args.item_emb_dim,
          hid_dim1=args.hid_dim1,
          hid_dim2=args.hid_dim2,
          hid_dim3=args.hid_dim3,
          lr_emb=args.lr_emb,
          lr_gcn=args.lr_gcn,
          l2_emb=args.l2_emb,
          l2_gcn=args.l2_gcn,
          epochs=args.epochs,
          lr_cnn=args.lr_cnn,
          l2_cnn=args.l2_cnn,
          params_file_name=args.params_file_name)


if __name__ == '__main__':
    main()