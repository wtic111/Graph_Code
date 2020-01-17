import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name')
parser.add_argument('--alpha', type=float, default=0.4, help='hot decay')
parser.add_argument('--A_type', type=str, default='decay', help='A type')
parser.add_argument('--normalize_type', type=str, default='random_walk', help='normalize_type')
parser.add_argument('--session_type', type=str, default='session_hot_items', help='session_type')
parser.add_argument('--pretrained_item_emb', type=str, default='True', help='pretrained_emb')
parser.add_argument('--model_type', type=str, default='ngcf2_session_hot_items', help='model_type')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--item_emb_dim', type=int, default=150, help='the dim of embeddings')
parser.add_argument('--hid_dim1', type=int, default=150, help='the dim of hidden1')
parser.add_argument('--hid_dim2', type=int, default=150, help='the dim of hidden2')
parser.add_argument('--hid_dim3', type=int, default=150, help='the dim of hidden3')
parser.add_argument('--lr_emb', type=float, default=0.05, help='learning rate')
parser.add_argument('--lr_gcn', type=float, default=0.05, help='learning rate')
parser.add_argument('--l2_emb', type=float, default=0.0, help='l2')
parser.add_argument('--l2_gcn', type=float, default=0.0, help='l2')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
args = parser.parse_args()
print(args)


def main():
    train(dataset=args.dataset,
          alpha=args.alpha,
          A_type=args.A_type,
          normalize_type=args.normalize_type,
          session_type=args.session_type,
          pretrained_item_emb=args.pretrained_item_emb,
          model_type=args.model_type,
          batch_size=args.batch_size,
          shuffle=True,
          item_emb_dim=args.item_emb_dim,
          hid_dim1=args.hid_dim1,
          hid_dim2=args.hid_dim2,
          hid_dim3=args.hid_dim3,
          lr_emb=args.lr_emb,
          lr_gcn=args.lr_gcn,
          l2_emb=args.l2_emb,
          l2_gcn=args.l2_gcn,
          epochs=args.epochs)



if __name__ == '__main__':
    main()
