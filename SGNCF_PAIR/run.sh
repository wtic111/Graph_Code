CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --lr_emb=1e-4 --l2_emb=0 --lr_gcn=1e-4 --l2_gcn=1e-6  >yoo_bs_4_8192_gcn2_1e-4_0_1e-4_1e-6.file 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --lr_emb=1e-5 --l2_emb=0 --lr_gcn=1e-4 --l2_gcn=1e-6  >yoo_bs_4_8192_gcn2_1e-5_0_1e-4_1e-6.file 2>&1 &

