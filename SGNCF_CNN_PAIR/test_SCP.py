from train_SCP import train



train(dataset='yoochoose1_64',
      alpha=0.4,
      A_type='decay',
      normalize_type='random_walk',
      model_pretrained_params='True',
      model_type='sgncf2_cnn',
      batch_size=8192*4,
      test_batch_size=50,
      negative_nums=3,
      item_emb_dim=150,
      hid_dim1=150,
      hid_dim2=150,
      hid_dim3=150,
      lr_emb=0.1,
      lr_gcn=0.01,
      l2_emb=0.0,
      l2_gcn=0.0,
      epochs=50,
      lr_cnn=0.01,
      l2_cnn=0.1,
      params_file_name='params-Alpha0.4__lr_emb0.001_l2_emb0.0_lr_gcn0.001_l2_gcn1e-05')