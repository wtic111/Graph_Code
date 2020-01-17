from train import train



train(dataset='LastFM',
      alpha=0.5,
      A_type='decay',
      normalize_type='random_walk',
      session_type='session_hot_items',
      pretrained_item_emb='False',
      model_type='ngcf2_session_hot_items',
      batch_size=1024,
      shuffle=True,
      item_emb_dim=150,
      hid_dim1=150,
      hid_dim2=150,
      hid_dim3=150,
      lr_emb=0.1,
      lr_gcn=0.01,
      l2_emb=0.0,
      l2_gcn=0.0,
      epochs=50)