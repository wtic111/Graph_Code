from datasets_object.yoochoose_ import YoochooseData
from session_dataloader import SessionDataloader


data_obj = YoochooseData(dataset='yoochoose1_64')
labels = data_obj.get_labels(data_obj.d)

trainloader = SessionDataloader(train_size=data_obj.train_size,
                                test_size=data_obj.test_size,
                                item_size=data_obj.item_size,
                                labels=labels,
                                batch_size=8192,
                                train=True,
                                negative_nums=3,
                                shuffle=False)

for data in trainloader:
    print(data)
