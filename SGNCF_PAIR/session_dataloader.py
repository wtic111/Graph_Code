import numpy as np
import torch
import time


# [sidx, iid, 0 or 1]
# sidx: 0 - train_size
# iidx: 0 - test_size
# model(sidx, iidx, A1, A2):
#     s_emb = mm(A2, items_emb)
#     x = cat(s_emb, items_emb)
#     h = g(A1, x)
#     h = g(A1, h)
#     h[sidx]
#     iidx += train_size
#     h[iidx]
#     new_h = cat(h[sidx], h[iidx])
#     p = 1dcnn(new_h)
#
#     return p

class SessionDataloader(object):
    def __init__(self, train_size, test_size, item_size, labels, batch_size,
                 train, negative_nums=3, shuffle=False):
        # train: we need every session have 3 negative items
        # test: we need every session have all items in batch
        if train:
            assert batch_size % (negative_nums+1) == 0
        else:
            assert batch_size % item_size == 0

        # init
        self.train_size = train_size
        self.test_size = test_size
        self.item_size = item_size
        self.labels = labels
        self.batch_size = batch_size
        self.negative_nums = negative_nums
        self.train = train
        self.shuffle = shuffle

    def __iter__(self):
        # train data
        if self.train == True:
            # get train labels
            train_labels = self.labels[:self.train_size]

            # get train sidxes
            train_sidxes = np.arange(self.train_size)

            # shuffle
            if self.shuffle:
                np.random.shuffle(train_sidxes)

            # get true batch_size, session_groups = batch_size / 1 + negative_nums
            session_groups = int(self.batch_size / (1 + self.negative_nums))

            # yield batch data: (batch_sidxes, batch_iidxes, 0 or 1)
            # if negative_nums = 3: we need batch_size / (3+1) groups,
            # in every session_groups we have 1 positive item, and 3
            # negative item
            for i in range(self.train_size // session_groups + 1):
                # batch output
                batch_pairs = []

                # session_groups sidxes
                end_idx = min((i+1) * session_groups, self.train_size)
                sidxes = train_sidxes[np.arange(i * session_groups, end_idx)]

                # generate positive item and negative item
                for sidx in sidxes:
                    # generate positive item
                    positive_item = train_labels[sidx]
                    batch_pairs.append((sidx, positive_item, 1))

                    # generate negative item
                    for sample_time in range(self.negative_nums):
                        negative_item = np.random.randint(self.item_size)

                        # if negative_item = positive_item, sample item again
                        while negative_item == positive_item:
                            negative_item = np.random.randint(self.item_size)

                        batch_pairs.append((sidx, negative_item, 0))
                yield torch.LongTensor(batch_pairs)

        else:
            # test data
            # get train labels
            test_labels = self.labels[self.train_size:]

            # get true batch_size, session_groups = batch_size / item_size
            # guarantee (sidx, all iidx)
            session_groups = int(self.batch_size / self.item_size)

            # yield batch data: (batch_sidxes, batch_iidxes, 0 or 1)
            # if negative_nums = 3: we need batch_size / (3+1) groups,
            # in every session_groups we have 1 positive item, and 3
            # negative item
            for i in range(self.test_size // session_groups + 1):
                # batch output
                batch_pairs = []
                batch_labels = []

                # session_groups sidxes
                end_idx = min((i+1) * session_groups, self.test_size)
                sidxes = np.arange(i * session_groups, end_idx)

                # generate positive item and negative item
                for sidx in sidxes:
                    # generate positive item
                    positive_item = test_labels[sidx]
                    batch_labels.append(positive_item)

                    # generate (sidx, all iidxes)
                    pad_sidxes = np.full(shape=(self.item_size, 1), fill_value=sidx+self.train_size)
                    all_iidxes = np.arange(self.item_size).reshape(self.item_size, -1)
                    batch_pairs.append(np.hstack((pad_sidxes, all_iidxes)))

                yield torch.LongTensor(np.concatenate(batch_pairs)), torch.LongTensor(batch_labels)



