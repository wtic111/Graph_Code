import torch
from torch.utils.data import Dataset, DataLoader

class SessionDataset(Dataset):
    def __init__(self, train_size ,test_size, train, labels):
        self.train_size = train_size
        self.test_size = test_size
        self.train = train
        self.labels = labels

    def __len__(self):
        length = self.train_size if self.train else self.test_size

        return length

    def __getitem__(self, idx):
        idx = idx if self.train else idx + self.train_size
        label = self.labels[idx]

        return idx, label