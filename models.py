import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, test_mode=False):
        self.data = data
        self.test_mode = test_mode
    def __getitem__(self, index):
        if self.test_mode:
            data_id = self.data[index][0]
            attr = self.data[index][1:22]
            return data_id, attr
        else:
            attr = self.data[index][0:21]
            label = self.data[index][21]
            return attr, label
    def __len__(self):
        return len(self.data)
        
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(21, 21*100),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(21*100, 21*10),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(21*10, 1),
            nn.Sigmoid(),
        )
        def init_weights(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
        self.module.apply(init_weights)

    def forward(self, x):
        x = self.module(x)
        return x
