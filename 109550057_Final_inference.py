import numpy as np
import pandas as pd
import csv
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from models import MyNet, MyDataset 


def test():
    #hyper parameters
    device = 'mps'
    batch_size = 64
    drop_col = ['product_code', 'attribute_0', 'attribute_1']

    #data_processing
    test_df = pd.read_csv("./test.csv")
    test_df.drop(drop_col, inplace=True, axis=1)
    #test_df = test_df.fillna(0)
    for col in test_df.columns:
        test_df[col].fillna(value=test_df[col].mean(), inplace=True)
    test_data = test_df.values
    test_ds = MyDataset(test_data, test_mode=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False)

    if os.path.exists('submission.csv'):
        os.remove('submission.csv')
    file = open('submission.csv', 'w', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(["id", "failure"])

    #load weight
    weight_file = "weights.pt"

    model = MyNet()
    model.load_state_dict(torch.load(f"{weight_file}"))
    model.to(device)

    ans = list()
    model.eval()
    with torch.no_grad():
        for indexs, attrs in test_dl:
            attrs = attrs.float().to(device)
            preds = model(attrs)
            for i in range(len(indexs)):
                ans.append((int(indexs[i].item()), preds[i].item()))

    for index, l in ans:
        csv_writer.writerow([index, l])
    file.close()
   
if __name__ == '__main__':
    test()
