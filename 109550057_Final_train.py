import numpy as np
import pandas as pd
import csv
import os
import random
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from models import MyNet, MyDataset
#from inference import test

def train(net, train_dataloader, valid_dataloader, criterion, optimizer, scheduler=None, epochs=10, device='cpu', checkpoint_epochs=10):
    print(f'Training for {epochs} epochs on {device}')
    best_loss = float("inf")
    #best_loss.to(device)
    for epoch in range(1,epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        
        net.train()  
        train_loss = torch.tensor(0., device=device)  
        for X, y in train_dataloader:
            X = X.float().to(device)
            y = y.float().to(device)
            preds = net(X)
            loss = criterion(torch.flatten(preds), y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size

        if valid_dataloader is not None:
            net.eval()  
            valid_loss = torch.tensor(0., device=device)
            with torch.no_grad():
                for X, y in valid_dataloader:
                    X = X.float().to(device)
                    y = y.float().to(device)
                    preds = net(X)
                    loss = criterion(torch.flatten(preds), y)
                    valid_loss += loss * valid_dataloader.batch_size

        if scheduler is not None: 
            scheduler.step()
        
        cur_loss = 0
        cur_loss = train_loss / len(train_dataloader.dataset)
        print(f'Training loss: {cur_loss:.2f}')
        
        if valid_dataloader is not None:
            cur_loss = valid_loss / len(valid_dataloader.dataset)
            print(f'Valid loss: {cur_loss:.2f}')
         
        print()

        #save model
        if epoch%checkpoint_epochs==0:  
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './checkpoint.pth.tar')
            if best_loss >= cur_loss:
                if os.path.isfile('weights.pt'):
                    os.remove('weights.pt')
                if os.path.isfile('model.pt'):
                    os.remove('model.pt')    
                best_loss = cur_loss
                torch.save(net.state_dict(), 'weights.pt')
                torch.save(net, 'model.pt')
    return net
 

def model_train(lr, weight_decay, batch_size):
    #model train function
    device = "mps"
    train_epochs = 50
    full_epochs = 50
    train_data = []
    val_data = []
    
    #data preprocessing
    drop_col = ['id', 'product_code', 'attribute_0', 'attribute_1']
    full_df = pd.read_csv("./train.csv")
    full_df.drop(drop_col, inplace=True, axis=1)
    for col in full_df.columns:
        full_df[col].fillna(value=full_df[col].mean(), inplace=True)
    full_data = full_df.values
    for i in range(full_data.shape[0]):
        if random.random() < 0.7:
            train_data.append(full_data[i])
        else:
            val_data.append(full_data[i])
    
    #load to dataloader
    full_ds = MyDataset(full_data) 
    full_dl = DataLoader(full_ds, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=False)

    train_ds = MyDataset(train_data) 
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=False)

    val_ds = MyDataset(val_data)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=False, num_workers=0, shuffle=False)

    model = MyNet().to(device) 
    criterion = nn.BCELoss()
    
    #train and validate
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = train(model, train_dl, val_dl, criterion, optimizer, None, train_epochs, device)

    #full train
    print("\nFull train start")
    print()
    model.load_state_dict(torch.load(f"weights.pt"))
    model = model.to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = train(model, full_dl, None, criterion, optimizer, None, full_epochs, device)


def grid_search(lr_base, weight_decay, batch_base):
    #grid search for good hyperparameters
    for lr_t in range(1, 121):
        for b_t in range(2, 4):
            lr = lr_base * lr_t
            batch_size = batch_base * b_t
            print(f"\nlr:{lr} batch:{batch_size}")
            model_train(lr=lr, weight_decay=weight_decay, batch_size=batch_size)
            test()
            #submit by kaggle api   
            status = subprocess.call(f"kaggle competitions submit -c tabular-playground-series-aug-2022 -f submission.csv -m 'dnn\nlr:{lr:.6f}\nbatch:{batch_size} \nepoch:50+50 \nweight_d:{weight_decay}'", shell=True)
            while status != 0:
                print("resubmit")
                status = subprocess.call(f"kaggle competitions submit -c tabular-playground-series-aug-2022 -f submission.csv -m 'dnn\nlr:{lr:.6f}\nbatch:{batch_size} \nepoch:50+50 \nweight_d:{weight_decay}'", shell=True)
       


if __name__ == '__main__':
    #hyper parameter
    lr = 6e-5
    weight_decay = 3e-4 
    batch_size = 64
    model_train(lr=lr, weight_decay=weight_decay, batch_size=batch_size)

       
