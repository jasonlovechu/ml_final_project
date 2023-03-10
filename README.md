# ML Final Project
## Introduction
This repo is about my final project of ml. We are asked to join a real-world machine learning competition on kaggle.  
class: ml, nycu, 2022fall  
series: Tabular Playground Series - Aug 2022
series link:https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview

## Environment and Dependency
* System: macOS Monterey
* GPU: **Apple M1 Pro**
* python: 3.9.7
* device: **mps**
* requirement.txt

## Model architecture
**models.py**  
3 level nn

## Step to reproduce the result
After run this code, it will generate weight.pt. The grid search function is for hyper-parameters searching and auto submit by kaggle api.
```
train model: python 109550057_Final_train.py
```
Load state from weight.pt and test, generate 109550057_submission.csv.
```
test model: python 109550057_Final_inference.py
```
## Result and model weight link
model weight link(only by mps): https://drive.google.com/drive/folders/1yz9xRz01KTCUbSCzmyC0978c18H6MaKp?usp=sharing    
private score: 0.59133(may differ due to random issue)

## Discussion
* I test two kinds of batch size, which are 64 and 96 with lr ranges from 1e-5 to 1e-3. The accuracies of batch size 64 are higher in average.   
* accuracies decrease when lr increase in most cases.  
* The result turn out to be different even with same hyperparameters. I tried to train three models with each hyperparameter to reduce the influence from randomness.   
