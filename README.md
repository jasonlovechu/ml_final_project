# ML Final Project
## Introduction
This repo is about my final project of ml. We are asked to join a real-world machine learning competition on kaggle.  
class: ml, nycu, 2022fall  
series: Tabular Playground Series - Aug 2022

## Environment and Dependency
* System: macOS Monterey
* GPU: Apple M1 Pro
* python: 3.9.7
* device: mps
* requirement.txt

## Step to reproduce the result
After run this code, it will generate weight.pt.
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
