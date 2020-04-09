#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
import torch
import torch.nn as nn
import torch.nn.functional as nnF

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#%% Net: From ArXiv 1805.00794
class Block(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=5, stride=1, padding=2, bias=bias)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=5, stride=1, padding=2, bias=bias)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
        x1=self.conv2(nnF.leaky_relu(self.conv1(x), inplace=True))
        x2=x1+x
        x3=nnF.leaky_relu(x2, inplace=True)
        x4=self.pool(x3)
        return x4

class Net(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=5, stride=1, padding=2, bias=bias)
        self.block = nn.ModuleList([Block(bias), Block(bias), Block(bias), Block(bias)])
        self.linear1 = nn.Linear(6*32, 32, bias=bias)
        self.linear2 = nn.Linear(32, 9, bias=bias)

    def forward(self, x):
        x =x.view(x.size(0),1,145)
        x=self.conv0(x)
        x=self.block[0](x)
        x=self.block[1](x)
        x=self.block[2](x)
        x=self.block[3](x)
        #x=self.block[4](x)
        #print(x.size())
        x=x.view(x.size(0),-1)
        x=nnF.leaky_relu(self.linear1(x), inplace=True)
        z=self.linear2(x)
        return z


def run_12ECG_classifier(data,header_data,classes,model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    X=np.asarray(get_12ECG_features(data,header_data))
    X = torch.from_numpy(X)
    X = X.type(torch.FloatTensor)
    X= X.to(device)
    X = X.view(1,1,-1)
    #feats_reshape = features.reshape(1,-1)

    Z = model(X)
    label = Z.data.max(dim=1)[1]
    #label = model.predict(feats_reshape)
    score=nnF.softmax(Z,dim=1)
    #score = model.predict_proba(feats_reshape)


    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] =score[0][i]

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='model99.pt'
    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
    model=Net()
    model.load_state_dict(checkpoint['model_state_dict'])


    return model
