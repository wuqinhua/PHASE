import cloudpred
import numpy as np
import sklearn
import torch
import math
import torch
from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(Xtrain, Xvalid, centers=2, regression=False):
    outputs = 3
    classifier = torch.nn.Sequential(torch.nn.Linear(Xtrain[0][0].shape[1], centers), torch.nn.ReLU(), torch.nn.Linear(centers, centers), torch.nn.ReLU(), cloudpred.utils.Aggregator(), torch.nn.Linear(centers, centers), torch.nn.ReLU(), torch.nn.Linear(centers, outputs)).to(device)
    reg = None
    return cloudpred.utils.train_classifier(Xtrain, Xvalid, [], classifier, regularize=reg, iterations=1000, eta=1e-4, stochastic=True, regression=regression)


def eval(model, Xtest, regression=False):
    model.to(device)
    reg = None
    model, res =  cloudpred.utils.train_classifier([], Xtest, [], model, regularize=reg, iterations=1, eta=0, stochastic=True, regression=regression)
    return res













