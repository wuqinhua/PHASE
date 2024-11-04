import numpy as np
import collections
import sklearn
import sys
import math
import logging
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


def train(Xtrain, centers=3):
    gm = collections.defaultdict(list)
    for (i, (X, y, *_)) in enumerate(Xtrain):
       
        model = sklearn.mixture.GaussianMixture(min(centers, X.shape[0]))
        model.fit(X.cpu().numpy()) 
        gm[y].append(model)
    
        print("Train " + str(i + 1) + " / " + str(len(Xtrain)), end="\r")
        sys.stdout.flush()
    
    return gm

def eval(gm, Xtest):
    total = 0.
    correct = 0
    prob = 0.
    y_score = []
    y_true = []
    

    for (i, (X, y, *_)) in enumerate(Xtest):
        logp = {}
        x = -float("inf")
        for state in gm:
            logp[state] = list(map(lambda m: sum(m.score_samples(X.cpu().numpy())), gm[state]))
            x = max(x, max(logp[state]))
        
        Z = 0
        for state in logp:
            logp[state] = sum(map(lambda lp: math.exp(lp - x), logp[state]))
            Z += logp[state]
        pred = None
        for state in logp:
            logp[state] /= Z
            if pred is None or logp[state] > logp[pred]:
                pred = state
        total += -math.log(max(1e-50, logp[y]))
        correct += (pred == y)
        prob += logp[y]
        
        y_true.append(y)
        y_score.append(max(logp, key=logp.get))
        y_true_onehot = label_binarize(y_true, classes=list(gm.keys()))
        y_score_onehot = label_binarize(y_score, classes=list(gm.keys()))
    
        print("Test " + str(i + 1) + " / " + str(len(Xtest)) + ": " + str(correct / float(i + 1)), end="\r", flush=True)
    print()
    
    n = len(Xtest)
    res = {}
    res["ce"] = total / float(n)
    res["accuracy"] = correct / float(n)
    res["soft"] = prob / float(n)
    res["auc"] = sklearn.metrics.roc_auc_score(y_true_onehot, y_score_onehot, multi_class='ovr')
    res["precision_score"] = sklearn.metrics.precision_score(y_true, y_score, average='weighted')
    res["recall_score"] = sklearn.metrics.recall_score(y_true, y_score, average='weighted')
    res["f1_score"] = f1_score(y_true, y_score, average='weighted')
    

    logger = logging.getLogger(__name__)
    logger.debug("        Genpat Cross-entropy:   " + str(res["ce"]))
    logger.debug("        Genpat Accuracy:        " + str(res["accuracy"]))
    logger.debug("        Genpat AUC:             " + str(res["auc"]))
    logger.debug("        Genpat precision_score: " + str(res["precision_score"]))
    logger.debug("        Genpat recall_score:    " + str(res["recall_score"]))
    logger.debug("        Genpat f1_score:        " + str(res["f1_score"]))

    return res