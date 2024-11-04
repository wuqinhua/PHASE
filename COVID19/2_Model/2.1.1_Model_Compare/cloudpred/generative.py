import collections
import numpy as np
import sklearn.mixture
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
    count = collections.defaultdict(int)
    for X, y, *_ in Xtrain:
        gm[y].append(X)
        count[y] += 1
    
    for state in gm:
        gm[state] = np.concatenate([tensor.cpu().numpy() for tensor in gm[state]])
        model = sklearn.mixture.GaussianMixture(centers)
        gm[state] = model.fit(gm[state])

    return (gm, count)

def eval(model, Xtest):
    gm, count = model
    total = 0.
    correct = 0
    prob = 0.
    y_score = []
    y_true = []
    for X, y, *_ in Xtest:
        logp = {}
        x = -float("inf")
        for state in gm:
            logp[state] = sum(gm[state].score_samples(X.cpu().numpy()))
            x = max(x, logp[state])

        Z = 0
        for state in logp:
            logp[state] = math.exp(logp[state] - x) * count[state]
            Z += logp[state]
        pred = None
        for state in logp:
            logp[state] /= Z
            if pred is None or logp[state] > logp[pred]:
                pred = state

                
        y_score.append([logp[0], logp[1], logp[2]])
        y_true.append(y)
        y_pred = [np.argmax(score) for score in y_score]

        correct += (pred == y)
        prob += logp[y]
    n = len(Xtest)

    res = {}
    res["accuracy"] = correct / float(n)
    res["soft"] = prob / float(n)
    res["auc"] = sklearn.metrics.roc_auc_score(y_true, y_score, multi_class='ovr')
    res["precision_score"] = sklearn.metrics.precision_score(y_true, y_pred, average='weighted')
    res["recall_score"] = sklearn.metrics.recall_score(y_true, y_pred, average='weighted')
    res["F1_score"] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    
    logger = logging.getLogger(__name__)
    logger.debug("        Generative Accuracy:        " + str(res["accuracy"]))
    logger.debug("        Generative AUC:             " + str(res["auc"]))
    logger.debug("        Generative Precision_score: " + str(res["precision_score"]))
    logger.debug("        Generative recall_score:    " + str(res["recall_score"]))
    logger.debug("        Generative F1_score:        " + str(res["F1_score"]))

    return res
