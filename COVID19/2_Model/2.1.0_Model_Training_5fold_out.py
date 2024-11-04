import os
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import networkx as nx
# from torch_sparse import SparseTensor
import csv
from tqdm import tqdm

os.chdir("/data/wuqinhua/phase/covid19")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets
with open('./Covid_DataList_560_1213.pickle', 'rb') as handle:
    DataList = pickle.load(handle)

with open('./Covid_DataLabel_560_1213.pickle', 'rb') as handle:
    DataLabel = pickle.load(handle)    
print(DataLabel)


def seed_torch(seed=777):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Build Pytorch Dataset
class COVID_Dataset(Dataset):
    def __init__(self,DataList,DataLabel):
        super().__init__()
        self.DataList = DataList
        self.DataLabel = DataLabel

    def __len__(self):
        sample_num = self.DataLabel.shape[0]
        return sample_num

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist() 
        dataList = self.DataList[index].to(device)
        dataLabel = self.DataLabel[index].to(device)
        return dataList, dataLabel


 
# Attn_Net_Gated
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()

        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x
    
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# SCMIL
class SCMIL(nn.Module):
    def __init__(self,  L = 1024, D = 256, n_classes = 3):
        super(SCMIL, self).__init__()

        fc1 = [nn.Linear(5000,256),nn.ReLU(),nn.Dropout(0.25)]
        self.fc1 = nn.Sequential(*fc1).to(device)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2).to(device)
        fc2 = [nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=64, D=64, dropout=0.25, n_classes = 1)
        fc2.append(attention_net)
        self.attention_net = nn.Sequential(*fc2)

        fc_classifier = [nn.Linear(64,3)]
        self.classifier = nn.Sequential(*fc_classifier)
        initialize_weights(self)

        self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, x):
        x = self.fc1(x)
        x = torch.unsqueeze(x,0)
        x = self.transformer_encoder(x)
        A, x = self.attention_net(x)
        A = torch.transpose(A, 1, 2)
        # A_row = A
        A = F.softmax(A, dim=2)
        x = torch.squeeze(x,0)
        A = torch.squeeze(A,0)
        
        M = torch.mm(A, x)
        x = self.classifier(M)
        
        return M,x

seed_torch(1)
##### Network Run #####
# Five-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
list_auc = []
list_precision = []  
list_recall = []     
list_f1 = []         


for fold, (train_idx, valid_idx) in enumerate(skf.split(DataList, DataLabel)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    print(train_idx)
    
    train_data = [DataList[i] for i in train_idx]
    train_label = DataLabel[train_idx]
    valid_data = [DataList[i] for i in valid_idx]
    valid_label = DataLabel[valid_idx]
    
    network = SCMIL().to(device) 
    trainData = COVID_Dataset(train_data, train_label)
    validData = COVID_Dataset(valid_data, valid_label)
    trainLoader = DataLoader(trainData, batch_size=1)
    validLoader = DataLoader(validData, batch_size=1)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(network.parameters(), lr=0.00001)


    epochs = 400
    max_auc = float("-inf")  

    for epoch in range(epochs):
        progress_bar = tqdm(trainLoader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        running_loss = 0.0  
        
        
        network.train() 
        for inputs, labels in progress_bar:
            labels = labels.long() 
            optimizer.zero_grad() 
            _,outputs = network(inputs[0]) 
            loss = criterion(outputs, labels) 
            loss.backward()
          
            optimizer.step() 
            running_loss += loss.item() 
            progress_bar.set_postfix(loss=running_loss) 

        avg_loss = running_loss / len(trainLoader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')
  

        network.eval()
        y_true = [] 
        y_pred = [] 
        
        for i, data in enumerate(validLoader, 0):
            inputs, labels = data
            labels = labels.long()
        
            with torch.no_grad():
                _,outputs = network(inputs[0]) 
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(F.softmax(outputs, dim=1).cpu().detach().numpy())
               
                
        # Compute ROC curve and ROC area for each class
        fpr = dict()  
        tpr = dict()  
        roc_auc = dict()
        num_classes = len(np.unique(y_true)) 
        y_true = LabelBinarizer().fit_transform(y_true)
        y_pred = np.array(y_pred)
        y_pred = np.squeeze(y_pred)
        y_true = np.array(y_true)
        y_true = np.squeeze(y_true)
 
        
        for i in range (num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])


        # Finally average it and compute AUC
        mean_tpr /= num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        roc_auc = roc_auc["macro"]
       
    
        if roc_auc > max_auc:
            torch.save(network.state_dict(), './Model_result/Training_5fold_{}.pt'.format(fold))
            max_auc = roc_auc
            print("max_auc increase",max_auc)
            
        
            y_true_fold = []  
            y_pred_fold = [] 
            for i, data in enumerate(validLoader, 0):
                inputs, labels = data
                labels = labels.long()
                
                with torch.no_grad():
                    _,outputs = network(inputs[0])

                    y_true_fold.extend(labels.cpu().numpy())
                    y_pred_fold.extend(F.softmax(outputs, dim=1).cpu().detach().numpy())
                    np.savetxt('./Model_result/Training_5fold_{}_ture.csv'.format(fold),y_true_fold)
                    np.savetxt('./Model_result/Training_5fold_{}_pred.csv'.format(fold),y_pred_fold)
          
            precision = precision_score(y_true_fold, np.argmax(y_pred_fold, axis=1), average='weighted', zero_division=1)
            recall = recall_score(y_true_fold, np.argmax(y_pred_fold, axis=1), average='weighted')
            f1 = f1_score(y_true_fold, np.argmax(y_pred_fold, axis=1), average='weighted')
           
    list_auc.append(max_auc)
    list_precision.append(precision)
    list_recall.append(recall)
    list_f1.append(f1)

mean_auc = np.mean(list_auc)
mean_precision = np.mean(list_precision)
mean_recall = np.mean(list_recall)
mean_f1 = np.mean(list_f1)

print("mean_auc =", mean_auc)
print("mean_precision =", mean_precision)
print("mean_recall =", mean_recall)
print("mean_f1 =", mean_f1)
print("AUC list:", list_auc)
print("Precision list:", list_precision)
print("Recall list:", list_recall)
print("F1 list:", list_f1)

print('Training Finished!')







