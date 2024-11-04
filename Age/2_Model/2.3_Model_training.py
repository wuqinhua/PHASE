
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
from numpy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import networkx as nx
import csv
from tqdm import tqdm

os.chdir("/data/wuqinhua/phase/age/")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Load datasets
with open('./age_datalist_20240908.pickle', 'rb') as handle:
    DataList = pickle.load(handle)

with open('./age_datalabel_20240908.pickle', 'rb') as handle:
    DataLabel = pickle.load(handle)    
# print(DataLabel)

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
    def __init__(self,  L = 1024, D = 256, n_classes = 1):  
        super(SCMIL, self).__init__()

        fc1 = [nn.Linear(5000,256),nn.LeakyReLU(),nn.Dropout(0.25)]
        self.fc1 = nn.Sequential(*fc1).to(device)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2).to(device)
        fc2 = [nn.Linear(256,64),nn.LeakyReLU(),nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=64, D=64, dropout=0.25, n_classes = 1)  
        fc2.append(attention_net)
        self.attention_net = nn.Sequential(*fc2)

        fc_classifier = [nn.Dropout(0.25),nn.Linear(64,1)] 
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
        A = F.softmax(A, dim=2) 
        x = torch.squeeze(x,0)
        A = torch.squeeze(A,0)
        M = torch.mm(A, x)
        x = self.classifier(M)
        return M, x


    

seed_torch(1)
##### Network Run #####

network = SCMIL().to(device)  
trainData = COVID_Dataset(DataList, DataLabel)
trainLoader = DataLoader(trainData, batch_size=1)
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(network.parameters(), lr=0.000005)  


epochs = 1000
max_auc = float("-inf")  

for epoch in range(epochs):
    progress_bar = tqdm(trainLoader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)  
    running_loss = 0.0  

    network.train()  
    for inputs, labels in progress_bar:
        optimizer.zero_grad() 
        _, outputs = network(inputs[0]) 
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels) 
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item() 
        progress_bar.set_postfix(loss=running_loss)  
        
    avg_loss = running_loss / len(trainLoader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
torch.save(network.state_dict(), './Model_result/Age_PHASE_all.pt')
    
print('Training Finished!')




