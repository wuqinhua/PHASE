import os,gc
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import cycle
from scipy.stats import rankdata
# from captum.attr import IntegratedGradients
import heapq 

os.chdir("/data/wuqinhua/phase/covid19")

# Load datasets
with open('./Covid_DataList_560_1213.pickle', 'rb') as handle:
    DataList = pickle.load(handle)

with open('./Covid_DataLabel_560_1213.pickle', 'rb') as handle:
    DataLabel = pickle.load(handle)
    
idTmps_csv = pd.read_csv("./sampleid_560.csv")
idTmps = idTmps_csv["sample_id"].tolist()
print(idTmps)

celladata = ad.read_h5ad('./Alldata_anno.h5ad')

        
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
    def __init__(self,DataList, DataLabel, idTmps):
        super().__init__()
        self.DataList = DataList
        self.DataLabel = DataLabel
        self.idTmps = idTmps

    def __len__(self):
        sample_num = self.DataLabel.shape[0]
        return sample_num

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist() 
        dataList = self.DataList[index].to(device)
        dataLabel = self.DataLabel[index].to(device)
        sampleID = self.idTmps[index]
        return dataList, dataLabel, sampleID

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
        A_row = A
        A = F.softmax(A, dim=2)
        x = torch.squeeze(x,0)
        A = torch.squeeze(A,0)
        M = torch.mm(A, x)
        x = self.classifier(M)
        
        return x, M


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_torch(1)

##### Network Run #####

network = SCMIL()
network.load_state_dict(torch.load("./Model_result/PHASE_300.pt"),strict=False)
allData = COVID_Dataset(DataList, DataLabel, idTmps)

allLoader = DataLoader(allData, batch_size=1)
network = network.to(device)
print("model_success")
network.eval()

sample_id_order = pd.read_csv("./sample_560_order.csv")["sample_id"].tolist()

###  sample embedding
attention_matrices = torch.zeros((560, 64), device=device)  # Initialize a tensor to store all attention matrices

for idx, sampleID in enumerate(sample_id_order):
    index = allData.idTmps.index(sampleID)
    data = next(iter(DataLoader(allData, batch_size=1, sampler=SubsetRandomSampler([index]))))
    
    inputs, labels, _ = data
    labels = labels.long()
    with torch.no_grad():
        _, attn_matrix = network(inputs[0])

    attn_matrix = attn_matrix.squeeze(0).squeeze(0)
    attention_matrices[idx] = attn_matrix  # Store each matrix in the row corresponding to the sample

    gc.collect()
    torch.cuda.empty_cache()

df_attention_matrices = pd.DataFrame(attention_matrices.cpu().numpy(), index=sample_id_order)
df_attention_matrices.to_csv("./Model_result/sample_embedding.csv")

 
print('attn_matrix Finished!')