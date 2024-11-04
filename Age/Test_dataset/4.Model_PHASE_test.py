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
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
from numpy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import networkx as nx
# from torch_sparse import SparseTensor
import csv
from tqdm import tqdm

os.chdir("/data/wuqinhua/phase/age/")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load datasets

with open('./test/test_age_datalist_20240909.pickle', 'rb') as handle:
    DataList = pickle.load(handle)

with open('./test/test_age_datalabel_20240909.pickle', 'rb') as handle:
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
    
    
##### Network Run #####
network = SCMIL()
# network.device('cpu')
network.load_state_dict(torch.load('./test/Model_result//Age_PHASE_test.pt',map_location=torch.device('cpu')))
network.eval()

testData = COVID_Dataset(DataList, DataLabel)
testLoader = DataLoader(testData, batch_size=1)

y_true = []
y_pred = []

for inputs, labels in tqdm(testLoader, desc='Evaluating', leave=False):
    with torch.no_grad():
        _,outputs = network(inputs[0])

        y_true.extend(labels.cpu().numpy())  
        y_pred.extend(outputs.cpu().detach().numpy().flatten())  

print("True Values:", y_true)
print("Predicted Values:", y_pred)

r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.4f}")

cor, _ = pearsonr(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1))
print("p:", _ )
print("Cor:", cor)

result_df = pd.DataFrame({
    'True': np.array(y_true),
    'Pred': np.array(y_pred)
})

result_df.to_csv('./test/Result/true_vs_pred_0909.csv', index=False)
