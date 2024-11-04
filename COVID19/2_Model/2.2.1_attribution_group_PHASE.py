import os,gc
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

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
from numpy import interp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import cycle
from captum.attr import IntegratedGradients
import heapq 

os.chdir("/data/wuqinhua/phase/covid19")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets
with open('./Covid_DataList_560_1213.pickle', 'rb') as handle:
    DataList = pickle.load(handle)

with open('./Covid_DataLabel_560_1213.pickle', 'rb') as handle:
    DataLabel = pickle.load(handle)
print(DataLabel)

gene_list_df = pd.read_csv(r"./hvg_genes.csv")
gene_list = gene_list_df['genes'].values.tolist()
print(gene_list)


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
        
        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_torch(1)

##### Network Run #####
network = SCMIL()
network.load_state_dict(torch.load("./Model_result/PHASE_300.pt"),strict=False)
allData = COVID_Dataset(DataList, DataLabel[DataLabel == 0])  ### "H"0，"M"1,"S"2 
print(DataLabel[DataLabel == 0]) ### "H"0，"M"1,"S"2
allLoader = DataLoader(allData, batch_size=1)
network = network.to(device)
print("model_success")
ig = IntegratedGradients(network)
print("IG success")
network.eval()

ig_attr_list = []
for i, data in enumerate(allLoader, 0):
    inputs, labels = data
    labels = labels.long()
    ig_attr_test = ig.attribute(inputs=inputs[0],target=labels,n_steps=10)

    inputs.detach()
    labels.detach()
    print(ig_attr_test.size())
    ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
    print(ig_attr_test_sum.shape)
    ig_attr_test_norm_sum_1 = (ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)).copy()
    ig_attr_list.append(ig_attr_test_norm_sum_1)
    gc.collect()
    torch.cuda.empty_cache()
ig_attr_list = np.array(ig_attr_list)
mean_attr = np.mean(ig_attr_list,axis=0)

df_fea = pd.DataFrame(list(zip(gene_list,mean_attr)),columns=["gene_name","attr_value"])
df_fea.to_csv(r"./Analysis_result/Attr_result/attr_H_PHASE.csv")
# df_fea.to_csv(r"./Attr_result/attr_M_PHASE.csv")
# df_fea.to_csv(r"./Attr_result/attr_S_PHASE.csv")
print('IG Finished!')






