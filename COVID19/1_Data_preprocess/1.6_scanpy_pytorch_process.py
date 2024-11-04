import os
import scanpy as sc
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle

os.chdir("/data/wuqinhua/phase/covid19")

TrainData = ad.read_h5ad('./Alldata_hvg.h5ad')

idTmps = []
dataList = list()
dataLabel = list()
cell_ids = [] 

for idTmp in TrainData.obs['sample_id'].cat.categories:
    print(idTmp)
    idTmps.append(idTmp)
    aDataTmp = TrainData[TrainData.obs['sample_id'] == idTmp]
    dataCell = torch.FloatTensor(aDataTmp.X.todense())
    dataList.append(dataCell)

    cell_ids.extend(aDataTmp.obs.index) 
    
    if aDataTmp.obs['group'].values[0] == 'H':
        dataLabel.append(0)
    elif aDataTmp.obs['group'].values[0] == 'M':
        dataLabel.append(1)
    elif aDataTmp.obs['group'].values[0] == 'S':
        dataLabel.append(2)
        
df = pd.DataFrame(idTmps, columns=["sample_id"])
df.to_csv("./sampleid_560.csv", index=False)    

df_cell_ids = pd.DataFrame(cell_ids, columns=["cell_id"])
df_cell_ids.to_csv('./cell_id_560.csv', index=False)
   
assert len(dataLabel) == len(dataList)
print(len(dataLabel))

# Build Datasets
lb =  LabelEncoder()
dataLabels = lb.fit_transform(dataLabel)
dataLabels = torch.Tensor(dataLabels)

with open('./Covid_DataList_560_1213.pickle', 'wb') as handle:
    pickle.dump(dataList, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./Covid_DataLabel_560_1213.pickle', 'wb') as handle:
    pickle.dump(dataLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)

