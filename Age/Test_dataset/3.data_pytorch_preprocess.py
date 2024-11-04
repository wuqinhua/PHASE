import os
import scanpy as sc
import pandas as pd
import anndata as ad

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle

os.chdir("/data/wuqinhua/phase/age/")

# Load annData
TrainData = ad.read_h5ad('./test/Age_17_hvg_s.h5ad')

dataList = list()
dataLabel = list()
subject_id_list = []

for idTmp in TrainData.obs['Tube_id'].cat.categories:
    print(idTmp)
    subject_id_list.append(idTmp)
    aDataTmp = TrainData[TrainData.obs['Tube_id'] == idTmp]

    dataCell = torch.FloatTensor(aDataTmp.X.todense())
    dataList.append(dataCell)

    dataLabel.append(float(aDataTmp.obs['Age'].values[0]))
      
assert len(dataLabel) == len(dataList)
print(dataLabel)

df_subject_ids = pd.DataFrame(subject_id_list, columns=['Tube_id'])
df_subject_ids.to_csv('./test/sample_ids.csv', index=False)

dataLabels = torch.FloatTensor(dataLabel)


with open('./test/test_age_datalist_20240909.pickle', 'wb') as handle:
    pickle.dump(dataList, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./test/test_age_datalabel_20240909.pickle', 'wb') as handle:
    pickle.dump(dataLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)














































