import os
import scanpy as sc
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder


def process_data(file_path, task_type):
    TrainData = ad.read_h5ad(file_path)
    dataList = []
    dataLabel = []
    idTmps = []

    for idTmp in TrainData.obs['sample_id'].cat.categories:
        print(idTmp)
        idTmps.append(idTmp)
        aDataTmp = TrainData[TrainData.obs['sample_id'] == idTmp]
            
        # Check if data is sparse and convert accordingly
        if sp.issparse(aDataTmp.X):
            dataCell = torch.FloatTensor(aDataTmp.X.todense())
        else:
            dataCell = torch.FloatTensor(aDataTmp.X)
        dataList.append(dataCell)

        if task_type == 'classification':
            all_labels = TrainData.obs['phenotype'].cat.categories.tolist()
            label_encoder = create_label_encoder(all_labels)
            label = label_encoder.transform([aDataTmp.obs['phenotype'].values[0]])[0] # For classification, you might want to map this to a class label
            print(f"Original phenotype: {aDataTmp.obs['phenotype'].values[0]}, Encoded label: {label}")
        elif task_type == 'regression':
            label = float(aDataTmp.obs['phenotype'].values[0])  # Direct float value for regression
        dataLabel.append(label)
    
    dataLabel = torch.FloatTensor(dataLabel)
    assert len(dataLabel) == len(dataList), "Data labels and data list should be of the same length"
    return dataList, dataLabel, idTmps


def create_label_encoder(categories):
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    return label_encoder


class PHASE_Dataset(Dataset):
    def __init__(self, DataList, DataLabel, idTmps):
        super().__init__()
        self.DataList = DataList
        self.DataLabel = DataLabel
        self.idTmps = idTmps

    def __len__(self):
        return len(self.DataLabel)

    def __getitem__(self, index):
        dataList = self.DataList[index]
        dataLabel = self.DataLabel[index]
        sampleID = self.idTmps[index]
        return dataList, dataLabel, sampleID 
    

def get_dataloader(data_list, data_label, idTmps, batch_size=1, shuffle=True):
    dataset = PHASE_Dataset(data_list, data_label, idTmps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def label_mapping(file_path):
    adata = ad.read_h5ad(file_path)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(adata.obs["phenotype"])
    label_to_group = dict(zip(encoded_labels, adata.obs["phenotype"]))
    return label_to_group