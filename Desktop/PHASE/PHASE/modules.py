import os
import numpy as np
import pandas as pd
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from . import dataset


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


class PHASE(nn.Module):
    def __init__(self, L=1024, D=256, n_classes=3, task_type='classification'):
        super(PHASE, self).__init__()
        
        self.task_type = task_type
        
        self.fc1 = nn.Sequential(
            nn.Linear(5000, 256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=2
        )
        self.attention_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            Attn_Net_Gated(L=64, D=64, dropout=0.25, n_classes=1)
        )

        if self.task_type == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(64, n_classes)
            )
        elif self.task_type == 'regression':
            self.output_layer = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(64, 1)
            )
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        
        initialize_weights(self)

    def forward(self, x, return_attention=False):
        x = self.fc1(x)
        x = self.transformer_encoder(x)
        attention_scores, features = self.attention_net(x)
        attention_scores = F.softmax(attention_scores, dim=1)
        if return_attention:
            return attention_scores.squeeze(-1)
        M = torch.bmm(attention_scores.transpose(1, 2), features)
        x = self.output_layer(M.squeeze(1))
        
        return x
    
    
    
def calculate_attribution_scores(model_path, task_type, datalist, datalabel, idTmps, file_path, result_path, device):
    """
    Calculate and save the attribution for the specified label using Integrated Gradients.

    :param model: The trained model (assumed to be loaded and transferred to the appropriate device).
    :param datalist: List or array of all data.
    :param datalabel: Corresponding labels for the data list.
    :param gene_list: List of gene names corresponding to the model inputs.
    :param result_path: Path to save the resulting CSV of attributions.
    :param device: Device on which computations will be performed ('cuda' or 'cpu').
    """
    model = PHASE()
    model.load_state_dict(torch.load(model_path,weights_only=True),strict=False)
    model.eval()
    model.to(device)
    ig = IntegratedGradients(model)

    TrainData = ad.read_h5ad(file_path)
    gene_list = TrainData.var.index.values.tolist()
    if task_type == 'classification':
        df_attr = []
        unique_labels = np.unique(datalabel)
        for label in unique_labels:
            print(f"Processing label {label}")
            indices = np.where(datalabel == label)[0].tolist() 
            filtered_data = [datalist[i] for i in indices]
            filtered_label = [datalabel[i] for i in indices]
            filtered_idTmps = [idTmps[i] for i in indices]
            data_loader = dataset.get_dataloader(filtered_data, filtered_label,filtered_idTmps,batch_size=1, shuffle=False)
        
            all_attributions = []
            for inputs, labels, _ in data_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                attributions = ig.attribute(inputs, target=labels, n_steps=10)
                attributions = attributions.detach().cpu().numpy().sum(axis=0)
                print("attributions.shape: ",attributions.shape)
                normalized_attributions = np.sum(attributions / np.linalg.norm(attributions, ord=1, axis=1, keepdims=True), axis=0)
                all_attributions.append(normalized_attributions)
                
            mean_attributions = np.mean(np.array(all_attributions), axis=0)
            df_features = pd.DataFrame(list(zip(gene_list, mean_attributions)), columns=["gene", "attr_value"])
            
            label_to_group = dataset.label_mapping(file_path)
            phenotype_name = label_to_group.get(label, 'unknown')
            df_features['phenotype'] = phenotype_name
            
            output_file = os.path.join(result_path, f"attr_{phenotype_name}.csv")
            df_features.to_csv(output_file, index=False)
            df_attr.append(df_features)
            print(f'Attribution results for phenotype {phenotype_name} saved to {output_file}')
    
    elif task_type == 'regression':
        data_loader = dataset.get_dataloader(datalist, datalabel, idTmps, batch_size=1, shuffle=False)
        all_attributions = []
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            attributions = ig.attribute(inputs, target=0, n_steps=10)
            attributions = attributions.detach().cpu().numpy().sum(axis=0)
            print("attributions.shape: ",attributions.shape)
            normalized_attributions = np.sum(attributions / np.linalg.norm(attributions, ord=1, axis=1, keepdims=True), axis=0)
            all_attributions.append(normalized_attributions)
            
        mean_attributions = np.mean(np.array(all_attributions), axis=0)
        df_features = pd.DataFrame(list(zip(gene_list, mean_attributions)), columns=["gene", "attr_value"])
        
        output_file = os.path.join(result_path, f"attr_all.csv")
        df_features.to_csv(output_file, index=False)
        df_attr = df_features
        print(f'Attribution results for phenotype saved to {output_file}')
        
    return df_attr



def calculate_attention_scores(model_path, dataloader, file_path, result_path, device):
    """
    Calculate and save the attention scores for each sample in the dataset.

    :param model_path: Path to the pre-trained model.
    :param dataloader: DataLoader providing batches of input data and labels.
    :param file_path: Path to the .h5ad file containing sample data.
    :param result_path: Directory to save the output CSV file with attention scores.
    :param device: Device on which to run computations ('cuda' or 'cpu').
    :return: DataFrame with calculated attention scores added.
    """
    # Load model and set to evaluation mode
    model = PHASE()
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()
    model.to(device)
    
    train_data = ad.read_h5ad(file_path)
    train_data.obs['attn'] = np.nan  

    sample_id_list = []
    attention_scores_list = []
    
    with torch.no_grad():
        for data in dataloader:
            inputs, labels, sample_id = data
            sample_id = str(sample_id[0] if isinstance(sample_id, (list, tuple)) else sample_id)
            sample_id_list.append(sample_id)
            inputs, labels = inputs.to(device), labels.to(device).long()
            attention_scores = model(inputs, return_attention=True).cpu().numpy().flatten()
            attention_scores_list.append(attention_scores)
            
    for sample_id, attn_scores in zip(sample_id_list, attention_scores_list):
        sample_mask = train_data.obs['sample_id'] == sample_id
        if sample_mask.sum() == len(attn_scores):
            train_data.obs.loc[sample_mask, 'attn'] = attn_scores
        else:
            print(f"Warning: Length mismatch for Sample ID {sample_id}. "
                  f"Mask length: {sample_mask.sum()}, Attention score length: {len(attn_scores)}")

    df_attn = train_data.obs
    output_file = os.path.join(result_path, "attn.csv")
    df_attn.to_csv(output_file)
    print(f"Attention scores saved to {output_file}")
    
    return df_attn