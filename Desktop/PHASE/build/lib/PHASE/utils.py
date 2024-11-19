import os
import gc
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
from torch import nn, optim
from tqdm import tqdm
from . import modules

import matplotlib.pyplot as plt
import seaborn as sns



def seed_torch(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def setup_device(devices):
    available_devices = []
    for device_id in devices:
        try:
            torch.cuda.get_device_properties(device_id)
            available_devices.append(torch.device(f"cuda:{device_id}"))
            print(f"Device cuda:{device_id} is available.")
        except AssertionError:
            print(f"Device cuda:{device_id} is not available.")
    if not available_devices:
        print("No valid CUDA devices specified, using CPU.")
        return torch.device("cpu")
    print(f"Using GPU device: {available_devices[0]}")
    return available_devices[0]



def train_model(task_type, lr, train_loader, device, epochs, result_path):
    model = modules.PHASE().to(device)
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Unsupported task type. Choose 'classification' or 'regression'.")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if task_type == 'classification':
                labels = labels.long()  
            optimizer.zero_grad()
            output = model(inputs)   
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}')
    model_path = os.path.join(result_path, f"PHASE_model_{lr}_{epochs}.pt")
    torch.save(model.state_dict(), model_path)
    return model_path
    
    
    
def plot_attribution_scores(task_type, df_attr, result_path):
    if task_type == 'classification':
        num_labels = len(df_attr)
        fig, axes = plt.subplots(1, num_labels, figsize=(8 * num_labels, 5 * num_labels)) 
        if num_labels == 1:
            axes = [axes] 

        for i, df_features in enumerate(df_attr):
            top20 = df_features.sort_values(by='attr_value', ascending=False).head(20)
            ax = axes[i]

            base_color = sns.color_palette('husl', n_colors=num_labels)[i]
            colors = sns.light_palette(base_color, n_colors=20, reverse=True)

            sns.barplot(data=top20, y='gene', x='attr_value', palette=colors, ax=ax, hue='gene', dodge=False, legend=False)
            ax.set_title(f"Attribution Score of Top 20 Genes for {df_features['phenotype'].iloc[0]}",fontsize=18)
            ax.set_xlabel('Attribution Score',fontsize=16)
            ax.set_ylabel('')
            
            for index, row in top20.iterrows():
                ax.text(row.attr_value, index, f'{row.attr_value:.3f}', color='black', ha="right")

            if ax.get_legend():
                ax.get_legend().remove()

        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.05, right=0.95)  
        output_file = os.path.join(result_path, "attr_top20_genes.pdf")
        plt.savefig(output_file)
        plt.close()
        print(f"Attribution plot saved to {output_file}")
        
    elif task_type == 'regression':
        top20 = df_attr.sort_values(by='attr_value', ascending=False).head(20)
        bottom20 = df_attr.sort_values(by='attr_value', ascending=True).head(20)

        fig, axes = plt.subplots(1, 2, figsize=(20,12))

        colors_top = sns.light_palette("green", n_colors=20, reverse=True)
        sns.barplot(data=top20, y='gene', x='attr_value', palette=colors_top, ax=axes[0], dodge=False, legend=False)
        axes[0].set_title(f"Top 20 Genes", fontsize=18)
        axes[0].set_xlabel('Attribution Score', fontsize=16)
        axes[0].set_ylabel('')

        for index, row in top20.iterrows():
            axes[0].text(row.attr_value, index, f'{row.attr_value:.3f}', color='black', ha="right")

        colors_bottom = sns.light_palette("gold", n_colors=20)
        sns.barplot(data=bottom20, y='gene', x='attr_value', palette=colors_bottom, ax=axes[1], dodge=False, legend=False)
        axes[1].set_title(f"Bottom 20 Genes", fontsize=18)
        axes[1].set_xlabel('Attribution Score', fontsize=16)
        axes[1].set_ylabel('')

        for index, row in bottom20.iterrows():
            axes[1].text(row.attr_value, index, f'{row.attr_value:.3f}', color='black', ha="right")
        
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()    
                    
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.05, right=0.95) 
        output_file = os.path.join(result_path, "attr_top20_bottom_20_genes.pdf")
        plt.savefig(output_file)
        plt.close()
        print(f"Attribution plot saved to {output_file}")
       
            

def plot_attention_scores(task_type, df_attn, file_path, result_path): 
    """
    Generate UMAP plots for attention scores across different groups and cell type predictions.
    :param df_attn: DataFrame with attention scores and sample IDs.
    :param file_path: Path to the .h5ad file containing data for additional annotations.
    :param result_path: Directory to save the plots.
    """
    id_list = df_attn['sample_id'].unique()
    for sample_id in id_list:
        attn_tmp = df_attn[df_attn['sample_id'] == sample_id]
        avg_score = 1 / len(attn_tmp)
        log_attn = np.log2(attn_tmp['attn'] / avg_score)
        attn_scaled = (log_attn - np.mean(log_attn)) / np.std(log_attn)
        attn_scaled_clipped = np.clip(attn_scaled, -1, 1)
        df_attn.loc[df_attn['sample_id'] == sample_id, 'attn_scaled'] = attn_scaled_clipped

    adata = ad.read_h5ad(file_path)
    adata.obs["attn_scaled"] = df_attn["attn_scaled"].values
    adata.obs["attn"] = df_attn["attn"].values

    if task_type == 'classification':
        unique_groups = adata.obs['phenotype'].unique()
        print(f"Detected groups: {unique_groups}")
        
        sc.settings.figdir = result_path
        sc.pl.umap(
            adata,
            color=['celltype'],
            show=False,
            palette=sns.color_palette("husl", 24),  
            legend_fontsize=6,
            frameon=True,
            title='Cell Type',
            save= "_celltype.pdf")
        print("UMAP plot for cell types saved as 'umap_celltype.pdf'")
        
        for group in unique_groups:
            group_adata = adata[adata.obs['phenotype'] == group]
            sc.pl.umap(
                group_adata, 
                color='attn_scaled', 
                show=False, 
                legend_fontsize=6, 
                color_map='viridis', 
                frameon=True, 
                title=f'Attention Score of {group}', 
                save=f"_attn_{group}.pdf")
            print(f"UMAP plot for group '{group}' saved as 'umap_attn_{group}.pdf'")
        
    elif task_type == 'regression':
        sc.settings.figdir = result_path
        sc.pl.umap(
            adata,
            color=['celltype'],
            show=False,
            palette=sns.color_palette("husl", 24),  
            legend_fontsize=6,
            frameon=True,
            title='Cell Type',
            save= "_celltype.pdf")
        print("UMAP plot for cell types saved as 'umap_celltype.pdf'")
        
        sc.pl.umap(
                adata, 
                color='attn_scaled', 
                show=False, 
                legend_fontsize=6, 
                color_map='viridis', 
                frameon=True, 
                title=f'Attention Score of phenotype', 
                save=f"_attn_all.pdf")
        print(f"UMAP plot for phenotype saved as 'umap_attn_all.pdf'")
        
        
