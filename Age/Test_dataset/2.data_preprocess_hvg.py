import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
import random
import scipy.sparse as sp

os.chdir("/data/wuqinhua/phase/age/")

adatas = ad.read_h5ad('./test/all_pbmc_anno_s.h5ad')
adata = ad.read_h5ad('./test/Age_17_raw.h5ad')
adata.obs_names_make_unique()

hvgGene = pd.read_csv('./Info/Age_hvg_genes.csv', index_col=0) 
hvgGenes = hvgGene.iloc[:, 0].tolist()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

all_genes = adata.var_names     
existing_genes = list(set(hvgGenes) & set(all_genes))
missing_genes = list(set(hvgGenes) - set(existing_genes))

zero_data = sp.lil_matrix((len(adata.obs_names), len(hvgGenes)))

for i, gene in enumerate(hvgGenes):
    if gene in existing_genes:
        print(f"Filling data for gene: {gene}")
        gene_index = adata.var_names.get_loc(gene)  
        zero_data[:, i] = adata[:, gene_index].X  

zero_data = zero_data.tocsr()
new_adata = ad.AnnData(X=zero_data, obs=adata.obs)

results_file = './test/Age_17_hvg_s.h5ad'
new_adata.write(results_file, compression='gzip')

