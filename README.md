# PHASE: PHenotype prediction with Attention mechanisms for Single-cell Exploring

**PHASE** utilizes an attention-based neural network framework to predict clinical phenotypes from scRNA-seq data while providing interpretability of key features linked to phenotypic outcomes at both the gene and cell levels. PHASE consists of several components:
- A data-preprocessing procedure
- A gene feature embedding module
- A self-attention (SA) module for cell embedding learning
- An attention-based deep multiple instance learning (AMIL) module for aggregating all single-cell information within a sample


The manuscript has been published in bioRxiv:
> Qinhua Wu, Junxiang Ding, Ruikun He, Lijian Hui, Junwei Liu, Yixue Li. Exploring phenotype-related single-cells through attention-enhanced representation learning. *bioRxiv* (2024). [https://doi.org/10.1101/2024.10.31.619327](https://doi.org/10.1101/2024.10.31.619327)

# 

<img src="https://github.com/wuqinhua/PHASE/blob/main/The%20framework%20of%20PHASE.png" alt="架构图" width="800"/>

***

## The PHASE pipeline

1. **Predict clinical phenotypes from scRNA-seq data**
   - 1.1 Data preprocessing: Encode the data into a format that can be read by PHASE.
   - 1.2 Gene feature embedding: Extract and represent gene features.
   - 1.3 Self-attention (SA): Learn cell embeddings.
   - 1.4 Attention-based deep multiple instance learning (AMIL): all single-cell information within a sample.
   
2. **Provide interpretability of key phenotype-related features**
   - 2.1 Attribution analysis: Use Integrated Gradients (IG) to link genes to phenotypes via attribution scores.
   - 2.2 Attention analysis: Use AMIL attention scores to relate individual cells to the phenotype.
   - 2.3 Conjoint analysis: Correlate top genes' expression levels with cells' attention scores to reveal gene-cell contributions to the phenotype.
     
***

## Usages

- **Data Preprocessing**: The folder contains preprocessing scripts and notebooks, including single-cell integration and annotation, available here [COVID-19](https://github.com/wuqinhua/PHASE/tree/main/COVID19/1_Data_preprocess) and [Age](https://github.com/wuqinhua/PHASE/tree/main/Age/1_Data_preprocess).

- **Model Training**: Details of model training can be found in the [COVID-19 Model Training](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.1.2_Model_Training.py) and [Age Model Training](https://github.com/wuqinhua/PHASE/blob/main/Age/2_Model/2.3_Model_training.py) scripts.

- **Attribution Analysis**: 
  - For COVID-19 data, gene attribution scores can be computed using [COVID-19 Attribution Group PHASE](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.2.1_attribution_group_PHASE.py) and [COVID-19 Attribution Sample PHASE](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.2.2_attribution_sample_PHASE.py). Visualization of results is available in the [COVID-19 Attribution Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/COVID19/3_Analysis/3.1_Attribution_analysis.ipynb).
  - For Age data, gene attribution scores can be computed using [Age Attribution PHASE](https://github.com/wuqinhua/PHASE/blob/main/Age/2_Model/2.4_Attribution_PHASE.py), and result visualization is available in the [Age Attribution Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/Age/3_Analysis/3.2_Attention_analysis.ipynb).

- **Attention Analysis**: 
  - For COVID-19 data, cell attention scores can be computed using [COVID-19 Attention PHASE](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.3_attention_cell_PHASE.py), and result visualization is available in the [COVID-19 Attention Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/COVID19/3_Analysis/3.2_Attention_analysis.ipynb).
  - For Age data, cell attention scores can be computed using [Age Attention PHASE](https://github.com/wuqinhua/PHASE/blob/main/Age/2_Model/2.5_Attention_PHASE.py), and result visualization is available in the [Age Attention Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/Age/3_Analysis/3.2_Attention_analysis.ipynb).

- **Conjoint Analysis**: Details are available in the [COVID-19 Conjoint Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/COVID19/3_Analysis/3.3_Conjoint_analysis.ipynb) and the [Age Conjoint Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/Age/3_Analysis/3.3_Conjoint_analysis.ipynb).





This repository will be continuously updated during the submission process.
