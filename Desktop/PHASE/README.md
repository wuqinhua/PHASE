# PHASE: PHenotype prediction with Attention mechanisms for Single-cell Exploring

**PHASE** utilizes an attention-based neural network framework to predict clinical phenotypes from scRNA-seq data while providing interpretability of key features linked to phenotypic outcomes at both the gene and cell levels. PHASE consists of several components:
- A data-preprocessing procedure
- A gene feature embedding module
- A self-attention (SA) module for cell embedding learning
- An attention-based deep multiple instance learning (AMIL) module for aggregating all single-cell information within a sample


The manuscript has been pre-printed in bioRxiv:
> Qinhua Wu, Junxiang Ding, Ruikun He, Lijian Hui, Junwei Liu, Yixue Li. Exploring phenotype-related single-cells through attention-enhanced representation learning. *bioRxiv* (2024). [https://doi.org/10.1101/2024.10.31.619327](https://doi.org/10.1101/2024.10.31.619327)

#

<img src="https://github.com/wuqinhua/PHASE/blob/main/The%20framework%20of%20PHASE.png" alt="架构图" width="800"/>

***

## Installation
### Installing PHASE package
PHASE is written in Python and can be installed using `pip`:

```bash
pip install phase-sc
```
### Requirements
PHASE should run on any environmnet where Python is available，utilizing PyTorch for its computational needs. The training of PHASE can be done using CPUs only or GPU acceleration. If you do not have powerful GPUs available, it is possible to run using only CPUs. Before using **PHASE**, make sure the following packages are installed:

```bash
scanpy>=1.10.2  
anndata>=0.10.8  
torch>=2.4.0  
tqdm>=4.66.4  
numpy>=1.23.5  
pandas>=1.5.3  
scipy>=1.11.4  
seaborn>=0.13.2  
matplotlib==3.6.3  
captum==0.7.0  
scikit-learn>=1.5.1  
```
To install these dependencies, you can run the following command using `pip`:
```bash
pip install scanpy>=1.10.2 anndata>=0.10.8 torch>=2.4.0 tqdm>=4.66.4 numpy>=1.23.5 pandas>=1.5.3 scipy>=1.11.4 seaborn>=0.13.2 matplotlib==3.6.3 captum==0.7.0 scikit-learn>=1.5.1
```

Alternatively, if you are using a requirements.txt file, you can add these lines to your file and install using:
```bash
pip install -r requirements.txt
```

***

## The PHASE pipeline

1. **Predict clinical phenotypes from scRNA-seq data**
   - 1.1 Data preprocessing: Encode the data into a format that can be read by PHASE.
   - 1.2 Gene feature embedding: Extract and represent gene features.
   - 1.3 Self-attention (SA): Learn cell embeddings.
   - 1.4 Attention-based deep multiple instance learning (AMIL): aggregate all single-cell information within a sample.
   
2. **Provide interpretability of key phenotype-related features**
   - 2.1 Attribution analysis: Use Integrated Gradients (IG) to link genes to phenotypes via attribution scores.
   - 2.2 Attention analysis: Use AMIL attention scores to relate individual cells to the phenotype.
   - 2.3 Conjoint analysis: Correlate top genes' expression levels with cells' attention scores to reveal gene-cell contributions to the phenotype.
     
***

## Usages

### Command Line Arguments

The following table lists the command line arguments available for training the model:

| Abbreviation | Parameter      | Description                                                       |
|--------------|----------------|-------------------------------------------------------------------|
| -t           | --type         | Type of task: classification or regression.                       |
| -p           | --path         | Path to the dataset.                                              |
| -r           | --result       | Path to the directory where results will be saved.                |
| -e           | --epoch        | Number of training epochs (default: 100).                         |
| -l           | --learningrate | Learning rate for the optimizer (default: 0.00001).               |
| -d           | --devices      | List of GPU device IDs to use for training (default: first GPU).  |

Each argument is required unless a default value is specified.

### Example
```bash
PHASEtrain -t classification -p /home/user/PHASE/demo_covid.h5ad -r /home/user/PHASE/result -e 100 -l 0.00001 -d 2
```
***