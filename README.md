# PHASE: PHenotype prediction with Attention mechanisms for Single-cell Exploring

**PHASE** utilizes an attention-based neural network framework to predict clinical phenotypes from scRNA-seq data while providing interpretability of key features linked to phenotypic outcomes at both the gene and cell levels. PHASE consists of several components:
- A data-preprocessing procedure
- A gene feature embedding module
- A self-attention (SA) module for cell embedding learning
- An attention-based deep multiple instance learning (AMIL) module for aggregating all single-cell information within a sample


The manuscript has been published in bioRxiv:
> Qinhua Wu, Junxiang Ding, Ruikun He, Lijian Hui, Junwei Liu, Yixue Li. Exploring phenotype-related single-cells through attention-enhanced representation learning. *bioRxiv* (2024). [https://doi.org/10.1101/2024.10.31.619327](https://doi.org/10.1101/2024.10.31.619327)


***

<img src="https://github.com/wuqinhua/PHASE/blob/main/The%20framework%20of%20PHASE.png" alt="架构图" width="800"/>

***

# The PHASE pipeline

```plaintext
1. Predict clinical phenotypes from scRNA-seq data.
   1.1 Data preprocessing: Encode the data into a format that can be read by PHASE.
   1.2 Gene feature embedding: Extract and represent gene features.
   1.3 Self-attention (SA): Learn cell embeddings.
   1.4 Attention-based deep multiple instance learning (AMIL): Aggregate all single-cell information within a sample.
   
2. Provide interpretability of key phenotype-related features.
   2.1 Attribution analysis: Use Integrated Gradients (IG) to link genes to phenotypes via attribution scores.
   2.2 Attention analysis: Use AMIL attention scores to relate individual cells to the phenotype.
   2.3 Conjoint analysis: Correlate top genes' expression levels with cells' attention scores to reveal gene-cell contributions to the phenotype.

***
This repository will be continuously updated during the submission process.
