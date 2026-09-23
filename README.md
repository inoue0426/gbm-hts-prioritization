# GBM-based HTS Prioritization

This repository contains the analysis code used for gradient boosting machine (GBM)-based compound prioritization from high-throughput screening (HTS) data.

The workflow evaluates whether chemical structure and biological annotation features improve the prioritization of active compounds across cell lines. The analysis includes LightGBM classification, learning-to-rank models, and ranking-based evaluation.

## Repository contents

- `01_feature_and_data_prep.ipynb`  
  Prepares the HTS dataset, defines the rule-based hit label, harmonizes compound identifiers, and prepares external annotation features.

- `02_baseline_and_ranking_models.ipynb`  
  Trains and evaluates LightGBM classifier and LambdaRank models using different feature sets, including structure-only and biological annotation-augmented representations.

- `03_results_summary.ipynb`  
  Summarizes predictive and ranking performance and generates the main result tables and visualizations.

- `gbm_pipeline.py`  
  Contains shared utility functions for feature construction, CTD annotation mapping, hit definition, and ranking metrics.

## Analysis overview

The analysis compares three feature configurations:

1. **Structure only** — molecular structure features together with categorical experimental covariates.
2. **Structure + pathway** — structure features augmented with pathway annotations.
3. **Structure + expanded annotations** — structure features augmented with pathway, gene, disease, Gene Ontology, and phenotype annotations.

Compounds are evaluated using both standard predictive metrics and ranking metrics, including:

- ROC-AUC
- PR-AUC
- Precision@K
- Recall@K
- NDCG@K
- MAP@K

Train/test splitting is performed by chemical structure to reduce leakage of identical compounds across partitions.

## Data availability

The data used in this study are not distributed in this repository.

**Data are available from the authors upon reasonable request.**

The notebooks expect the corresponding HTS input data and annotation files to be available locally using the paths specified in the notebooks.

## Requirements

The analysis was implemented in Python and uses the following main packages:

- NumPy
- pandas
- SciPy
- scikit-learn
- LightGBM

RDKit can optionally be used for Morgan fingerprint generation when available. Otherwise, the current workflow supports SMILES character n-gram features.

## Running the analysis

Run the notebooks in the following order:

```text
01_feature_and_data_prep.ipynb
        ↓
02_baseline_and_ranking_models.ipynb
        ↓
03_results_summary.ipynb
```

The first notebook prepares the modeling dataset, the second performs model training and ranking evaluation, and the third summarizes the resulting performance.

## Reproducibility

The modeling workflow uses a fixed random seed where applicable. Because the underlying study data are not included in the public repository, full reproduction requires access to the data described above.

## Citation

Citation information will be added upon publication of the associated manuscript.
