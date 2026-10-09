# Compound Prioritization from Glioblastoma High-Throughput Screening

**Machine-learning prioritization of candidate compounds using chemical structure and biological annotations**

This repository contains an analysis workflow for prioritizing compounds from high-throughput screening (HTS) experiments in glioblastoma (GBM) cell lines. It combines molecular representations, experimental covariates, and biological annotations to investigate whether knowledge-enriched features improve hit prediction and candidate ranking.

The workflow is designed to address a practical screening question: **which compounds should be prioritized for follow-up experiments, and how much do biological annotations contribute beyond chemical structure alone?**

> **Scope and availability:** The repository provides analysis code and documentation. The underlying screening data, CTD mapping inputs, trained models, and generated result tables are **not** distributed here. Accordingly, the repository is **not** a self-contained, end-to-end reproducibility package.

## Study overview

The associated analysis considers HTS data from **four GBM cell lines** (A172, H4, T98G, and U251) across approximately **9,500 compounds** drawn from NCATS screening libraries. The chemical collections include NPC (repurposing-oriented compounds), LOPAC (well-characterized pharmacological agents), and NPACT (mechanistic tool compounds).

We investigate three nested feature configurations:

| Feature configuration | Inputs | Question |
| --- | --- | --- |
| **Structure only** | Chemical structure and experimental covariates (cell line and library) | How well can chemical structure predict HTS hits? |
| **Structure + pathway** | Baseline features plus chemical–pathway associations | Do pathway annotations provide additional information? |
| **Structure + expanded annotations** | Baseline features plus pathways, genes, diseases, Gene Ontology terms, and phenotypes | Does broader biological context improve hit prediction and ranking? |

Biological annotations are loaded from Comparative Toxicogenomics Database (CTD)-derived mapping tables keyed by harmonized chemical names. Structure features use **RDKit Morgan fingerprints** when RDKit is installed; otherwise the code supports a **SMILES character n-gram TF-IDF** representation. These alternatives are **not identical experimental conditions**, so their results should not be compared without tracking the chosen representation.

## Modeling and evaluation

- **Hit definition.** The analysis derives a binary hit label using maximum response, AC50, and curve-class filtering. The implementation is in [`define_hit_label`](gbm_pipeline.py); inspect it for the exact rule and thresholds.
- **Train/test separation.** The modeling notebook uses a single group-based 80/20 split keyed by the `Structure` string, with random seed 42, to keep identical structure strings out of both partitions. This is **not** a scaffold split or independent prospective validation.
- **Models.** LightGBM classification estimates hit probabilities, while LightGBM **LambdaRank** learns cell-line-grouped rankings.
- **Classification metrics.** ROC-AUC and average precision (reported in the code as PR-AUC).
- **Ranking metrics.** Precision@K, Recall@K, NDCG@K, and MAP@K for K = 10, 25, 50. The workflow also produces compound-level rankings aggregated across cell lines.

**Interpretation:** Retrospective ranking metrics estimate how effectively a model prioritizes known HTS hits in the held-out partition. They do **not** demonstrate experimental confirmation of newly nominated compounds.

### Reported research findings

The related study reports improved retrospective performance with expanded annotations compared with structure-based features alone. The study-level summary reports ROC-AUC / PR-AUC of **0.680 / 0.767** for the structure baseline and **0.703 / 0.788** for expanded annotations; reported top-25 ranking metrics include **Precision@25 = 0.940** and **NDCG@25 = 0.957**.

**Important:** These are *study-reported summary figures*, **not results independently verified or reproduced from the files in this repository**. Input data and generated metrics are not included, and the numerical results can depend on the exact data snapshot, annotations, split, feature representation, and evaluation protocol. See the associated manuscript for the definitive experiment definitions when available.

## Repository contents

| File | Purpose |
| --- | --- |
| [`01_feature_and_data_prep.ipynb`](01_feature_and_data_prep.ipynb) | Load HTS summaries, derive hit labels, harmonize chemical names, inspect CTD mappings, and save prepared records. |
| [`02_baseline_and_ranking_models.ipynb`](02_baseline_and_ranking_models.ipynb) | Fit structure/annotation feature sets, LightGBM classifiers, and LambdaRank models; evaluate and export rankings. |
| [`03_results_summary.ipynb`](03_results_summary.ipynb) | Summarize saved evaluation tables and create performance figures. |
| [`gbm_pipeline.py`](gbm_pipeline.py) | Shared utilities for label construction, structure and annotation features, encoders, and ranking metrics. |

## Running the notebooks

**Prerequisites:** Python 3, Jupyter, NumPy, pandas, SciPy, scikit-learn, LightGBM, and Matplotlib. RDKit is optional, but the feature representation changes when it is available.

A minimal environment can be prepared with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas scipy scikit-learn lightgbm matplotlib
# Optional, for Morgan fingerprints:
python -m pip install rdkit
```

The notebooks currently expect **user-provided local input files** relative to the repository root, including:

```text
HTS_summary.csv
CTD mapping/
  CTD_chem_pathways_enriched_1535.tsv
  CTD_chem_cgixns_1535.tsv
  CTD_chem_diseases_1535.tsv
  CTD_chem_go_enriched_1535.tsv
  CTD_chem_phenotypes_curated_1535.tsv
```

Run the notebooks from the repository root, in this order:

1. `01_feature_and_data_prep.ipynb` creates `outputs/hts_prepared.csv`.
2. `02_baseline_and_ranking_models.ipynb` creates classification metrics, ranking metrics, and ranked compound lists under `outputs/`.
3. `03_results_summary.ipynb` reads these outputs and produces summary tables and plots.

The main expected input includes `Structure` (SMILES), `Cell_line`, `library`, `CTD_chemical_name`, `Max Resp`, `AC50 (uM)`, and `CC-v2`; some notebook steps also reference `Hit?`, `Sample ID`, and `Sample Name`. Consult the notebook cells for the complete expected schema.

**Reproduction limitations:** The code is provided for transparency and inspection, but it has not been packaged as a fully data-independent executable example. Reproducing study figures requires matching input files, dependencies, and analysis settings. In particular, preprocessing and feature alignment should be independently audited before using this workflow for a new benchmark or a prospective ranking exercise.

## Data availability and responsible use

The underlying high-throughput screening dataset is being described in a **separate data-focused publication**. Links to that publication and to the dataset will be added here once they are publicly available. The HTS summary and processed annotation mappings are **not included in this repository** at present.

This repository focuses on the **machine-learning analysis and compound-prioritization workflow**; the data publication and any associated modeling publication are distinct outputs. The public code does not include identifiable patient data or screening-result files.

This repository is intended for **computational research**, not clinical decision-making or validated experimental hit selection.

## Publication and citation

This repository accompanies ongoing work on biological-annotation-informed prioritization of compounds from GBM high-throughput screens. A formal bibliographic citation and definitive result reference will be added when the associated manuscript is publicly available.

For other research projects and publications, see the [author's academic website](https://inoue0426.github.io/).
