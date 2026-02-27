# GenePT Paper Tracking (Cell-Level)

Sources:
- Paper (PMC mirror): https://pmc.ncbi.nlm.nih.gov/articles/PMC10614824/
- Local PDF: `/Users/andrewrusli/Downloads/genept.pdf`

## Paper tasks and datasets (cell-level)
- Aorta (20% subset), 11 cell types; phenotype labels; patient labels
- Artery, 10 cell types (GEO: GSE159677; data availability: https://www.nature.com/articles/s42003-022-04056-7)
- Bones, 7 cell types (GEO: GSE152805; data availability: https://www.nature.com/articles/s41598-020-67730-y#data-availability)
- Myeloid, 13,468 cells; 11 cell types + 3 cancer types
- Pancreas, 4,218 cells; 11 cell types
- Multiple Sclerosis, 3,430 cells; 18 cell types; 12 donors; age label
- Cardiomyocyte (10% subset), disease phenotype + patient labels (batch effect)

## Task 1: Cell-state association (k-means clustering)
Paper Table 2 (ARI / AMI / ASW). Values copied from the paper table.

| Dataset | Annotation | Geneformer | scGPT | GenePT-w | GenePT-s |
| --- | --- | --- | --- | --- | --- |
| Aorta | Phenotype | 0.10 / 0.12 / -0.005 | 0.12 / 0.12 / 0.01 | 0.09 / 0.12 / -0.04 | 0.12 / 0.11 / 0.02 |
| Aorta | Cell type | 0.21 / 0.31 / -0.04 | 0.47 / 0.64 / 0.18 | 0.54 / 0.60 / 0.03 | 0.31 / 0.47 / 0.04 |
| Artery | Cell type | 0.39 / 0.59 / 0.10 | 0.36 / 0.59 / 0.15 | 0.42 / 0.67 / 0.16 | 0.36 / 0.56 / 0.06 |
| Bones | Cell type | 0.09 / 0.16 / -0.01 | 0.12 / 0.21 / -0.01 | 0.21 / 0.29 / 0.02 | 0.17 / 0.28 / 0.003 |
| Myeloid | Cancer type | 0.16 / 0.18 / 0.03 | 0.27 / 0.29 / 0.08 | 0.25 / 0.27 / 0.02 | 0.17 / 0.17 / 0.06 |
| Myeloid | Cell type | 0.19 / 0.29 / -0.02 | 0.44 / 0.53 / 0.13 | 0.21 / 0.28 / 0.001 | 0.30 / 0.41 / 0.03 |
| Pancreas | Cell type | 0.04 / 0.11 / -0.09 | 0.21 / 0.41 / 0.05 | 0.49 / 0.69 / 0.15 | 0.30 / 0.50 / 0.10 |
| Multiple Sclerosis | Age | 0.04 / 0.11 / -0.10 | 0.04 / 0.11 / -0.06 | 0.07 / 0.13 / -0.07 | 0.06 / 0.12 / -0.03 |
| Multiple Sclerosis | Cell type | 0.21 / 0.35 / -0.05 | 0.25 / 0.44 / 0.04 | 0.17 / 0.32 / -0.02 | 0.19 / 0.35 / 0.002 |

### Reproduction checklist
- Datasets: Aorta (20% subset), Artery, Bones, Myeloid, Pancreas, MS.
- Embeddings: pretrained Geneformer, scGPT, GenePT-w, GenePT-s.
- Clustering: k-means with k = number of classes in the annotation.
- Metrics: ARI, AMI, ASW computed against true annotations.

## Task 2: Cell-type annotation (kNN)
Paper Appendix C Table C4 (10-nearest-neighbor classifier). Metrics are accuracy, precision, recall, F1 (macro).

| Dataset | Embeddings | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Aorta | scGPT | 0.95 | 0.95 | 0.93 | 0.93 |
| Aorta | Geneformer | 0.86 | 0.70 | 0.60 | 0.62 |
| Aorta | GenePT-w | 0.88 | 0.91 | 0.68 | 0.72 |
| Aorta | GenePT-s | 0.86 | 0.70 | 0.60 | 0.62 |
| Aorta | Ensemble (scGPT + GenePT-w + GenePT-s) | 0.93 | 0.95 | 0.82 | 0.86 |
| Artery | scGPT | 0.94 | 0.92 | 0.89 | 0.90 |
| Artery | Geneformer | 0.93 | 0.91 | 0.84 | 0.87 |
| Artery | GenePT-w | 0.95 | 0.92 | 0.87 | 0.88 |
| Artery | GenePT-s | 0.92 | 0.88 | 0.82 | 0.84 |
| Artery | Ensemble (scGPT + GenePT-w + GenePT-s) | 0.95 | 0.93 | 0.88 | 0.90 |
| Bones | scGPT | 0.34 | 0.36 | 0.48 | 0.25 |
| Bones | Geneformer | 0.22 | 0.28 | 0.37 | 0.17 |
| Bones | GenePT-w | 0.49 | 0.49 | 0.60 | 0.36 |
| Bones | GenePT-s | 0.37 | 0.37 | 0.49 | 0.28 |
| Bones | Ensemble (scGPT + GenePT-w + GenePT-s) | 0.45 | 0.43 | 0.57 | 0.33 |
| Myeloid | scGPT | 0.53 | 0.34 | 0.29 | 0.30 |
| Myeloid | Geneformer | 0.44 | 0.26 | 0.18 | 0.20 |
| Myeloid | GenePT-w | 0.50 | 0.35 | 0.30 | 0.31 |
| Myeloid | GenePT-s | 0.52 | 0.33 | 0.27 | 0.28 |
| Myeloid | Ensemble (scGPT + GenePT-w + GenePT-s) | 0.55 | 0.38 | 0.34 | 0.35 |
| Pancreas | scGPT | 0.77 | 0.61 | 0.56 | 0.55 |
| Pancreas | Geneformer | 0.50 | 0.25 | 0.34 | 0.27 |
| Pancreas | GenePT-w | 0.95 | 0.76 | 0.65 | 0.66 |
| Pancreas | GenePT-s | 0.89 | 0.65 | 0.53 | 0.56 |
| Pancreas | Ensemble (scGPT + GenePT-w + GenePT-s) | 0.95 | 0.80 | 0.67 | 0.70 |
| Multiple Sclerosis | scGPT | 0.76 | 0.67 | 0.62 | 0.61 |
| Multiple Sclerosis | Geneformer | 0.44 | 0.47 | 0.36 | 0.34 |
| Multiple Sclerosis | GenePT-w | 0.38 | 0.46 | 0.28 | 0.24 |
| Multiple Sclerosis | GenePT-s | 0.49 | 0.50 | 0.41 | 0.40 |
| Multiple Sclerosis | Ensemble (scGPT + GenePT-w + GenePT-s) | 0.72 | 0.66 | 0.57 | 0.55 |

### Reproduction checklist
- kNN: k = 10, cosine similarity.
- Datasets: Aorta, Artery, Bones, Myeloid, Pancreas, MS.
- Embeddings: scGPT, Geneformer, GenePT-w, GenePT-s, plus ensemble.

## Task 3: Context awareness and batch integration
Paper Section 4.4; focuses on Cardiomyocyte and Aorta subsets.

### Cardiomyocyte (10% subset)
- Patient batch effect (ARI between k-means clusters and patient labels):
  - scRNA-seq data: 0.33
  - GenePT-s: 0.07
  - Geneformer: 0.01
  - scGPT: 0.01
- Disease phenotype classification (80/20 split, logistic regression on embeddings):
  - GenePT-s: 88% accuracy, 88% precision, 88% recall
  - scGPT: 88% accuracy, 88% precision, 88% recall
  - Geneformer: 71% accuracy, 72% precision, 71% recall

### Aorta (20% subset)
- Patient batch effect (ARI between k-means clusters and patient labels, k = 11):
  - scRNA-seq data: 0.24
  - Geneformer: 0.11
  - GenePT-s: 0.10
  - scGPT: 0.18
- Phenotype clustering agreement (ARI between k-means clusters and phenotype labels):
  - Geneformer: 0.12
  - GenePT-s: 0.11
  - scGPT: 0.12
  - scRNA-seq data: 0.12
- Phenotype classification (80/20 split, logistic regression on embeddings):
  - GenePT-s: 73% accuracy, 68% precision, 74% recall
  - scGPT: 75% accuracy, 75% precision, 75% recall
  - Geneformer: 69% accuracy, 68% precision, 69% recall

### Figure references
- Figure 3: Aorta UMAPs. Panels show original scRNA-seq (phenotype, cell type, patient) and GenePT-s embeddings (phenotype, cell type, patient).
- Figure D8 (Appendix): Cardiomyocyte UMAPs. Panels show original scRNA-seq (disease, patient) and GenePT-s embeddings (disease, patient).

### Reproduction checklist
- Datasets: Aorta (20% subset), Cardiomyocyte (10% subset).
- Embeddings: GenePT-s, scGPT, Geneformer, original scRNA-seq.
- Metrics: ARI for patient batch effect; accuracy/precision/recall for phenotype classification.
- Figures: UMAPs for disease/phenotype and patient labels.

## Status snapshot (what we have, what we did, what we need)
### What we have locally
- Datasets present (current paths):
  - Aorta: `datasets/GenePT_analysis_datasets/sample_aorta_data_updated.h5ad`.
  - hPancreas: `datasets/hpancreas/demo_test.h5ad`.
  - Myeloid: `datasets/myeloid/myeloid_combined.h5ad`.
  - MS: `datasets/ms/c_data.h5ad`.
  - Cardiomyocyte: `datasets/GenePT_analysis_datasets/sample_heart_data.h5ad`.
  - Additional (paper subsets): `datasets/GenePT_analysis_datasets/sample_aorta_data_updated.h5ad`, `datasets/GenePT_analysis_datasets/sample_heart_data.h5ad`.
- GEO raw downloads:
  - Bones (GSE152805): `datasets/bones_GSE152805/GSE152805_RAW.tar`.
  - Artery (GSE159677): `datasets/artery_GSE159677/GSE159677_RAW.tar`.
- Generated h5ad (from GEO raw):
  - Bones: `datasets/bones/bones_GSE152805.h5ad` with `tissue`/`region`/`sample_id` in `obs`.
  - Artery: `datasets/artery/artery_GSE159677.h5ad` with `condition`/`sample_id` in `obs`.
- GenePT-s files present:
  - `datasets/GenePT_analysis_datasets/GenePT_s_data/train_X_cell_gpt_pancreas.pickle`.
  - `datasets/GenePT_analysis_datasets/GenePT_s_data/train_X_cell_gpt_pancreas_v2024.pickle`.
- GenePT-w gene embeddings: `embeddings/GenePT_embedding_v2/GenePT_gene_embedding_ada_text.pickle`.
- Additional embeddings in analysis bundle:
  - `datasets/GenePT_analysis_datasets/GPT_model_3_gene_embeddings_model_3_large_clean.pickle`.
  - `datasets/GenePT_analysis_datasets/GPT_model_3_gene_and_protein.pickle`.
- Paper tracking numbers (cell-level) captured in this file.

### What we have produced
- Cell-level benchmark outputs (GenePT-w only): `benchmarks/outputs/cell_level_benchmark.csv|md|json`.
- Benchmark runner now supports scGPT/Geneformer embeddings, ensemble kNN, and Section 4.4 metrics (needs re-run).
- Paper-style tables (GenePT-w only): `benchmarks/outputs/paper_tables/table2_genept_w.*`, `tableC4_genept_w.*`.
- Paper-style figures (GenePT-w only): `benchmarks/outputs/paper_figures/figure3_aorta_genept_w.png`, `figureD8_cardiomyocyte_genept_w.png`.

### What is missing (needed to match the paper)
- Labels: Artery/Bones author-provided cell-type labels.
- Embeddings: GenePT-s cell embeddings (all datasets), Geneformer embeddings, scGPT embeddings.
- Paper-specific runs: GenePT-s UMAPs for Figure 3 and Figure D8; Table 2 / Table C4 results with GenePT-s, Geneformer, scGPT.

### What to verify
- MS age label handling (paper may discretize age).
- Exact sampling strategy for Aorta 20% and Cardiomyocyte 10% subsets.
- Matching label keys/annotation cleaning between paper datasets and local `.h5ad` files.
 - Artery and Bones: download from GEO (GSE159677, GSE152805) and confirm the cell-type labels match paper tasks.

## Comparison to our current runs
### Our current results (GenePT-w only)
Source: `benchmarks/outputs/cell_level_benchmark.csv` (latest run).

| Dataset | Annotation | GenePT-w (ARI / AMI / ASW) | Notes |
| --- | --- | --- | --- |
| Aorta | Phenotype | 0.096 / 0.118 / -0.076 | Uses `datasets/aorta/sample_aorta_data_updated.h5ad` |
| Aorta | Cell type | 0.344 / 0.501 / -0.03 | 5,344 missing genes vs embedding vocab |
| Myeloid | Cancer type | 0.064 / 0.116 / -0.059 | Combined ref+query (13,178 cells) |
| Myeloid | Cell type | 0.159 / 0.269 / -0.088 | Combined ref+query |
| Pancreas (hPancreas) | Cell type | 0.307 / 0.512 / 0.143 | hPancreas `demo_test` (4,218 cells) |
| Multiple Sclerosis | Age | 0.076 / 0.132 / -0.143 | MS `c_data` (3,430 cells), age labels present |
| Multiple Sclerosis | Cell type | 0.039 / 0.144 / -0.21 | MS `c_data` |

### Paper vs our GenePT-w (ARI / AMI / ASW)
Delta = ours - paper. This makes it easy to see how close we are.

| Dataset | Annotation | Paper GenePT-w | Our GenePT-w | Delta (ARI / AMI / ASW) |
| --- | --- | --- | --- | --- |
| Aorta | Phenotype | 0.09 / 0.12 / -0.04 | 0.096 / 0.118 / -0.076 | +0.006 / -0.002 / -0.036 |
| Aorta | Cell type | 0.54 / 0.60 / 0.03 | 0.344 / 0.501 / -0.03 | -0.196 / -0.099 / -0.06 |
| Myeloid | Cancer type | 0.25 / 0.27 / 0.02 | 0.064 / 0.116 / -0.059 | -0.186 / -0.154 / -0.079 |
| Myeloid | Cell type | 0.21 / 0.28 / 0.001 | 0.159 / 0.269 / -0.088 | -0.051 / -0.011 / -0.089 |
| Pancreas (hPancreas) | Cell type | 0.49 / 0.69 / 0.15 | 0.307 / 0.512 / 0.143 | -0.183 / -0.178 / -0.007 |
| Multiple Sclerosis | Age | 0.07 / 0.13 / -0.07 | 0.076 / 0.132 / -0.143 | +0.006 / +0.002 / -0.073 |
| Multiple Sclerosis | Cell type | 0.17 / 0.32 / -0.02 | 0.039 / 0.144 / -0.21 | -0.131 / -0.176 / -0.19 |

## Gaps to close
- Labels: Artery/Bones author-provided cell-type annotations.
- Embeddings: GenePT-s, Geneformer, scGPT (cell-level).
- Verification: MS age binning; subset sampling; label key alignment.
