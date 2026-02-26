---
name: Cell-Level Reproduction
overview: Reproduce the full GenePT cell-level section by locking datasets/labels, generating all embeddings (GenePT-w/-s, scGPT, Geneformer, fine-tuned GenePT-w), and running Table 2, Appendix C, Section 4.4 metrics plus Figures 3/D8.
todos:
  - id: verify-datasets
    content: Verify datasets, standardize label keys, age bins
    status: pending
  - id: artery-bones-labels
    content: Acquire or generate Artery/Bones cell-type labels
    status: pending
  - id: embeddings-pipelines
    content: Build scGPT/Geneformer/GenePT-s/fine-tune scripts
    status: pending
  - id: run-benchmarks
    content: Run all benchmarks + generate figures/tables
    status: pending
  - id: update-tracking
    content: Update paper_tracking with results and gaps
    status: pending
isProject: false
---

# GenePT Cell-Level Reproduction Plan

## Scope and decisions

- Prioritize author-provided processed objects/metadata for Artery/Bones; if missing, re-cluster and marker-label per paper methods.
- Generate all embeddings, including fine-tuned GenePT-w.
- Use GPU for scGPT/Geneformer and fine-tuning once available; proceed with CPU-safe steps now.

## Files we will update

- [benchmarks/cell_level_config.yaml](/Users/andrewrusli/Documents/GenePT/benchmarks/cell_level_config.yaml)
- [benchmarks/run_cell_level_benchmark.py](/Users/andrewrusli/Documents/GenePT/benchmarks/run_cell_level_benchmark.py)
- [benchmarks/generate_cell_level_paper_outputs.py](/Users/andrewrusli/Documents/GenePT/benchmarks/generate_cell_level_paper_outputs.py)
- [benchmarks/paper_tracking.md](/Users/andrewrusli/Documents/GenePT/benchmarks/paper_tracking.md)
- New: `benchmarks/embeddings/` scripts for scGPT/Geneformer/fine-tune pipelines
- New: `datasets/artery/` and `datasets/bones/` label-merge utilities

## Plan

### 1) Dataset verification + label standardization

- Verify each dataset matches paper counts/labels and normalize obs keys to a common schema (`cell_type`, `phenotype`, `patient`, `disease`, `cancer_type`, `age_bin`).
- Aorta/Cardiomyocyte: confirm the GenePT subsets already downloaded are used as the canonical inputs in config.
- MS: add deterministic age binning (e.g., quartiles) and store in `obs["age_bin"]` with a fixed seed.

### 2) Artery/Bones labels (author sources, then reanalysis)

- Search author repos/Zenodo for processed Seurat/metadata with cell-type annotations.
- If not found, run reanalysis pipeline:
  - QC + normalization + HVGs + PCA + neighbors + clustering.
  - Assign cell types via marker gene signatures reported in the papers.
  - Save labeled `.h5ad` and document steps in `benchmarks/research_journal.md`.

### 3) Embedding generation pipelines

- **GenePT-w/GenePT-s**: implement expression-weighted embeddings consistently (log1p + normalize). Ensure gene symbol harmonization and report missing vocab rate.
- **scGPT**: add a script to map genes to scGPT vocab and run inference to extract cell embeddings.
- **Geneformer**: add a script to convert gene rankings to token ids and extract CLS embeddings.
- **Fine-tuned GenePT-w**:
  - Define a lightweight fine-tune head (e.g., MLP/logreg) and store fine-tuned projection.
  - Train separately for cardiomyocyte disease and aorta phenotype tasks.

### 4) Benchmark execution + paper tables/figures

- Run `benchmarks/run_cell_level_benchmark.py` to produce Table 2 / Appendix C / Section 4.4 metrics for all embeddings.
- Run `benchmarks/generate_cell_level_paper_outputs.py` to refresh Figure 3 / Figure D8 and paper-style tables.

### 5) Tracking + gap closure

- Update `benchmarks/paper_tracking.md` with confirmed dataset sources, label provenance, and newly reproduced metrics/figures.
- Summarize remaining gaps (if any) and next steps.

## Notes

- GPU steps (scGPT/Geneformer/fine-tune) will be queued for when GPU access is available.
- All sampling and splits will use a fixed seed for reproducibility.
