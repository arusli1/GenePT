# GenePT (local workflow)

This repo includes the original notebooks from GenePT plus a preconfigured Aorta notebook that avoids OpenAI API calls.

## What’s new
- `notebooks/aorta_data_analysis_precomputed.ipynb` runs the Aorta analysis **without OpenAI** by computing GenePT-w (gene-level embeddings) and reusing them wherever GenePT-s cell embeddings are expected.
- The original README is preserved at `README.original.md`.

## Recommended notebook
### Aorta (no OpenAI)
Use:
- `notebooks/aorta_data_analysis_precomputed.ipynb`

This notebook:
- Loads the Aorta dataset.
- Builds GenePT-w from `embeddings/GenePT_embedding_v2/GenePT_gene_embedding_ada_text.pickle`.
- Reuses GenePT-w for the UMAP/batch-effect blocks that normally expect GenePT-s.

## Files and data
This repo expects local data files (not committed):
- `datasets/aorta/sample_aorta_data_updated.h5ad`
- `embeddings/GenePT_embedding_v2/` (gene-level embeddings)
- `embeddings/GenePT_s_data/` (optional; not used by the precomputed notebook)
- `datasets/input_data/` (annotations)

These are ignored via `.gitignore`.

## Original notebooks
If you want the paper’s original workflow (including OpenAI-based GenePT-s):
- `notebooks/aorta_data_analysis.ipynb`

## Notes
- GenePT-w and GenePT-s are **different** in the paper. In the precomputed notebook, GenePT-w is used in place of GenePT-s for convenience.
- If you obtain true GenePT-s Aorta embeddings, you can swap them in by replacing `sampled_cell_aorta_gpt`.

## Cell-level benchmark pipeline (all datasets)
Use the unified benchmark runner for all paper datasets and both GenePT-s/GenePT-w:
- Config: `benchmarks/cell_level_config.yaml`
- Runner: `benchmarks/run_cell_level_benchmark.py`

Steps:
1. Download datasets listed in `README.original.md` (Aorta, hPancreas, Myeloid, MS, Cardiomyocyte).
2. Place files where `benchmarks/cell_level_config.yaml` expects them (or update paths/label keys).
3. Add precomputed GenePT-s cell embeddings for each dataset (required).
4. Run:
   - `python benchmarks/run_cell_level_benchmark.py`

Outputs (under `benchmarks/outputs/`):
- `cell_level_benchmark.csv` / `.md` / `.json`
- UMAP plots per dataset/embedding
- Bar charts for metric comparisons
