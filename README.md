# GenePT (local workflow)

This repo includes the original notebooks from GenePT plus a preconfigured Aorta notebook that avoids OpenAI API calls.

## What’s new
- `aorta_data_analysis_precomputed.ipynb` runs the Aorta analysis **without OpenAI** by computing GenePT-w (gene-level embeddings) and reusing them wherever GenePT-s cell embeddings are expected.
- The original README is preserved at `README.original.md`.

## Recommended notebook
### Aorta (no OpenAI)
Use:
- `aorta_data_analysis_precomputed.ipynb`

This notebook:
- Loads the Aorta dataset.
- Builds GenePT-w from `GenePT_embedding_v2/GenePT_gene_embedding_ada_text.pickle`.
- Reuses GenePT-w for the UMAP/batch-effect blocks that normally expect GenePT-s.

## Files and data
This repo expects local data files (not committed):
- `sample_aorta_data_updated.h5ad`
- `GenePT_embedding_v2/` (gene-level embeddings)
- `GenePT_s_data/` (optional; not used by the precomputed notebook)
- `input_data/` (annotations)

These are ignored via `.gitignore`.

## Original notebooks
If you want the paper’s original workflow (including OpenAI-based GenePT-s):
- `aorta_data_analysis.ipynb`

## Notes
- GenePT-w and GenePT-s are **different** in the paper. In the precomputed notebook, GenePT-w is used in place of GenePT-s for convenience.
- If you obtain true GenePT-s Aorta embeddings, you can swap them in by replacing `sampled_cell_aorta_gpt`.
