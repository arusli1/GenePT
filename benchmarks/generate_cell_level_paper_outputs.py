import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import umap

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


RANDOM_SEED = 2023


PAPER_TABLE2_GENEPT_W = {
    ("Aorta", "Phenotype"): (0.09, 0.12, -0.04),
    ("Aorta", "Cell type"): (0.54, 0.60, 0.03),
    ("Myeloid", "Cancer type"): (0.25, 0.27, 0.02),
    ("Myeloid", "Cell type"): (0.21, 0.28, 0.001),
    ("Pancreas", "Cell type"): (0.49, 0.69, 0.15),
    ("Multiple Sclerosis", "Age"): (0.07, 0.13, -0.07),
    ("Multiple Sclerosis", "Cell type"): (0.17, 0.32, -0.02),
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_genept_w_embeddings(gene_embedding_path: Path, gene_names, X):
    with gene_embedding_path.open("rb") as fp:
        gene_embeddings = pickle.load(fp)
    sample = next(iter(gene_embeddings.values()))
    embed_dim = np.asarray(sample, dtype=float).flatten().shape[0]
    lookup = np.zeros((len(gene_names), embed_dim))
    for i, gene in enumerate(gene_names):
        val = gene_embeddings.get(gene)
        if val is None:
            continue
        arr = np.asarray(val, dtype=float).flatten()
        if arr.shape[0] != embed_dim:
            raise ValueError(
                f"Unexpected embedding size for {gene}: {arr.shape[0]} (expected {embed_dim})"
            )
        lookup[i, :] = arr
    X_mat = X.tocsr() if sparse.issparse(X) else np.asarray(X)
    X_mat = X_mat.astype(np.float64)
    row_sums = np.asarray(X_mat.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    if sparse.issparse(X_mat):
        X_norm = X_mat.multiply(1.0 / row_sums[:, None])
        cell_emb = X_norm.dot(lookup)
    else:
        X_norm = X_mat / row_sums[:, None]
        cell_emb = X_norm.dot(lookup)
    cell_emb = np.asarray(cell_emb, dtype=np.float32)
    cell_emb = np.nan_to_num(cell_emb, nan=0.0, posinf=0.0, neginf=0.0)
    return cell_emb


def sanitize_embeddings(cell_emb: np.ndarray) -> np.ndarray:
    cell_emb = np.nan_to_num(cell_emb, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(cell_emb, axis=1)
    zero_rows = norms == 0
    if np.any(zero_rows):
        norms[zero_rows] = 1.0
    cell_emb = cell_emb / norms[:, None]
    return cell_emb


def compute_umap(cell_emb: np.ndarray):
    pca = PCA(n_components=50, random_state=RANDOM_SEED, svd_solver="full")
    pca_result = pca.fit_transform(cell_emb)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value 1 overridden to 1 by setting random_state",
            category=UserWarning,
        )
        return umap.UMAP(
            min_dist=0.5, spread=1, random_state=RANDOM_SEED
        ).fit_transform(pca_result)


def compute_umap_from_expression(adata):
    expr = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    if expr.uns.get("log1p") is None:
        sc.pp.normalize_total(expr, target_sum=1e4)
        sc.pp.log1p(expr)
    sc.pp.pca(expr, n_comps=50, svd_solver="arpack", random_state=RANDOM_SEED)
    sc.pp.neighbors(expr, n_neighbors=15, n_pcs=50)
    sc.tl.umap(expr, random_state=RANDOM_SEED)
    return expr.obsm["X_umap"]


def plot_umap_ax(ax, embedding, labels, title):
    labels_series = pd.Series(labels).astype(str)
    labels_unique = labels_series.unique()
    colors = sns.color_palette("husl", len(labels_unique))
    for i, label_name in enumerate(labels_unique):
        mask = labels_series == label_name
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=2,
            label=label_name,
            color=colors[i],
            alpha=0.3,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def write_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    md_lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    path.with_suffix(".md").write_text("\n".join(md_lines), encoding="utf-8")


def build_table2_genept_w(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, annotation), paper_vals in PAPER_TABLE2_GENEPT_W.items():
        if dataset == "Aorta":
            ds_key = "aorta"
        elif dataset == "Myeloid":
            ds_key = "myeloid"
        elif dataset == "Pancreas":
            ds_key = "hpancreas"
        elif dataset == "Multiple Sclerosis":
            ds_key = "ms"
        else:
            continue
        row = results[(results["dataset"] == ds_key) & (results["embedding"] == "genept_w")]
        if row.empty:
            continue
        row = row.iloc[0]
        if annotation in {"Phenotype", "Cancer type", "Age"}:
            ours = (row["phenotype_ari"], row["phenotype_ami"], row["phenotype_asw"])
        else:
            ours = (row["celltype_ari"], row["celltype_ami"], row["celltype_asw"])
        delta = tuple(round(ours[i] - paper_vals[i], 3) for i in range(3))
        rows.append(
            {
                "Dataset": dataset,
                "Annotation": annotation,
                "Paper GenePT-w (ARI/AMI/ASW)": f"{paper_vals[0]} / {paper_vals[1]} / {paper_vals[2]}",
                "Our GenePT-w (ARI/AMI/ASW)": f"{round(ours[0],3)} / {round(ours[1],3)} / {round(ours[2],3)}",
                "Delta (ours - paper)": f"{delta[0]} / {delta[1]} / {delta[2]}",
            }
        )
    return pd.DataFrame(rows)


def build_table_c4_genept_w(results: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "aorta": "Aorta",
        "hpancreas": "Pancreas",
        "myeloid": "Myeloid",
        "ms": "Multiple Sclerosis",
    }
    rows = []
    for ds_key, display in mapping.items():
        row = results[(results["dataset"] == ds_key) & (results["embedding"] == "genept_w")]
        if row.empty:
            continue
        row = row.iloc[0]
        rows.append(
            {
                "Dataset": display,
                "Embedding": "GenePT-w",
                "Accuracy": round(row["accuracy"], 3),
                "Precision": round(row["precision"], 3),
                "Recall": round(row["recall"], 3),
                "F1": round(row["f1"], 3),
            }
        )
    return pd.DataFrame(rows)


def generate_aorta_figure(adata, genept_w_emb, out_path: Path, config):
    phenotype_key = config["datasets"]["aorta"]["phenotype_key"]
    celltype_key = config["datasets"]["aorta"]["celltype_key"]
    patient_key = config["datasets"]["aorta"]["patient_key"]
    phenotype_labels = adata.obs[phenotype_key].astype(str).to_numpy()
    celltype_labels = adata.obs[celltype_key].astype(str).to_numpy()
    patient_labels = adata.obs[patient_key].astype(str).to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        umap_expr = compute_umap_from_expression(adata)
    umap_genept = compute_umap(sanitize_embeddings(genept_w_emb))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_umap_ax(axes[0, 0], umap_expr, phenotype_labels, "Original: phenotype")
    plot_umap_ax(axes[0, 1], umap_expr, celltype_labels, "Original: cell type")
    plot_umap_ax(axes[0, 2], umap_expr, patient_labels, "Original: patient")
    plot_umap_ax(axes[1, 0], umap_genept, phenotype_labels, "GenePT-w: phenotype")
    plot_umap_ax(axes[1, 1], umap_genept, celltype_labels, "GenePT-w: cell type")
    plot_umap_ax(axes[1, 2], umap_genept, patient_labels, "GenePT-w: patient")
    for ax in axes.flatten():
        ax.legend([], [], frameon=False)
    fig.suptitle("Aorta UMAPs (Original vs GenePT-w)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_cardiomyocyte_figure(adata, genept_w_emb, out_path: Path, config):
    phenotype_key = config["datasets"]["cardiomyocyte"]["phenotype_key"]
    patient_key = config["datasets"]["cardiomyocyte"]["patient_key"]
    phenotype_labels = adata.obs[phenotype_key].astype(str).to_numpy()
    patient_labels = adata.obs[patient_key].astype(str).to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        umap_expr = compute_umap_from_expression(adata)
    umap_genept = compute_umap(sanitize_embeddings(genept_w_emb))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_umap_ax(axes[0, 0], umap_expr, phenotype_labels, "Original: disease")
    plot_umap_ax(axes[0, 1], umap_expr, patient_labels, "Original: patient")
    plot_umap_ax(axes[1, 0], umap_genept, phenotype_labels, "GenePT-w: disease")
    plot_umap_ax(axes[1, 1], umap_genept, patient_labels, "GenePT-w: patient")
    for ax in axes.flatten():
        ax.legend([], [], frameon=False)
    fig.suptitle("Cardiomyocyte UMAPs (Original vs GenePT-w)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "benchmarks" / "cell_level_config.yaml"
    config = load_config(config_path)

    out_dir = root / "benchmarks" / "outputs"
    tables_dir = out_dir / "paper_tables"
    figures_dir = out_dir / "paper_figures"

    results_path = out_dir / "cell_level_benchmark.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing benchmark results: {results_path}. Run run_cell_level_benchmark.py first."
        )
    results = pd.read_csv(results_path)
    results = results[results["embedding"] == "genept_w"].reset_index(drop=True)

    table2 = build_table2_genept_w(results)
    write_table(table2, tables_dir / "table2_genept_w")

    table_c4 = build_table_c4_genept_w(results)
    write_table(table_c4, tables_dir / "tableC4_genept_w")

    # Generate Aorta and Cardiomyocyte figures (GenePT-w only)
    aorta_path = root / config["datasets"]["aorta"]["data_path"]
    cardio_path = root / config["datasets"]["cardiomyocyte"]["data_path"]
    genept_w_path = root / config["embeddings"]["genept_w"]["gene_embedding_path"]

    for ds_name, data_path, fig_name in [
        ("aorta", aorta_path, "figure3_aorta_genept_w.png"),
        ("cardiomyocyte", cardio_path, "figureD8_cardiomyocyte_genept_w.png"),
    ]:
        if not data_path.exists():
            continue
        adata = sc.read_h5ad(data_path)
        adata.obs_names_make_unique()
        expr = adata.raw.to_adata() if adata.raw is not None else adata.copy()
        if "gene_name" in expr.var.columns:
            gene_names = expr.var["gene_name"].astype(str).tolist()
        else:
            gene_names = expr.var.index.astype(str).tolist()
        if expr.uns.get("log1p") is None:
            sc.pp.normalize_total(expr, target_sum=1e4)
            sc.pp.log1p(expr)
        genept_w_emb = load_genept_w_embeddings(
            genept_w_path, gene_names, expr.X
        )
        if ds_name == "aorta":
            generate_aorta_figure(adata, genept_w_emb, figures_dir / fig_name, config)
        else:
            generate_cardiomyocyte_figure(
                adata, genept_w_emb, figures_dir / fig_name, config
            )

    print(f"Wrote tables to {tables_dir} and figures to {figures_dir}")


if __name__ == "__main__":
    main()
