import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap
import yaml
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


RANDOM_SEED = 2023
FIG_DPI = 300


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_plot_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


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
        if arr.shape[0] == embed_dim:
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
    norms = np.linalg.norm(cell_emb, axis=1)
    norms[norms == 0] = 1.0
    return cell_emb / norms[:, None]


def dense_array(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def prepare_expr(adata):
    expr = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    if expr.uns.get("log1p") is None:
        sc.pp.normalize_total(expr, target_sum=1e4)
        sc.pp.log1p(expr)
    return expr


def jaccard(a: set, b: set) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def connected_components_by_similarity(program_sets, threshold: float):
    n = len(program_sets)
    visited = np.zeros(n, dtype=bool)
    components = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            cur = stack.pop()
            for j in range(n):
                if visited[j]:
                    continue
                if jaccard(program_sets[cur], program_sets[j]) >= threshold:
                    visited[j] = True
                    stack.append(j)
                    comp.append(j)
        components.append(comp)
    return components


def build_candidate_programs(expr, label_key: str, labels: np.ndarray, top_labels):
    sc.tl.rank_genes_groups(expr, groupby=label_key, method="wilcoxon", n_genes=240)
    programs = []
    for label in top_labels:
        ranked = sc.get.rank_genes_groups_df(expr, group=label)["names"].tolist()
        ranked = [g for g in ranked if g in expr.var_names]
        ranked = ranked[:200]
        # Build multiple overlapping candidate sets per cell type.
        for start in range(0, 120, 10):
            genes = ranked[start : start + 20]
            genes = list(dict.fromkeys(genes))
            if len(genes) > 10:
                programs.append({"source_label": label, "genes": genes})
    return programs


def build_program_matrix(expr, labels: np.ndarray, top_labels, merged_programs):
    X = expr.X.tocsr() if sparse.issparse(expr.X) else np.asarray(expr.X)
    data = np.zeros((len(merged_programs), len(top_labels)), dtype=float)
    row_names = []
    for row_idx, genes in enumerate(merged_programs, start=1):
        gene_idx = [expr.var_names.get_loc(g) for g in genes if g in expr.var_names]
        if not gene_idx:
            continue
        row_names.append(f"Gene set {row_idx}: {', '.join(genes[:6])}...")
        for col_idx, target_label in enumerate(top_labels):
            mask = labels == target_label
            vals = X[mask][:, gene_idx].mean()
            data[row_idx - 1, col_idx] = float(vals)
    row_min = data.min(axis=1, keepdims=True)
    row_max = data.max(axis=1, keepdims=True)
    denom = np.where((row_max - row_min) == 0, 1.0, row_max - row_min)
    data = (data - row_min) / denom
    return pd.DataFrame(data, index=row_names, columns=top_labels)


def build_gene_program_heatmap(
    config: dict,
    root: Path,
    out_dir: Path,
    threshold: float,
    fig_tag: str,
    out_name: str,
    max_programs: int = 35,
    random_subset_labels: bool = False,
):
    ds_name = "myeloid"
    ds_cfg = config["datasets"].get(ds_name)
    if ds_cfg is None:
        raise ValueError("Expected myeloid dataset for immune-style gene-program panel.")

    adata = sc.read_h5ad(root / ds_cfg["data_path"])
    label_key = ds_cfg.get("celltype_key")
    if label_key is None or label_key not in adata.obs:
        raise ValueError(f"{ds_name}: missing celltype labels for gene-program panel.")

    labels_ser = adata.obs[label_key].astype(str)
    top_labels = labels_ser.value_counts().head(14).index.tolist()
    keep = labels_ser.isin(top_labels).to_numpy()
    adata = adata[keep].copy()
    labels = adata.obs[label_key].astype(str).to_numpy()
    expr = prepare_expr(adata)

    candidate_programs = build_candidate_programs(expr, label_key, labels, top_labels)
    program_sets = [set(p["genes"]) for p in candidate_programs]
    comps = connected_components_by_similarity(program_sets, threshold=threshold)

    merged_programs = []
    for comp in comps:
        union_genes = set()
        for idx in comp:
            union_genes |= program_sets[idx]
        # Keep stable program-size rows similar to paper's ">10 genes".
        if len(union_genes) > 10:
            merged_programs.append(sorted(union_genes))

    # Keep a readable panel size while preserving non-square shape.
    merged_programs = sorted(merged_programs, key=len, reverse=True)[:max_programs]
    df = build_program_matrix(expr, labels, top_labels, merged_programs)

    if random_subset_labels:
        rng = np.random.default_rng(RANDOM_SEED)
        new_idx = []
        for i, genes in enumerate(merged_programs, start=1):
            if len(genes) <= 6:
                sampled = genes
            else:
                sampled = sorted(rng.choice(genes, size=6, replace=False).tolist())
            new_idx.append(f"Gene set {i}: {', '.join(sampled)}...")
        df.index = new_idx

    g = sns.clustermap(
        df,
        cmap="mako",
        linewidths=0.05,
        figsize=(14, 10),
        cbar_kws={"label": "Avg. expr. of gene sets (normalized to [0,1])"},
    )
    g.fig.suptitle(
        f"{fig_tag}: Cell-type specific activation among GenePT-derived gene programs "
        f"(size > 10, similarity threshold {threshold})",
        y=1.02,
    )
    if fig_tag.lower().startswith("fig 2(g)"):
        g.ax_heatmap.text(
            -0.24,
            1.08,
            "g",
            transform=g.ax_heatmap.transAxes,
            fontsize=20,
            fontweight="bold",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    g.savefig(out_dir / out_name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(g.fig)


def compute_umap(X):
    pca = PCA(n_components=min(50, X.shape[1]), random_state=RANDOM_SEED, svd_solver="full")
    X_pca = pca.fit_transform(X)
    return umap.UMAP(min_dist=0.4, spread=1.0, random_state=RANDOM_SEED).fit_transform(X_pca), X_pca


def knn_confusion(X, y):
    y = np.asarray(y)
    stratify = y if np.unique(y).shape[0] > 1 else None
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
    )
    clf = KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    classes = np.unique(np.concatenate([y[test_idx], pred]))
    cm = confusion_matrix(y[test_idx], pred, labels=classes, normalize="true")
    return classes, cm


def plot_umap(ax, embedding, labels, title):
    labels = pd.Series(labels).astype(str)
    uniq = labels.unique()
    palette = sns.color_palette("husl", len(uniq))
    for i, val in enumerate(uniq):
        mask = labels == val
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=3,
            alpha=0.35,
            color=palette[i],
            linewidths=0,
            label=val,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", markerscale=2, frameon=True)


def build_model_grid_like_panel(config: dict, root: Path, out_dir: Path):
    ds_cfg = config["datasets"]["aorta"]
    adata = sc.read_h5ad(root / ds_cfg["data_path"])
    phenotype_key = ds_cfg["phenotype_key"]
    celltype_key = ds_cfg["celltype_key"]
    if phenotype_key not in adata.obs or celltype_key not in adata.obs:
        raise ValueError("Aorta labels are required for model-grid-like panel.")

    expr = prepare_expr(adata)
    X_expr = dense_array(expr.X)
    umap_expr, X_expr_pca = compute_umap(X_expr)

    gene_names = (
        expr.var["gene_name"].astype(str).tolist()
        if "gene_name" in expr.var.columns
        else expr.var.index.astype(str).tolist()
    )
    genept_path = root / config["embeddings"]["genept_w"]["gene_embedding_path"]
    X_genept = load_genept_w_embeddings(genept_path, gene_names, expr.X)
    umap_genept, _ = compute_umap(X_genept)

    phenotype = adata.obs[phenotype_key].astype(str).to_numpy()
    celltype = adata.obs[celltype_key].astype(str).to_numpy()

    expr_classes, cm_expr = knn_confusion(X_expr_pca, celltype)
    genept_classes, cm_genept = knn_confusion(X_genept, celltype)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_umap(axes[0, 0], umap_expr, phenotype, "Original scRNA-seq data (UMAP, phenotype-colored)")
    plot_umap(axes[0, 1], umap_genept, phenotype, "GenePT-w (UMAP, phenotype-colored)")
    sns.heatmap(cm_expr, cmap="Blues", ax=axes[1, 0], vmin=0.0, vmax=1.0, cbar=False)
    axes[1, 0].set_title("Original scRNA-seq data: 10-NN cell-type confusion")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_xticks(np.arange(len(expr_classes)) + 0.5)
    axes[1, 0].set_yticks(np.arange(len(expr_classes)) + 0.5)
    axes[1, 0].set_xticklabels(expr_classes, rotation=90)
    axes[1, 0].set_yticklabels(expr_classes, rotation=0)
    sns.heatmap(cm_genept, cmap="Blues", ax=axes[1, 1], vmin=0.0, vmax=1.0, cbar=True)
    axes[1, 1].set_title("GenePT-w: 10-NN cell-type confusion")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")
    axes[1, 1].set_xticks(np.arange(len(genept_classes)) + 0.5)
    axes[1, 1].set_yticks(np.arange(len(genept_classes)) + 0.5)
    axes[1, 1].set_xticklabels(genept_classes, rotation=90)
    axes[1, 1].set_yticklabels(genept_classes, rotation=0)
    fig.suptitle(
        "Model-grid-like panel (local data): Original scRNA-seq data vs GenePT-w\n"
        "Unavailable columns (Geneformer/scGPT/fine-tuned) intentionally omitted",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "model_grid_like_aorta_original_vs_geneptw.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "benchmarks" / "cell_level_config.yaml")
    out_dir = root / "benchmarks" / "outputs" / "extra_figures" / "paperstyle"
    apply_plot_style()
    build_gene_program_heatmap(
        config,
        root,
        out_dir,
        threshold=0.9,
        fig_tag="Fig 2(g)-like",
        out_name="fig2g_like_gene_set_activation.png",
        max_programs=22,
        random_subset_labels=True,
    )
    build_gene_program_heatmap(
        config,
        root,
        out_dir,
        threshold=0.9,
        fig_tag="B4",
        out_name="b4_like_gene_set_activation.png",
    )
    build_gene_program_heatmap(
        config,
        root,
        out_dir,
        threshold=0.7,
        fig_tag="B5",
        out_name="b5_like_gene_set_activation.png",
    )
    build_model_grid_like_panel(config, root, out_dir)
    print(f"Wrote paper-style figures to {out_dir}")


if __name__ == "__main__":
    main()
