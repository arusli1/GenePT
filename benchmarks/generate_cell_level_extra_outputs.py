import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


RANDOM_SEED = 2023
FIG_DPI = 300
FONT_SCALE = 0.95


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
    return cell_emb / norms[:, None]


def get_labels(adata, key):
    if key is None or key not in adata.obs:
        return None
    return adata.obs[key].astype(str).to_numpy()


def apply_plot_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def knn_predictions(cell_emb: np.ndarray, labels, test_size: float = 0.20):
    labels = np.asarray(labels)
    stratify = None
    if len(np.unique(labels)) > 1:
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() >= 2:
            stratify = labels
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(cell_emb[train_idx])
    neighbors = nn.kneighbors(cell_emb[test_idx], return_distance=False)
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    preds = [pd.Series(y_train[n]).mode().iloc[0] for n in neighbors]
    return y_test, np.asarray(preds)


def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    size = max(6, min(18, 0.5 * len(labels)))
    plt.figure(figsize=(size, size))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    return labels, cm


def plot_label_distribution(labels, out_path: Path, title: str):
    values = pd.Series(labels).value_counts()
    plt.figure(figsize=(max(8, 0.3 * len(values)), 4))
    sns.barplot(x=values.index, y=values.values, color="#2F5D8A")
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_per_class_accuracy(labels, cm, out_path: Path, title: str):
    acc = np.diag(cm)
    plt.figure(figsize=(max(8, 0.3 * len(labels)), 4))
    sns.barplot(x=labels, y=acc, color="#2E8B57")
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Per-class accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_metrics_heatmap(results_path: Path, out_path: Path):
    df = pd.read_csv(results_path)
    df = df[df["embedding"] == "genept_w"].copy()
    metrics = [
        "celltype_ari",
        "celltype_ami",
        "celltype_asw",
        "phenotype_ari",
        "phenotype_ami",
        "phenotype_asw",
    ]
    df = df[["dataset"] + metrics].set_index("dataset")
    df = df.dropna(how="all")
    if df.empty:
        return
    plt.figure(figsize=(10, max(4, 0.5 * len(df))))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("GenePT-w metrics heatmap")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "benchmarks" / "cell_level_config.yaml")
    out_dir = root / "benchmarks" / "outputs" / "extra_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    warnings.filterwarnings(
        "ignore",
        message="Observation names are not unique",
        category=UserWarning,
    )

    genept_w_path = root / config["embeddings"]["genept_w"]["gene_embedding_path"]

    for ds_name, ds_cfg in config["datasets"].items():
        if not ds_cfg.get("enabled", True):
            continue
        data_path = root / ds_cfg["data_path"]
        if not data_path.exists():
            continue
        adata = sc.read_h5ad(data_path)
        adata.obs_names_make_unique()
        expr = adata.raw.to_adata() if adata.raw is not None else adata.copy()
        if expr.uns.get("log1p") is None:
            sc.pp.normalize_total(expr, target_sum=1e4)
            sc.pp.log1p(expr)
        gene_names = (
            expr.var["gene_name"].astype(str).tolist()
            if "gene_name" in expr.var.columns
            else expr.var.index.astype(str).tolist()
        )
        genept_w = sanitize_embeddings(load_genept_w_embeddings(genept_w_path, gene_names, expr.X))

        celltype_labels = get_labels(adata, ds_cfg.get("celltype_key"))
        phenotype_labels = get_labels(adata, ds_cfg.get("phenotype_key"))

        if celltype_labels is not None:
            y_true, y_pred = knn_predictions(genept_w, celltype_labels)
            labels, cm = plot_confusion(
                y_true,
                y_pred,
                out_dir / ds_name / "confusion_celltype_genept_w.png",
                f"{ds_name} celltype kNN confusion (GenePT-w)",
            )
            plot_label_distribution(
                celltype_labels,
                out_dir / ds_name / "distribution_celltype.png",
                f"{ds_name} celltype distribution",
            )
            plot_per_class_accuracy(
                labels,
                cm,
                out_dir / ds_name / "per_class_accuracy_celltype.png",
                f"{ds_name} celltype per-class accuracy (GenePT-w)",
            )
        if phenotype_labels is not None:
            y_true, y_pred = knn_predictions(genept_w, phenotype_labels)
            labels, cm = plot_confusion(
                y_true,
                y_pred,
                out_dir / ds_name / "confusion_phenotype_genept_w.png",
                f"{ds_name} phenotype kNN confusion (GenePT-w)",
            )
            plot_label_distribution(
                phenotype_labels,
                out_dir / ds_name / "distribution_phenotype.png",
                f"{ds_name} phenotype distribution",
            )
            plot_per_class_accuracy(
                labels,
                cm,
                out_dir / ds_name / "per_class_accuracy_phenotype.png",
                f"{ds_name} phenotype per-class accuracy (GenePT-w)",
            )

    plot_metrics_heatmap(
        root / "benchmarks" / "outputs" / "cell_level_benchmark.csv",
        out_dir / "genept_w_metrics_heatmap.png",
    )


if __name__ == "__main__":
    main()
