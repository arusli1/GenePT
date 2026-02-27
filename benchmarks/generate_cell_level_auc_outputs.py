import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import yaml
from scipy import sparse
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize


RANDOM_SEED = 2023
FIG_DPI = 300


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


def apply_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
        }
    )


def build_scores(X, y):
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
    )
    clf = KNeighborsClassifier(n_neighbors=25, metric="cosine", weights="distance")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    classes = clf.classes_

    # Keep score columns and true-label binarization on exactly the same class order.
    if len(classes) == 2:
        pos_class = classes[1]
        y_true_bin = (y_test == pos_class).astype(int)
        y_score = probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), probs.ravel())
    return fpr, tpr, precision, recall


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks" / "cell_level_config.yaml").read_text())
    genept_path = root / cfg["embeddings"]["genept_w"]["gene_embedding_path"]
    targets = [
        ("aorta", cfg["datasets"]["aorta"]["phenotype_key"], "Aorta phenotype"),
        ("cardiomyocyte", cfg["datasets"]["cardiomyocyte"]["phenotype_key"], "Cardiomyocyte disease"),
    ]

    curves = []
    for ds_name, label_key, display in targets:
        ds = cfg["datasets"][ds_name]
        adata = sc.read_h5ad(root / ds["data_path"])
        expr = adata.raw.to_adata() if adata.raw is not None else adata.copy()
        if expr.uns.get("log1p") is None:
            sc.pp.normalize_total(expr, target_sum=1e4)
            sc.pp.log1p(expr)
        gene_names = (
            expr.var["gene_name"].astype(str).tolist()
            if "gene_name" in expr.var.columns
            else expr.var.index.astype(str).tolist()
        )
        X = load_genept_w_embeddings(genept_path, gene_names, expr.X)
        y = adata.obs[label_key].astype(str).to_numpy()
        fpr, tpr, precision, recall = build_scores(X, y)
        curves.append((display, fpr, tpr, precision, recall))

    out_dir = root / "benchmarks" / "outputs" / "extra_figures" / "auc"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for i, (name, fpr, tpr, _, _) in enumerate(curves):
        axes[i].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        axes[i].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, linewidth=1.2)
        axes[i].set_xlabel("False Positive Rate")
        axes[i].set_ylabel("True Positive Rate")
        axes[i].set_title(f"ROC curve: {name}")
        axes[i].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_aorta_cardiomyocyte_genept_w.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for i, (name, _, _, precision, recall) in enumerate(curves):
        axes[i].plot(recall, precision, label=name)
        axes[i].set_xlabel("Recall")
        axes[i].set_ylabel("Precision")
        axes[i].set_title(f"PR curve: {name}")
        axes[i].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_aorta_cardiomyocyte_genept_w.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
