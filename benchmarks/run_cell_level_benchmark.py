import json
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns
import umap

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


RANDOM_SEED = 2023


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_genept_w_embeddings(gene_embedding_path: Path, gene_names, X):
    with gene_embedding_path.open("rb") as fp:
        gene_embeddings = pickle.load(fp)
    sample = next(iter(gene_embeddings.values()))
    embed_dim = np.asarray(sample, dtype=float).flatten().shape[0]
    lookup = np.zeros((len(gene_names), embed_dim))
    missing = 0
    for i, gene in enumerate(gene_names):
        val = gene_embeddings.get(gene)
        if val is None:
            missing += 1
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
    return cell_emb, embed_dim, missing


def load_genept_s_embeddings(path: Path, n_obs: int):
    if not path.exists():
        raise FileNotFoundError(
            f"GenePT-s embedding file not found: {path}. Please download it."
        )
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        npz = np.load(path)
        data = npz["embeddings"] if "embeddings" in npz else npz[list(npz.files)[0]]
    elif path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fp:
            data = pickle.load(fp)
        data = np.asarray(data)
    else:
        raise ValueError(f"Unsupported GenePT-s format: {path.suffix}")
    data = np.asarray(data, dtype=np.float32)
    if data.shape[0] != n_obs:
        raise ValueError(
            f"GenePT-s rows {data.shape[0]} do not match cells {n_obs} for {path}"
        )
    return data, data.shape[1]


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


def plot_umap(embedding, labels, title, out_path: Path):
    plt.figure(figsize=(6, 5))
    labels_unique = pd.Series(labels).unique()
    colors = sns.color_palette("husl", len(labels_unique))
    for i, label_name in enumerate(labels_unique):
        mask = labels == label_name
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=2,
            label=label_name,
            color=colors[i],
            alpha=0.3,
        )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="best", markerscale=2, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def clustering_metrics(cell_emb: np.ndarray, labels):
    if labels is None:
        return None, None
    kmeans = MiniBatchKMeans(
        n_clusters=len(pd.Series(labels).unique()),
        random_state=RANDOM_SEED,
        batch_size=20,
    )
    kmeans.fit(cell_emb)
    ari = sklearn.metrics.adjusted_rand_score(kmeans.labels_, labels)
    ami = sklearn.metrics.adjusted_mutual_info_score(kmeans.labels_, labels)
    return ari, ami


def knn_metrics(cell_emb: np.ndarray, labels):
    if labels is None:
        return None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(
        cell_emb, labels, test_size=0.20, random_state=RANDOM_SEED
    )
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(X_train)
    neighbors = nn.kneighbors(X_test, return_distance=False)
    preds = [pd.Series(y_train[n]).mode().iloc[0] for n in neighbors]
    acc = sklearn.metrics.accuracy_score(y_test, preds)
    prec, rec, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    return acc, prec, rec, f1


def silhouette_metrics(cell_emb: np.ndarray, labels, sample_size: int = 5000):
    if labels is None:
        return None
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    if cell_emb.shape[0] > sample_size:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(cell_emb.shape[0], size=sample_size, replace=False)
        emb_sample = cell_emb[idx]
        label_sample = labels[idx]
    else:
        emb_sample = cell_emb
        label_sample = labels
    return silhouette_score(emb_sample, label_sample, metric="cosine")


def get_labels(adata, key):
    if key is None:
        return None
    if key not in adata.obs:
        raise KeyError(f"Label key '{key}' not found in adata.obs")
    labels = adata.obs[key].astype(str).to_numpy()
    return labels


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "benchmarks" / "cell_level_config.yaml"
    config = load_config(config_path)

    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in matmul",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="overflow encountered in matmul",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in matmul",
        category=RuntimeWarning,
    )

    out_dir = root / config["outputs"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    missing_requirements = []

    for ds_name, ds_cfg in config["datasets"].items():
        data_path = root / ds_cfg["data_path"]
        if not data_path.exists():
            missing_requirements.append(
                f"Dataset missing for {ds_name}: {data_path}"
            )
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Observation names are not unique",
                category=UserWarning,
            )
            adata = sc.read_h5ad(data_path)
        adata.obs_names_make_unique()
        expr = adata.raw.to_adata() if adata.raw is not None else adata.copy()
        if "gene_name" in expr.var.columns:
            gene_names = expr.var["gene_name"].astype(str).tolist()
        else:
            gene_names = expr.var.index.astype(str).tolist()
        # Match GenePT preprocessing: normalize to 1e4 counts + log1p
        if expr.uns.get("log1p") is None:
            sc.pp.normalize_total(expr, target_sum=1e4)
            sc.pp.log1p(expr)
        X = expr.X

        celltype_labels = get_labels(adata, ds_cfg.get("celltype_key"))
        phenotype_labels = get_labels(adata, ds_cfg.get("phenotype_key"))
        patient_labels = get_labels(adata, ds_cfg.get("patient_key"))

        if celltype_labels is not None:
            known_idx = celltype_labels != "Unknown"
        else:
            known_idx = None

        # GenePT-w
        genept_w_path = root / config["embeddings"]["genept_w"]["gene_embedding_path"]
        genept_w_emb, genept_w_dim, genept_w_missing = load_genept_w_embeddings(
            genept_w_path, gene_names, X
        )
        # GenePT-s
        genept_s_path = root / config["embeddings"]["genept_s"]["cell_embedding_paths"][
            ds_name
        ]
        genept_s_emb = None
        genept_s_dim = None
        if genept_s_path.exists():
            genept_s_emb, genept_s_dim = load_genept_s_embeddings(
                genept_s_path, adata.n_obs
            )
        else:
            missing_requirements.append(
                f"GenePT-s missing for {ds_name}: {genept_s_path}"
            )

        embedding_variants = [
            ("genept_w", genept_w_emb, genept_w_dim, genept_w_missing),
        ]
        if genept_s_emb is not None:
            embedding_variants.insert(0, ("genept_s", genept_s_emb, genept_s_dim, "NA"))

        for emb_name, cell_emb, emb_dim, missing in embedding_variants:
            cell_emb = sanitize_embeddings(cell_emb)
            umap_emb = compute_umap(cell_emb)
            if phenotype_labels is not None:
                plot_umap(
                    umap_emb,
                    phenotype_labels,
                    f"{ds_name} {emb_name} phenotype",
                    out_dir / "umap" / ds_name / emb_name / "phenotype.png",
                )
            if celltype_labels is not None:
                plot_umap(
                    umap_emb,
                    celltype_labels,
                    f"{ds_name} {emb_name} celltype",
                    out_dir / "umap" / ds_name / emb_name / "celltype.png",
                )
            if patient_labels is not None:
                plot_umap(
                    umap_emb,
                    patient_labels,
                    f"{ds_name} {emb_name} patient",
                    out_dir / "umap" / ds_name / emb_name / "patient.png",
                )
            if celltype_labels is not None:
                ct_labels = celltype_labels if known_idx is None else celltype_labels[known_idx]
                ct_emb = cell_emb if known_idx is None else cell_emb[known_idx]
            else:
                ct_labels = None
                ct_emb = None

            celltype_ari, celltype_ami = clustering_metrics(
                ct_emb, ct_labels
            ) if ct_labels is not None else (None, None)
            phenotype_ari, phenotype_ami = clustering_metrics(
                cell_emb, phenotype_labels
            )
            patient_ari, patient_ami = clustering_metrics(
                cell_emb, patient_labels
            )
            celltype_asw = silhouette_metrics(ct_emb, ct_labels)
            phenotype_asw = silhouette_metrics(cell_emb, phenotype_labels)
            patient_asw = silhouette_metrics(cell_emb, patient_labels)

            acc, prec, rec, f1 = knn_metrics(ct_emb, ct_labels)
            ph_acc, ph_prec, ph_rec, ph_f1 = knn_metrics(cell_emb, phenotype_labels)

            results.append(
                {
                    "dataset": ds_name,
                    "embedding": emb_name,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "phenotype_accuracy": ph_acc,
                    "phenotype_precision": ph_prec,
                    "phenotype_recall": ph_rec,
                    "phenotype_f1": ph_f1,
                    "celltype_ari": celltype_ari,
                    "celltype_ami": celltype_ami,
                    "celltype_asw": celltype_asw,
                    "phenotype_ari": phenotype_ari,
                    "phenotype_ami": phenotype_ami,
                    "phenotype_asw": phenotype_asw,
                    "patient_ari": patient_ari,
                    "patient_ami": patient_ami,
                    "patient_asw": patient_asw,
                    "embed_dim": emb_dim,
                    "missing_genes": missing,
                }
            )

    df = pd.DataFrame(results)
    df_rounded = df.copy()
    float_cols = df_rounded.select_dtypes(include=["float"]).columns
    df_rounded[float_cols] = df_rounded[float_cols].round(3)

    csv_path = out_dir / f"{config['outputs']['results_basename']}.csv"
    md_path = out_dir / f"{config['outputs']['results_basename']}.md"
    json_path = out_dir / f"{config['outputs']['results_basename']}.json"
    df_rounded.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown_table(df_rounded))
    json_path.write_text(df_rounded.to_json(orient="records", indent=2))

    if missing_requirements:
        missing_path = out_dir / "missing_requirements.txt"
        missing_path.write_text("\n".join(missing_requirements))

    # Simple bar plots for quick comparisons
    plot_metrics = [
        "accuracy",
        "f1",
        "phenotype_accuracy",
        "phenotype_f1",
        "celltype_ari",
        "phenotype_ari",
        "patient_ari",
    ]
    for metric in plot_metrics:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(10, 4))
        sns.barplot(data=df_rounded, x="dataset", y=metric, hue="embedding")
        plt.title(metric)
        plt.tight_layout()
        out_path = out_dir / "plots" / f"{metric}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"Wrote results to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
