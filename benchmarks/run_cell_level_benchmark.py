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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
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


def clustering_metrics_k(cell_emb: np.ndarray, labels, k: int):
    if labels is None:
        return None, None
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        batch_size=20,
    )
    kmeans.fit(cell_emb)
    ari = sklearn.metrics.adjusted_rand_score(kmeans.labels_, labels)
    ami = sklearn.metrics.adjusted_mutual_info_score(kmeans.labels_, labels)
    return ari, ami


def get_train_test_indices(labels, test_size: float = 0.20):
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
    return train_idx, test_idx


def knn_metrics_from_split(cell_emb: np.ndarray, labels, train_idx, test_idx):
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(cell_emb[train_idx])
    neighbors = nn.kneighbors(cell_emb[test_idx], return_distance=False)
    y_train = np.asarray(labels)[train_idx]
    y_test = np.asarray(labels)[test_idx]
    preds = [pd.Series(y_train[n]).mode().iloc[0] for n in neighbors]
    acc = sklearn.metrics.accuracy_score(y_test, preds)
    prec, rec, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    return acc, prec, rec, f1


def knn_metrics(cell_emb: np.ndarray, labels):
    if labels is None:
        return None, None, None, None
    train_idx, test_idx = get_train_test_indices(labels, test_size=0.20)
    return knn_metrics_from_split(cell_emb, labels, train_idx, test_idx)


def knn_ensemble_metrics(embeddings, labels, train_idx, test_idx):
    if labels is None or not embeddings:
        return None, None, None, None
    y_train = np.asarray(labels)[train_idx]
    y_test = np.asarray(labels)[test_idx]
    all_neighbor_labels = []
    for emb in embeddings:
        nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        nn.fit(emb[train_idx])
        neighbors = nn.kneighbors(emb[test_idx], return_distance=False)
        neighbor_labels = [y_train[n] for n in neighbors]
        all_neighbor_labels.append(neighbor_labels)
    final_preds = []
    for i in range(len(test_idx)):
        votes = np.concatenate([labels_list[i] for labels_list in all_neighbor_labels])
        final_preds.append(pd.Series(votes).mode().iloc[0])
    acc = sklearn.metrics.accuracy_score(y_test, final_preds)
    prec, rec, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_test, final_preds, average="macro", zero_division=0
    )
    return acc, prec, rec, f1


def pca_kmeans_ari(X, labels, k: int, pca_n: int = 50):
    if labels is None:
        return None
    n_components = min(pca_n, X.shape[0] - 1, X.shape[1])
    if n_components < 2:
        return None
    if sparse.issparse(X):
        reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        X_reduced = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=n_components, random_state=RANDOM_SEED, svd_solver="full")
        X_reduced = reducer.fit_transform(X)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, batch_size=50)
    kmeans.fit(X_reduced)
    return sklearn.metrics.adjusted_rand_score(kmeans.labels_, labels)


def logreg_metrics(cell_emb: np.ndarray, labels, train_idx, test_idx):
    if labels is None:
        return None, None, None
    X_train = cell_emb[train_idx]
    X_test = cell_emb[test_idx]
    y_train = np.asarray(labels)[train_idx]
    y_test = np.asarray(labels)[test_idx]
    model = LogisticRegression(max_iter=1000, n_jobs=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, preds)
    prec, rec, _f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    return acc, prec, rec


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
        if not ds_cfg.get("enabled", True):
            missing_requirements.append(f"Dataset disabled in config: {ds_name}")
            continue
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
        # scGPT (optional)
        scgpt_emb = None
        scgpt_dim = None
        scgpt_paths = config.get("embeddings", {}).get("scgpt", {}).get(
            "cell_embedding_paths", {}
        )
        scgpt_path = scgpt_paths.get(ds_name)
        if scgpt_path is not None:
            scgpt_path = root / scgpt_path
            if scgpt_path.exists():
                scgpt_emb, scgpt_dim = load_genept_s_embeddings(
                    scgpt_path, adata.n_obs
                )
            else:
                missing_requirements.append(
                    f"scGPT embeddings missing for {ds_name}: {scgpt_path}"
                )
        # Geneformer (optional)
        geneformer_emb = None
        geneformer_dim = None
        geneformer_paths = config.get("embeddings", {}).get("geneformer", {}).get(
            "cell_embedding_paths", {}
        )
        geneformer_path = geneformer_paths.get(ds_name)
        if geneformer_path is not None:
            geneformer_path = root / geneformer_path
            if geneformer_path.exists():
                geneformer_emb, geneformer_dim = load_genept_s_embeddings(
                    geneformer_path, adata.n_obs
                )
            else:
                missing_requirements.append(
                    f"Geneformer embeddings missing for {ds_name}: {geneformer_path}"
                )

        # Section 4.4 metrics (batch-effect + phenotype preservation)
        section44_cfg = ds_cfg.get("section44")
        batch_ari_expr = batch_ari_genept_w = None
        phenotype_ari_k = None
        phenotype_lr_acc = phenotype_lr_prec = phenotype_lr_rec = None
        if section44_cfg:
            pca_n = int(section44_cfg.get("pca_n", 50))
            batch_k = int(section44_cfg.get("batch_k"))
            batch_ari_expr = pca_kmeans_ari(expr.X, patient_labels, batch_k, pca_n=pca_n)
            batch_ari_genept_w = pca_kmeans_ari(
                sanitize_embeddings(genept_w_emb), patient_labels, batch_k, pca_n=pca_n
            )
            phenotype_k = section44_cfg.get("phenotype_k")
            if phenotype_k is not None and phenotype_labels is not None:
                phenotype_ari_k, _ = clustering_metrics_k(
                    sanitize_embeddings(genept_w_emb), phenotype_labels, int(phenotype_k)
                )
            if phenotype_labels is not None:
                ph_train_idx, ph_test_idx = get_train_test_indices(phenotype_labels)
                phenotype_lr_acc, phenotype_lr_prec, phenotype_lr_rec = logreg_metrics(
                    sanitize_embeddings(genept_w_emb),
                    phenotype_labels,
                    ph_train_idx,
                    ph_test_idx,
                )

        embedding_variants = [
            ("genept_w", genept_w_emb, genept_w_dim, genept_w_missing),
        ]
        if genept_s_emb is not None:
            embedding_variants.insert(0, ("genept_s", genept_s_emb, genept_s_dim, "NA"))
        if scgpt_emb is not None:
            embedding_variants.append(("scgpt", scgpt_emb, scgpt_dim, "NA"))
        if geneformer_emb is not None:
            embedding_variants.append(("geneformer", geneformer_emb, geneformer_dim, "NA"))

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

            if ct_labels is not None:
                ct_train_idx, ct_test_idx = get_train_test_indices(ct_labels)
                acc, prec, rec, f1 = knn_metrics_from_split(
                    ct_emb, ct_labels, ct_train_idx, ct_test_idx
                )
            else:
                acc, prec, rec, f1 = (None, None, None, None)
            if phenotype_labels is not None:
                ph_train_idx, ph_test_idx = get_train_test_indices(phenotype_labels)
                ph_acc, ph_prec, ph_rec, ph_f1 = knn_metrics_from_split(
                    cell_emb, phenotype_labels, ph_train_idx, ph_test_idx
                )
            else:
                ph_acc, ph_prec, ph_rec, ph_f1 = (None, None, None, None)

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
                    "batch_patient_ari_expr": batch_ari_expr if emb_name == "genept_w" else None,
                    "batch_patient_ari_genept_w": batch_ari_genept_w if emb_name == "genept_w" else None,
                    "phenotype_ari_k": phenotype_ari_k if emb_name == "genept_w" else None,
                    "phenotype_lr_accuracy": phenotype_lr_acc if emb_name == "genept_w" else None,
                    "phenotype_lr_precision": phenotype_lr_prec if emb_name == "genept_w" else None,
                    "phenotype_lr_recall": phenotype_lr_rec if emb_name == "genept_w" else None,
                }
            )

        # 10-NN ensemble (GenePT-w + GenePT-s + scGPT)
        ensemble_acc = ensemble_prec = ensemble_rec = ensemble_f1 = None
        if ct_labels is not None:
            ensemble_inputs = []
            for emb in (genept_w_emb, genept_s_emb, scgpt_emb):
                if emb is not None:
                    ensemble_inputs.append(sanitize_embeddings(emb))
            if len(ensemble_inputs) >= 2:
                ct_train_idx, ct_test_idx = get_train_test_indices(ct_labels)
                ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1 = knn_ensemble_metrics(
                    ensemble_inputs, ct_labels, ct_train_idx, ct_test_idx
                )

        results.append(
            {
                "dataset": ds_name,
                "embedding": "ensemble",
                "accuracy": ensemble_acc,
                "precision": ensemble_prec,
                "recall": ensemble_rec,
                "f1": ensemble_f1,
                "phenotype_accuracy": None,
                "phenotype_precision": None,
                "phenotype_recall": None,
                "phenotype_f1": None,
                "celltype_ari": None,
                "celltype_ami": None,
                "celltype_asw": None,
                "phenotype_ari": None,
                "phenotype_ami": None,
                "phenotype_asw": None,
                "patient_ari": None,
                "patient_ami": None,
                "patient_asw": None,
                "embed_dim": None,
                "missing_genes": None,
                "batch_patient_ari_expr": None,
                "batch_patient_ari_genept_w": None,
                "phenotype_ari_k": None,
                "phenotype_lr_accuracy": None,
                "phenotype_lr_precision": None,
                "phenotype_lr_recall": None,
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
