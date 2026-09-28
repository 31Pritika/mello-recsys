"""
Train, save all artifacts to ./model_artifacts/.

Runs the full cleaned pipeline once:
  1. build_clean_dataset()
  2. build_sparse_matrix()
  3. TruncatedSVD → user_factors
  4. Save everything
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD

from config import (
    ARTIFACTS_DIR,
    CLUSTER_LABELS_PATH,
    IDX_TO_USER_PATH,
    KMEANS_K,
    KMEANS_PATH,
    KMEANS_SEED,
    MOVIE_IDS_PATH,
    MOVIE_TO_IDX_PATH,
    MOVIES_CLEAN_PARQUET,
    N_COMPONENTS,
    RATINGS_CLEAN_PARQUET,
    SVD_PATH,
    SVD_SEED,
    USER_FACTORS_PATH,
    USER_TO_IDX_PATH,
)
from data_loading import build_clean_dataset, build_sparse_matrix


def main() -> None:
    print("=" * 60)
    print("TRAIN & SAVE")
    print("=" * 60)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    ratings, movies_clean = build_clean_dataset()

    # ── 2. Save clean parquets ─────────────────────────────────────────────────
    print("\nSaving clean parquets …")
    ratings.to_parquet(RATINGS_CLEAN_PARQUET, index=False)
    movies_clean.to_parquet(MOVIES_CLEAN_PARQUET, index=False)
    print(f"  → {RATINGS_CLEAN_PARQUET}")
    print(f"  → {MOVIES_CLEAN_PARQUET}")

    # ── 3. Sparse matrix ───────────────────────────────────────────────────────
    print("\nBuilding sparse matrix …")
    matrix, user_to_idx, idx_to_user, movie_to_idx, movie_ids = build_sparse_matrix(ratings)

    # ── 4. SVD ─────────────────────────────────────────────────────────────────
    print(f"\nFitting TruncatedSVD (n_components={N_COMPONENTS}) …")
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    user_factors = svd.fit_transform(matrix)  # shape (n_users, n_components)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  → explained variance: {explained:.3%}")

    # ── 5. KMeans on user factors ──────────────────────────────────────────────
    print(f"\nFitting KMeans (k={KMEANS_K}) …")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(user_factors)
    for cid in range(KMEANS_K):
        size = (cluster_labels == cid).sum()
        print(f"  Cluster {cid}: {size:,} users")

    # ── 6. Save artifacts ──────────────────────────────────────────────────────
    print("\nSaving artifacts …")
    with open(SVD_PATH, "wb") as f:
        pickle.dump(svd, f, protocol=5)
    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(kmeans, f, protocol=5)
    with open(USER_TO_IDX_PATH, "wb") as f:
        pickle.dump(user_to_idx, f, protocol=5)
    with open(IDX_TO_USER_PATH, "wb") as f:
        pickle.dump(idx_to_user, f, protocol=5)
    with open(MOVIE_TO_IDX_PATH, "wb") as f:
        pickle.dump(movie_to_idx, f, protocol=5)
    np.save(USER_FACTORS_PATH, user_factors)
    np.save(CLUSTER_LABELS_PATH, cluster_labels)
    np.save(MOVIE_IDS_PATH, movie_ids)

    print(f"  SVD          → {SVD_PATH}")
    print(f"  KMeans       → {KMEANS_PATH}")
    print(f"  user_factors → {USER_FACTORS_PATH}")
    print(f"  cluster_lbl  → {CLUSTER_LABELS_PATH}")
    print(f"  user_to_idx  → {USER_TO_IDX_PATH}")
    print(f"  idx_to_user  → {IDX_TO_USER_PATH}")
    print(f"  movie_to_idx → {MOVIE_TO_IDX_PATH}")
    print(f"  movie_ids    → {MOVIE_IDS_PATH}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
