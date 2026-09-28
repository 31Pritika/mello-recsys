"""
Cluster analysis on user factors.

- Runs KMeans for k=2..10, prints inertia and per-step drop (elbow analysis).
- Fits final k=KMEANS_K, prints cluster sizes.
- For each cluster prints a sample user's top-rated movie titles.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans

from config import (
    CLUSTER_LABELS_PATH,
    IDX_TO_USER_PATH,
    KMEANS_K,
    KMEANS_PATH,
    KMEANS_SEED,
    MOVIES_CLEAN_PARQUET,
    RATINGS_CLEAN_PARQUET,
    USER_FACTORS_PATH,
    USER_TO_IDX_PATH,
)


def main() -> None:
    print("=" * 60)
    print("CLUSTER ANALYSIS")
    print("=" * 60)

    # Load artifacts
    print("Loading artifacts …")
    user_factors: np.ndarray = np.load(USER_FACTORS_PATH)
    with open(USER_TO_IDX_PATH, "rb") as f:
        user_to_idx: dict = pickle.load(f)
    with open(IDX_TO_USER_PATH, "rb") as f:
        idx_to_user: dict = pickle.load(f)

    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    movies = pd.read_parquet(MOVIES_CLEAN_PARQUET)
    id_to_title = dict(zip(movies["id"], movies["title"]))

    n_users = user_factors.shape[0]
    print(f"  {n_users:,} users, {user_factors.shape[1]} SVD components")

    # ── Elbow analysis k=2..10 ──────────────────────────────────────────────────
    print("\nElbow analysis (KMeans inertia for k=2..10):")
    print(f"{'k':>4}  {'inertia':>14}  {'drop':>14}")
    inertias: list[float] = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=5, max_iter=100)
        km.fit(user_factors)
        inertias.append(km.inertia_)
        drop = (inertias[-2] - inertias[-1]) if len(inertias) > 1 else float("nan")
        drop_str = f"{drop:14.1f}" if not (isinstance(drop, float) and np.isnan(drop)) else f"{'—':>14}"
        print(f"  {k:>2}  {km.inertia_:14.1f}  {drop_str}")

    # ── Final KMeans k=KMEANS_K ─────────────────────────────────────────────────
    print(f"\nFitting final KMeans (k={KMEANS_K}) …")
    kmeans = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_SEED, n_init=10)
    cluster_labels: np.ndarray = kmeans.fit_predict(user_factors)

    print("\nCluster sizes:")
    for cid in range(KMEANS_K):
        size = (cluster_labels == cid).sum()
        pct = size / n_users * 100
        print(f"  Cluster {cid}: {size:,} users ({pct:.1f}%)")

    # Save updated artifacts
    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(kmeans, f, protocol=5)
    np.save(CLUSTER_LABELS_PATH, cluster_labels)

    # ── Sample user top-rated movies per cluster ────────────────────────────────
    print("\nSample user top-rated movies per cluster:")
    rng = np.random.default_rng(KMEANS_SEED)

    for cid in range(KMEANS_K):
        members_idx = np.where(cluster_labels == cid)[0]
        sample_idx = rng.choice(members_idx)
        sample_user = idx_to_user[int(sample_idx)]
        user_ratings = ratings[ratings["userId"] == sample_user].copy()
        top = user_ratings.nlargest(5, "rating")
        titles = [id_to_title.get(mid, f"<id={mid}>") for mid in top["movieId_tmdb"]]
        print(f"\n  Cluster {cid} (user {sample_user}):")
        for t in titles:
            print(f"    • {t}")


if __name__ == "__main__":
    main()
