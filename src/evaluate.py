"""
Evaluate the SVD recommender vs a popularity baseline.

Metric: precision@10
  - 80/20 random split on per-user ratings
  - SVD refit on TRAIN only
  - A hit = recommended movie user rated >= 4.0 in TEST
  - Skip users who have no liked (>= 4.0) movies in test set
  - Also compute popularity baseline (top-10 most-liked in train, recommended globally)

Usage:
    python evaluate.py            # sample EVAL_N_USERS test users
    python evaluate.py --full     # use all test users
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from config import (
    EVAL_N_USERS,
    EVAL_SEED,
    LIKE_THRESHOLD,
    MOVIES_CLEAN_PARQUET,
    N_COMPONENTS,
    RATINGS_CLEAN_PARQUET,
    SVD_SEED,
    TOP_K,
)


def precision_at_k(recommended: list[int], liked_test: set[int], k: int = 10) -> float:
    hits = sum(1 for m in recommended[:k] if m in liked_test)
    return hits / k


def predict_scores_for_user(
    u_idx: int,
    user_factors: np.ndarray,
    svd_components: np.ndarray,
    train_movie_indices: set[int],
    n_movies: int,
    top_k: int = 10,
) -> list[int]:
    """Predict scores for one user; mask already-rated movies; return top-k movie indices."""
    scores = user_factors[u_idx] @ svd_components  # shape (n_movies,)
    # Mask already-rated
    scores[list(train_movie_indices)] = -np.inf
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return top_indices.tolist()


def main(full: bool = False) -> None:
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Load clean data
    print("Loading clean data …")
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)

    # 80/20 split per user
    print("Splitting 80/20 per user …")
    rng = np.random.default_rng(EVAL_SEED)
    train_rows, test_rows = [], []
    for uid, grp in ratings.groupby("userId"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * 0.8))
        train_rows.append(grp.loc[idx[:cut]])
        test_rows.append(grp.loc[idx[cut:]])

    train = pd.concat(train_rows, ignore_index=True)
    test  = pd.concat(test_rows,  ignore_index=True)
    print(f"  Train: {len(train):,} | Test: {len(test):,}")

    # Build index mappings from TRAIN
    users  = sorted(train["userId"].unique())
    movies = sorted(train["movieId_tmdb"].unique())
    user_to_idx  = {u: i for i, u in enumerate(users)}
    movie_to_idx = {m: j for j, m in enumerate(movies)}
    n_users, n_movies = len(users), len(movies)

    # Build train sparse matrix
    print("Building train sparse matrix …")
    row  = train["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
    col  = train["movieId_tmdb"].map(movie_to_idx).to_numpy(dtype=np.int32)
    data = train["rating"].to_numpy(dtype=np.float32)
    valid = (row >= 0) & (col >= 0)
    train_matrix = csr_matrix(
        (data[valid], (row[valid], col[valid])),
        shape=(n_users, n_movies),
        dtype=np.float32,
    )

    # Fit SVD on train only
    print(f"Fitting TruncatedSVD (n_components={N_COMPONENTS}) on train …")
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    user_factors = svd.fit_transform(train_matrix)

    # Popularity baseline: top-10 most-liked movies in train (rated >= 4.0), by count
    print("Computing popularity baseline …")
    liked_train = train[train["rating"] >= LIKE_THRESHOLD]
    pop_counts  = liked_train["movieId_tmdb"].value_counts()
    # Get top-k popular movie column indices
    top_pop_tmdb = pop_counts.index[:TOP_K].tolist()
    top_pop_cols = [movie_to_idx[m] for m in top_pop_tmdb if m in movie_to_idx]

    # Determine test users to evaluate
    test_users = [u for u in test["userId"].unique() if u in user_to_idx]
    if not full:
        rng2 = np.random.default_rng(EVAL_SEED + 1)
        rng2.shuffle(test_users)
        test_users = test_users[:EVAL_N_USERS]

    print(f"Evaluating on {len(test_users):,} test users …")

    # Build per-user test liked sets and train rated sets
    test_liked: dict[int, set[int]] = {}
    for uid, grp in test[test["userId"].isin(set(test_users))].groupby("userId"):
        liked = set(grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"].tolist())
        if liked:
            test_liked[uid] = liked

    train_rated: dict[int, set[int]] = {}
    for uid, grp in train[train["userId"].isin(set(test_users))].groupby("userId"):
        rated_cols = set(movie_to_idx[m] for m in grp["movieId_tmdb"] if m in movie_to_idx)
        train_rated[uid] = rated_cols

    # Evaluate
    svd_precisions = []
    pop_precisions = []
    skipped = 0

    for uid in test_liked:
        u_idx = user_to_idx[uid]
        liked_set_tmdb = test_liked[uid]
        liked_set_cols = set(movie_to_idx[m] for m in liked_set_tmdb if m in movie_to_idx)
        rated_cols = train_rated.get(uid, set())

        # SVD recommendations
        svd_recs = predict_scores_for_user(
            u_idx, user_factors, svd.components_, rated_cols, n_movies, TOP_K
        )
        svd_liked_cols = set(movie_to_idx[m] for m in liked_set_tmdb if m in movie_to_idx)
        svd_p = precision_at_k(svd_recs, svd_liked_cols, TOP_K)
        svd_precisions.append(svd_p)

        # Popularity recommendations (exclude already rated)
        pop_recs = [c for c in top_pop_cols if c not in rated_cols][:TOP_K]
        pop_p = precision_at_k(pop_recs, liked_set_cols, TOP_K)
        pop_precisions.append(pop_p)

    evaluated = len(svd_precisions)
    svd_p10 = float(np.mean(svd_precisions)) if svd_precisions else 0.0
    pop_p10 = float(np.mean(pop_precisions)) if pop_precisions else 0.0
    lift = (svd_p10 - pop_p10) / pop_p10 * 100 if pop_p10 > 0 else float("inf")

    print("\n" + "=" * 40)
    print(f"Users evaluated : {evaluated:,} (skipped {len(test_liked) - evaluated + (len(test_users) - len(test_liked)):,} with no liked test items)")
    print(f"SVD  precision@{TOP_K}: {svd_p10:.4f}")
    print(f"Pop  precision@{TOP_K}: {pop_p10:.4f}")
    print(f"Lift            : {lift:+.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Evaluate all test users")
    args = parser.parse_args()
    main(full=args.full)
