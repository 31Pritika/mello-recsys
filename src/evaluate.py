"""
Phase 1 Evaluation: SVD vs popularity baseline.

Metric: precision@10
  - 80/20 random split on per-user ratings
  - SVD refit on TRAIN only
  - A hit = recommended movie user rated >= 4.0 in TEST
  - Skip users who have no liked (>= 4.0) movies in test set
  - Popularity baseline: top movies by liked-count in train, backfilled to exactly K
    unseen movies per user (identical pool logic to Phase 2 for fair comparison)
  - 95% bootstrap CI (1000 resamples over users)

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
    N_COMPONENTS,
    RATINGS_CLEAN_PARQUET,
    SVD_SEED,
    TOP_K,
)


def precision_at_k(recommended: list[int], liked_test: set[int], k: int = 10) -> float:
    hits = sum(1 for m in recommended[:k] if m in liked_test)
    return hits / k


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) via bootstrap over users."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), lo, hi


def main(full: bool = False) -> None:
    print("=" * 60)
    print("PHASE 1 EVALUATION")
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

    # Build index mappings from full rating set so movie_ids are stable
    movies = sorted(ratings["movieId_tmdb"].unique())
    users  = sorted(ratings["userId"].unique())
    user_to_idx  = {u: i for i, u in enumerate(users)}
    movie_to_idx = {m: j for j, m in enumerate(movies)}
    n_users  = len(users)
    n_movies = len(movies)

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

    # Popularity baseline:
    # - Count movies in train rated >= LIKE_THRESHOLD
    # - Rank them; for each user backfill up to TOP_K movies they have NOT yet rated in train.
    # This guarantees every user gets exactly K candidates (or fewer if catalog is tiny).
    print("Computing popularity baseline …")
    liked_train = train[train["rating"] >= LIKE_THRESHOLD]
    pop_col_counts = (
        liked_train["movieId_tmdb"]
        .map(movie_to_idx)
        .dropna()
        .astype(int)
        .value_counts()
    )
    # Sorted array of column indices by descending liked-count
    pop_ranked_cols: np.ndarray = pop_col_counts.index.to_numpy()  # already sorted desc

    # Determine test users to evaluate
    all_test_users = [u for u in test["userId"].unique() if u in user_to_idx]
    if not full:
        rng2 = np.random.default_rng(EVAL_SEED + 1)
        chosen = list(rng2.choice(all_test_users, size=min(EVAL_N_USERS, len(all_test_users)), replace=False))
        test_eval_users = chosen
    else:
        test_eval_users = all_test_users

    print(f"Evaluating on {len(test_eval_users):,} test users …")

    # Build per-user test liked sets (column indices) and train rated sets
    test_liked: dict[int, set[int]] = {}
    for uid, grp in test[test["userId"].isin(set(test_eval_users))].groupby("userId"):
        liked = set(
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        )
        if liked:
            test_liked[uid] = liked

    train_rated: dict[int, set[int]] = {}
    for uid, grp in train[train["userId"].isin(set(test_eval_users))].groupby("userId"):
        rated_cols = set(movie_to_idx[m] for m in grp["movieId_tmdb"] if m in movie_to_idx)
        train_rated[uid] = rated_cols

    # Evaluate
    svd_precisions: list[float] = []
    pop_precisions: list[float] = []

    for uid in test_liked:
        u_idx = user_to_idx[uid]
        liked_test_cols = test_liked[uid]
        rated_train_cols = train_rated.get(uid, set())

        # SVD recommendations — mask train-rated, take top-k
        scores = (user_factors[u_idx] @ svd.components_).copy()
        if rated_train_cols:
            scores[list(rated_train_cols)] = -np.inf
        top_k_idx = np.argpartition(scores, -TOP_K)[-TOP_K:]
        svd_recs = top_k_idx[np.argsort(scores[top_k_idx])[::-1]].tolist()
        svd_precisions.append(precision_at_k(svd_recs, liked_test_cols, TOP_K))

        # Popularity: backfill to exactly TOP_K unseen movies per user
        pop_recs = [c for c in pop_ranked_cols if c not in rated_train_cols][:TOP_K]
        pop_precisions.append(precision_at_k(pop_recs, liked_test_cols, TOP_K))

    bs_rng = np.random.default_rng(EVAL_SEED + 99)
    svd_mean, svd_lo, svd_hi = bootstrap_ci(svd_precisions, rng=bs_rng)
    pop_mean, pop_lo, pop_hi = bootstrap_ci(pop_precisions, rng=bs_rng)

    lift = (svd_mean - pop_mean) / pop_mean * 100 if pop_mean > 0 else float("inf")

    evaluated = len(svd_precisions)
    print("\n" + "=" * 60)
    print(f"Users evaluated : {evaluated:,}")
    print(f"SVD  precision@{TOP_K}: {svd_mean:.4f}  95% CI [{svd_lo:.4f}, {svd_hi:.4f}]")
    print(f"Pop  precision@{TOP_K}: {pop_mean:.4f}  95% CI [{pop_lo:.4f}, {pop_hi:.4f}]")
    print(f"Lift            : {lift:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Evaluate all test users")
    args = parser.parse_args()
    main(full=args.full)
