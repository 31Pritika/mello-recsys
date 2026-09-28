"""
Phase 2 Evaluation: Popularity vs SVD vs Content-only vs Hybrid.

- Splits ratings 70/10/20 per user (Train / Val / Test).
- Refits SVD on Train only.
- Computes content profiles from Train (rating >= 4.0) only.
- Tunes alpha in [0.0, 0.1, ..., 1.0] on VALIDATION precision@10 only.
- Evaluates on TEST precision@10 on a 5,000-user sample (or --full for all).
- Breaks down metrics overall, for COLD-START (<= 5 train ratings) and HEAVY (>= 50 train ratings).
- Computes popularity skew: mean log10(train rating count + 1) of recommended items.
- Computes catalog coverage: distinct recommended movies across evaluated users.
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
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    MOVIE_TO_IDX_PATH,
    N_COMPONENTS,
    RATINGS_CLEAN_PARQUET,
    SVD_SEED,
    TOP_K,
)
from content_model import build_user_profile, compute_content_scores
from hybrid import compute_hybrid_scores, save_alpha, z_score


def precision_at_k(recommended_cols: list[int], liked_cols: set[int], k: int = 10) -> float:
    if not recommended_cols or not liked_cols:
        return 0.0
    hits = sum(1 for c in recommended_cols[:k] if c in liked_cols)
    return hits / k


def get_top_k_indices(scores: np.ndarray, k: int = 10) -> list[int]:
    """Return indices of top k highest finite values."""
    valid_idx = np.where(np.isfinite(scores))[0]
    if len(valid_idx) == 0:
        return []
    if len(valid_idx) <= k:
        return valid_idx[np.argsort(scores[valid_idx])[::-1]].tolist()
    top_part = np.argpartition(scores, -k)[-k:]
    return top_part[np.argsort(scores[top_part])[::-1]].tolist()


def main(full: bool = False) -> None:
    print("=" * 60)
    print("PHASE 2 EVALUATION (Hybrid & Neural Content Recommender)")
    print("=" * 60)

    # 1. Load clean ratings & movie embeddings
    print("Loading ratings and movie embeddings …")
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    movie_embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
    movie_ids = np.load(MOVIE_IDS_PATH)
    n_catalog_movies = len(movie_ids)
    print(f"  Ratings: {len(ratings):,} | Catalog movies: {n_catalog_movies:,}")

    # 2. Disjoint 70 / 10 / 20 split per user
    print("Splitting ratings 70% Train / 10% Validation / 20% Test per user …")
    rng = np.random.default_rng(EVAL_SEED)
    train_rows, val_rows, test_rows = [], [], []

    for uid, grp in ratings.groupby("userId"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)

        n_test = max(1, int(round(n * 0.20)))
        n_val = max(1, int(round(n * 0.10)))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1

        train_rows.append(grp.loc[idx[:n_train]])
        val_rows.append(grp.loc[idx[n_train : n_train + n_val]])
        test_rows.append(grp.loc[idx[n_train + n_val :]])

    train = pd.concat(train_rows, ignore_index=True)
    val   = pd.concat(val_rows,   ignore_index=True)
    test  = pd.concat(test_rows,  ignore_index=True)
    print(f"  Train: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}")

    # Build index mappings
    # Note: movie column index is aligned with movie_ids array (from Phase 1 / embed_movies)
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    users = sorted(ratings["userId"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    # 3. Build train sparse matrix & fit SVD on train only
    print(f"\nBuilding train sparse matrix and fitting SVD (n_components={N_COMPONENTS}) on train only …")
    r_user = train["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
    r_movie = train["movieId_tmdb"].map(movie_to_idx).to_numpy(dtype=np.int32)
    r_rating = train["rating"].to_numpy(dtype=np.float32)

    valid_mask = (r_user >= 0) & (r_movie >= 0)
    train_matrix = csr_matrix(
        (r_rating[valid_mask], (r_user[valid_mask], r_movie[valid_mask])),
        shape=(n_users, n_catalog_movies),
        dtype=np.float32,
    )

    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    train_user_factors = svd.fit_transform(train_matrix)
    svd_components = svd.components_  # (n_components, n_catalog_movies)

    # Movie train rating counts for popularity and skew calculation
    train_movie_counts = np.zeros(n_catalog_movies, dtype=np.int32)
    counts = train["movieId_tmdb"].map(movie_to_idx).value_counts()
    for col_idx, cnt in counts.items():
        if 0 <= col_idx < n_catalog_movies:
            train_movie_counts[col_idx] = cnt

    # Popularity baseline: top-k movies in train rated >= 4.0
    liked_train = train[train["rating"] >= LIKE_THRESHOLD]
    pop_liked_counts = np.zeros(n_catalog_movies, dtype=np.int32)
    pcounts = liked_train["movieId_tmdb"].map(movie_to_idx).value_counts()
    for col_idx, cnt in pcounts.items():
        if 0 <= col_idx < n_catalog_movies:
            pop_liked_counts[col_idx] = cnt
    pop_ranked_cols = np.argsort(pop_liked_counts)[::-1]

    # Pre-build per-user train data:
    # 1. rated_cols: set of movie column indices rated in train (to mask)
    # 2. liked_cols: set of movie column indices rated >= 4.0 in train (for content profile)
    # 3. train_count: total ratings in train
    print("Indexing per-user train histories …")
    user_train_rated: dict[int, set[int]] = {}
    user_train_liked: dict[int, set[int]] = {}
    user_train_count: dict[int, int] = {}

    for uid, grp in train.groupby("userId"):
        cols = set(movie_to_idx[m] for m in grp["movieId_tmdb"] if m in movie_to_idx)
        user_train_rated[uid] = cols
        liked = set(
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        )
        user_train_liked[uid] = liked
        user_train_count[uid] = len(grp)

    # 4. Tune alpha on VALIDATION split
    print("\nTuning alpha over [0.0, 0.1, ..., 1.0] on VALIDATION precision@10 …")
    val_liked: dict[int, set[int]] = {}
    for uid, grp in val.groupby("userId"):
        liked = set(
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        )
        if liked:
            val_liked[uid] = liked

    val_eval_users = list(val_liked.keys())
    # Subsample validation users if large to speed up tuning
    if len(val_eval_users) > EVAL_N_USERS:
        val_rng = np.random.default_rng(EVAL_SEED)
        val_eval_users = list(val_rng.choice(val_eval_users, size=EVAL_N_USERS, replace=False))
    print(f"  Validating with {len(val_eval_users):,} validation users who have liked items …")

    # Precompute standardized SVD and Content scores once per validation user
    val_user_scores: list[tuple[set[int], np.ndarray, np.ndarray]] = []
    for uid in val_eval_users:
        u_idx = user_to_idx[uid]
        rated_cols = user_train_rated.get(uid, set())
        liked_cols = user_train_liked.get(uid, set())

        # SVD score
        svd_s = (train_user_factors[u_idx] @ svd_components).copy()
        if rated_cols:
            svd_s[list(rated_cols)] = -np.inf
        z_svd = z_score(svd_s)

        # Content score
        profile = build_user_profile(liked_cols, movie_embeddings)
        cnt_s = compute_content_scores(profile, movie_embeddings, rated_cols)
        z_cnt = z_score(cnt_s) if cnt_s is not None else z_svd

        val_user_scores.append((val_liked[uid], z_svd, z_cnt))

    candidate_alphas = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]
    best_alpha = 0.5
    best_val_p10 = -1.0

    print(f"{'alpha':>6}  {'val_precision@10':>18}")
    for alpha in candidate_alphas:
        p_list = []
        for liked_set, z_svd, z_cnt in val_user_scores:
            h_scores = alpha * z_svd + (1.0 - alpha) * z_cnt
            recs = get_top_k_indices(h_scores, TOP_K)
            p_list.append(precision_at_k(recs, liked_set, TOP_K))
        val_p10 = float(np.mean(p_list))
        print(f"  {alpha:>4.1f}  {val_p10:>18.4f}")
        if val_p10 > best_val_p10:
            best_val_p10 = val_p10
            best_alpha = alpha

    print(f"\n→ Best alpha on validation set: {best_alpha:.1f} (precision@10 = {best_val_p10:.4f})")
    save_alpha(best_alpha)

    # 5. Evaluate on TEST split
    print(f"\nEvaluating on TEST split (sample={EVAL_N_USERS if not full else 'ALL'}) …")
    test_liked: dict[int, set[int]] = {}
    for uid, grp in test.groupby("userId"):
        liked = set(
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        )
        if liked:
            test_liked[uid] = liked

    test_candidates = sorted(test_liked.keys())
    if not full and len(test_candidates) > EVAL_N_USERS:
        rng_eval = np.random.default_rng(EVAL_SEED + 1)
        test_eval_users = list(rng_eval.choice(test_candidates, size=EVAL_N_USERS, replace=False))
    else:
        test_eval_users = test_candidates

    print(f"  Test users evaluated: {len(test_eval_users):,} (each has >= 1 liked test item)")

    # Data structures to collect results per model
    models = ["popularity", "SVD", "content-only", f"hybrid (alpha={best_alpha:.1f})"]

    # Storage for user precisions
    precisions: dict[str, list[float]] = {m: [] for m in models}
    cold_precisions: dict[str, list[float]] = {m: [] for m in models}
    heavy_precisions: dict[str, list[float]] = {m: [] for m in models}

    # Storage for all recommended items (catalog coverage and popularity skew)
    recommended_items: dict[str, list[int]] = {m: [] for m in models}

    cold_users_count = 0
    heavy_users_count = 0

    for uid in test_eval_users:
        u_idx = user_to_idx[uid]
        liked_test_cols = test_liked[uid]
        rated_train_cols = user_train_rated.get(uid, set())
        liked_train_cols = user_train_liked.get(uid, set())
        t_cnt = user_train_count.get(uid, 0)

        is_cold = (t_cnt <= 5)
        is_heavy = (t_cnt >= 50)
        if is_cold:
            cold_users_count += 1
        if is_heavy:
            heavy_users_count += 1

        # 1. Popularity
        pop_recs = [c for c in pop_ranked_cols if c not in rated_train_cols][:TOP_K]

        # 2. SVD
        svd_s = (train_user_factors[u_idx] @ svd_components).copy()
        if rated_train_cols:
            svd_s[list(rated_train_cols)] = -np.inf
        svd_recs = get_top_k_indices(svd_s, TOP_K)

        # 3. Content-only
        profile = build_user_profile(liked_train_cols, movie_embeddings)
        cnt_s = compute_content_scores(profile, movie_embeddings, rated_train_cols)
        cnt_recs = get_top_k_indices(cnt_s, TOP_K) if cnt_s is not None else []

        # 4. Hybrid
        hyb_s = compute_hybrid_scores(svd_s, cnt_s, best_alpha)
        hyb_recs = get_top_k_indices(hyb_s, TOP_K)

        rec_dict = {
            "popularity": pop_recs,
            "SVD": svd_recs,
            "content-only": cnt_recs,
            f"hybrid (alpha={best_alpha:.1f})": hyb_recs,
        }

        for m_name, recs in rec_dict.items():
            p = precision_at_k(recs, liked_test_cols, TOP_K)
            precisions[m_name].append(p)
            recommended_items[m_name].extend(recs)

            if is_cold:
                cold_precisions[m_name].append(p)
            if is_heavy:
                heavy_precisions[m_name].append(p)

    # Compute metrics summary
    def get_skew(item_list: list[int]) -> float:
        if not item_list:
            return 0.0
        counts = [train_movie_counts[c] for c in item_list]
        return float(np.mean(np.log10(np.array(counts, dtype=np.float32) + 1.0)))

    def get_coverage(item_list: list[int]) -> int:
        return len(set(item_list))

    print("\n" + "=" * 78)
    print("OVERALL TEST RESULTS (N = {:,} users)".format(len(test_eval_users)))
    print("=" * 78)
    header = f"{'Model':<25}  {'Precision@10':>14}  {'Pop Skew':>12}  {'Catalog Coverage':>18}"
    print(header)
    print("-" * len(header))
    for m in models:
        p_val = np.mean(precisions[m]) if precisions[m] else 0.0
        skew_val = get_skew(recommended_items[m])
        cov_val = get_coverage(recommended_items[m])
        cov_pct = cov_val / n_catalog_movies * 100
        print(f"{m:<25}  {p_val:>14.4f}  {skew_val:>12.3f}  {cov_val:>8,} ({cov_pct:>4.1f}%)")

    # Cold-start table
    print("\n" + "=" * 78)
    print(f"COLD-START USERS (<= 5 train ratings, N = {cold_users_count:,} users)")
    print("=" * 78)
    print(f"{'Model':<25}  {'Precision@10':>14}")
    print("-" * 42)
    for m in models:
        p_val = np.mean(cold_precisions[m]) if cold_precisions[m] else 0.0
        print(f"{m:<25}  {p_val:>14.4f}")

    # Heavy users table
    print("\n" + "=" * 78)
    print(f"HEAVY USERS (>= 50 train ratings, N = {heavy_users_count:,} users)")
    print("=" * 78)
    print(f"{'Model':<25}  {'Precision@10':>14}")
    print("-" * 42)
    for m in models:
        p_val = np.mean(heavy_precisions[m]) if heavy_precisions[m] else 0.0
        print(f"{m:<25}  {p_val:>14.4f}")

    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Evaluate all test users")
    args = parser.parse_args()
    main(full=args.full)
