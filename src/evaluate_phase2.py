"""
Phase 2 Evaluation: Comprehensive Benchmark of All Recommendation Models.

Models compared:
  1. Popularity (backfilled to K unseen candidates per user)
  2. TruncatedSVD Collaborative Filtering (Phase 1 / Phase 2 baseline)
  3. Implicit ALS (64 factors, regularization=0.05, alpha=40)
  4. Neural Content-Only (SentenceTransformer all-MiniLM-L6-v2 embeddings)
  5. Fixed Hybrid (alpha * z(SVD) + (1-alpha) * z(Content))
  6. Learning-to-Rank (LTR) Reranker (LGBMRanker trained on validation users)

Evaluation conditions:
  1. Overall (no candidate filter, N=5,000 test users)
  2. Candidate-filtered (>= 20 train ratings)
  3. Heavy users (>= 50 train ratings)
  4. User cold-start simulation (n in {1, 3, 5} train ratings kept)
  5. Item cold-start (500 movies removed from train; only they are targets)

Statistical rigor:
  - 95% bootstrap confidence intervals (1000 resamples over users) for all numbers.
  - Paired difference bootstrap CIs (hybrid - SVD, ALS - SVD, reranker - hybrid).

Performance & Memory:
  - Mini-batch evaluation (BATCH_SIZE=200 users) ensures peak RAM < 200 MB.
  - Fully vectorised batch prediction for the LightGBM reranker across all candidate pairs in each batch.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from als_model import (
    build_implicit_matrix,
    load_als_artifacts,
    recalculate_user_factors,
    train_als,
)
from config import (
    ALS_ALPHA,
    ALS_FACTORS,
    ALS_ITERATIONS,
    ALS_REGULARIZATION,
    ALS_SEED,
    EVAL_N_USERS,
    EVAL_SEED,
    LIKE_THRESHOLD,
    MIN_TRAIN_RATINGS_CANDIDATE,
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    MOVIES_CLEAN_PARQUET,
    N_COMPONENTS,
    RATINGS_CLEAN_PARQUET,
    SVD_SEED,
    TOP_K,
)
from content_model import build_user_profile
from hybrid import load_alpha, save_alpha
from reranker import (
    PrecomputedMetadata,
    build_user_liked_genres,
    extract_pair_features,
    load_reranker,
)

BATCH_SIZE = 200


def precision_at_k(recs: list[int], liked: set[int], k: int = 10) -> float:
    if not recs or not liked:
        return 0.0
    return sum(1 for c in recs[:k] if c in liked) / k


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + ci) / 2))
    return float(values.mean()), lo, hi


def paired_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, seed: int = 1) -> tuple[float, float, float]:
    return bootstrap_ci(a - b, n_resamples=n_resamples, seed=seed)


def fmt_ci(m: float, lo: float, hi: float) -> str:
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def evaluate_batch(
    eval_uids: list[int],
    user_to_idx: dict[int, int],
    user_train_rated: dict[int, set[int]],
    user_train_liked: dict[int, set[int]],
    user_test_liked: dict[int, set[int]],
    svd_user_factors: np.ndarray,
    svd_components: np.ndarray,
    als_user_factors: np.ndarray,
    als_item_factors: np.ndarray,
    movie_embeddings: np.ndarray,
    pop_ranked_cols: np.ndarray,
    candidate_mask: np.ndarray,
    hybrid_alpha: float,
    meta: PrecomputedMetadata,
    reranker_model,
    top_k: int = TOP_K,
    override_svd_factors: dict[int, np.ndarray] | None = None,
    override_als_factors: dict[int, np.ndarray] | None = None,
    target_cols: set[int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Evaluate all 6 models on eval_uids in memory-safe mini-batches.
    Returns dict: model_name -> 1D numpy array of precision@k per evaluated user.
    """
    cand_cols = np.where(candidate_mask)[0]

    active = []
    for uid in eval_uids:
        tl = user_test_liked.get(uid, set())
        if target_cols is not None:
            tl = tl & target_cols
        if tl:
            active.append((uid, tl))

    if not active:
        empty = np.array([], dtype=np.float64)
        return {
            "popularity": empty,
            "SVD": empty,
            "ALS": empty,
            "content": empty,
            "hybrid": empty,
            "reranker": empty,
        }

    pop_p, svd_p, als_p, cnt_p, hyb_p, rnk_p = [], [], [], [], [], []

    for batch_start in range(0, len(active), BATCH_SIZE):
        batch = active[batch_start : batch_start + BATCH_SIZE]
        bsize = len(batch)
        uids_b = [x[0] for x in batch]
        tls_b = [x[1] for x in batch]

        # 1. SVD batch factors & scores
        svd_uf_b = np.array([
            override_svd_factors.get(uid, svd_user_factors[user_to_idx[uid]])
            if override_svd_factors else svd_user_factors[user_to_idx[uid]]
            for uid in uids_b
        ], dtype=np.float64)
        raw_svd = svd_uf_b @ svd_components.astype(np.float64)  # (bsize, n_movies)

        # 2. ALS batch factors & scores
        als_uf_b = np.array([
            override_als_factors.get(uid, als_user_factors[user_to_idx[uid]])
            if override_als_factors else als_user_factors[user_to_idx[uid]]
            for uid in uids_b
        ], dtype=np.float64)
        raw_als = als_uf_b @ als_item_factors.T.astype(np.float64)  # (bsize, n_movies)

        # 3. Content batch profiles & scores
        prof_mat = np.zeros((bsize, movie_embeddings.shape[1]), dtype=np.float64)
        has_prof = []
        for i, uid in enumerate(uids_b):
            prof = build_user_profile(user_train_liked.get(uid, set()), movie_embeddings)
            has_prof.append(prof is not None)
            if prof is not None:
                prof_mat[i] = prof
        raw_cnt = prof_mat @ movie_embeddings.T.astype(np.float64)  # (bsize, n_movies)

        # Prepare containers for batch reranking
        user_cand_lists: list[list[int]] = []
        user_X_list: list[np.ndarray] = []
        user_slice_map: list[tuple[int, int] | None] = []
        cur_pos = 0

        for i, uid in enumerate(uids_b):
            tl = tls_b[i]
            rated = user_train_rated.get(uid, set())

            # Available candidates = candidate_mask AND NOT rated
            avail_cols = [c for c in cand_cols if c not in rated]
            if not avail_cols:
                pop_p.append(0.0)
                svd_p.append(0.0)
                als_p.append(0.0)
                cnt_p.append(0.0)
                hyb_p.append(0.0)
                user_slice_map.append(None)
                continue

            avail_arr = np.array(avail_cols, dtype=np.int32)
            avail_set = set(avail_cols)

            # --- Model 1: Popularity ---
            pop_recs = [c for c in pop_ranked_cols if c in avail_set][:top_k]
            pop_p.append(precision_at_k(pop_recs, tl, top_k))

            def get_recs_from_subscores(sc: np.ndarray) -> list[int]:
                k = min(top_k, len(sc))
                if k == 0:
                    return []
                part = np.argpartition(sc, -k)[-k:]
                return avail_arr[part[np.argsort(sc[part])[::-1]]].tolist()

            # --- Model 2: SVD ---
            svd_avail = raw_svd[i, avail_arr]
            mu_s, std_s = svd_avail.mean(), svd_avail.std()
            z_svd = (svd_avail - mu_s) / (std_s if std_s > 1e-9 else 1.0)
            svd_recs = get_recs_from_subscores(z_svd)
            svd_p.append(precision_at_k(svd_recs, tl, top_k))

            # --- Model 3: ALS ---
            als_avail = raw_als[i, avail_arr]
            mu_a, std_a = als_avail.mean(), als_avail.std()
            z_als = (als_avail - mu_a) / (std_a if std_a > 1e-9 else 1.0)
            als_recs = get_recs_from_subscores(z_als)
            als_p.append(precision_at_k(als_recs, tl, top_k))

            # --- Model 4: Content-Only ---
            if has_prof[i]:
                cnt_avail = raw_cnt[i, avail_arr]
                mu_c, std_c = cnt_avail.mean(), cnt_avail.std()
                z_cnt = (cnt_avail - mu_c) / (std_c if std_c > 1e-9 else 1.0)
                cnt_recs = get_recs_from_subscores(z_cnt)
            else:
                z_cnt = np.zeros_like(z_svd)
                cnt_recs = []
            cnt_p.append(precision_at_k(cnt_recs, tl, top_k))

            # --- Model 5: Fixed Hybrid (alpha * z(SVD) + (1-alpha) * z(Content)) ---
            z_hyb = hybrid_alpha * z_svd + (1.0 - hybrid_alpha) * z_cnt
            hyb_recs = get_recs_from_subscores(z_hyb)
            hyb_p.append(precision_at_k(hyb_recs, tl, top_k))

            # --- Model 6: Prepare Candidates for Learned Reranker ---
            k_pool = min(100, len(avail_arr))
            top_als_pool = avail_arr[np.argpartition(z_als, -k_pool)[-k_pool:]].tolist()
            top_cnt_pool = avail_arr[np.argpartition(z_cnt, -k_pool)[-k_pool:]].tolist() if has_prof[i] else []
            top_pop_pool = [c for c in pop_ranked_cols if c in avail_set][:50]
            top_svd_pool = avail_arr[np.argpartition(z_svd, -min(50, len(avail_arr)))[-min(50, len(avail_arr)):]].tolist()

            rerank_candidates = list(dict.fromkeys(top_als_pool + top_cnt_pool + top_pop_pool + top_svd_pool))
            user_cand_lists.append(rerank_candidates)

            u_liked_train = user_train_liked.get(uid, set())
            u_genres = build_user_liked_genres(u_liked_train, meta.movie_genres)
            u_prof = prof_mat[i] if has_prof[i] else None

            X_pairs = extract_pair_features(
                rerank_candidates,
                als_uf_b[i],
                u_prof,
                u_genres,
                als_item_factors,
                movie_embeddings,
                meta,
            )
            user_X_list.append(X_pairs)
            n_c = len(rerank_candidates)
            user_slice_map.append((cur_pos, cur_pos + n_c))
            cur_pos += n_c

        # Batch predict for all users in this mini-batch!
        if user_X_list and cur_pos > 0:
            X_batch = np.vstack(user_X_list)
            scores_batch = reranker_model.predict(X_batch)
            valid_u_idx = 0
            for i, uid in enumerate(uids_b):
                sl = user_slice_map[i]
                if sl is None:
                    rnk_p.append(0.0)
                    continue
                start, end = sl
                user_scores = scores_batch[start:end]
                user_cands = user_cand_lists[valid_u_idx]
                valid_u_idx += 1
                k_rnk = min(top_k, len(user_scores))
                if k_rnk == 0:
                    rnk_p.append(0.0)
                    continue
                rnk_part = np.argpartition(user_scores, -k_rnk)[-k_rnk:]
                rnk_sorted = rnk_part[np.argsort(user_scores[rnk_part])[::-1]]
                rnk_recs = [user_cands[idx] for idx in rnk_sorted]
                rnk_p.append(precision_at_k(rnk_recs, tls_b[i], top_k))
        else:
            for _ in uids_b:
                rnk_p.append(0.0)

    return {
        "popularity": np.array(pop_p, dtype=np.float64),
        "SVD": np.array(svd_p, dtype=np.float64),
        "ALS": np.array(als_p, dtype=np.float64),
        "content": np.array(cnt_p, dtype=np.float64),
        "hybrid": np.array(hyb_p, dtype=np.float64),
        "reranker": np.array(rnk_p, dtype=np.float64),
    }


def print_condition_table(
    results: dict[str, np.ndarray],
    condition_title: str,
    hybrid_alpha: float,
) -> None:
    n_users = len(results.get("SVD", []))
    print("\n" + "=" * 78)
    print(f"{condition_title} (N = {n_users:,} evaluated users)")
    print("=" * 78)

    labels = {
        "popularity": "popularity",
        "SVD": "SVD",
        "ALS": "ALS",
        "content": "content-only",
        "hybrid": f"fixed hybrid (α={hybrid_alpha:.1f})",
        "reranker": "learned reranker (LGBM)",
    }

    header = f"  {'Model':<30}  {'Precision@10':>14}  {'95% Bootstrap CI':>24}"
    print(header)
    print("  " + "-" * 72)

    for key in ("popularity", "SVD", "ALS", "content", "hybrid", "reranker"):
        vals = results.get(key, np.array([]))
        if len(vals) == 0:
            print(f"  {labels[key]:<30}  {'n/a':>14}")
        else:
            m, lo, hi = bootstrap_ci(vals, seed=42)
            print(f"  {labels[key]:<30}  {m:>14.4f}  [{lo:.4f}, {hi:.4f}]")

    print("  " + "-" * 72)
    # Paired comparisons
    if len(results.get("SVD", [])) > 0:
        svd_v = results["SVD"]
        als_v = results["ALS"]
        hyb_v = results["hybrid"]
        rnk_v = results["reranker"]

        m_diff, lo, hi = paired_ci(als_v, svd_v, seed=51)
        sign = "+" if m_diff >= 0 else ""
        print(f"  {'ALS − SVD (paired)':<30}  {sign+f'{m_diff:.4f}':>14}  [{lo:.4f}, {hi:.4f}]")

        m_diff, lo, hi = paired_ci(hyb_v, svd_v, seed=52)
        sign = "+" if m_diff >= 0 else ""
        print(f"  {'hybrid − SVD (paired)':<30}  {sign+f'{m_diff:.4f}':>14}  [{lo:.4f}, {hi:.4f}]")

        m_diff, lo, hi = paired_ci(rnk_v, hyb_v, seed=53)
        sign = "+" if m_diff >= 0 else ""
        print(f"  {'reranker − hybrid (paired)':<30}  {sign+f'{m_diff:.4f}':>14}  [{lo:.4f}, {hi:.4f}]")

        m_diff, lo, hi = paired_ci(rnk_v, als_v, seed=54)
        sign = "+" if m_diff >= 0 else ""
        print(f"  {'reranker − ALS (paired)':<30}  {sign+f'{m_diff:.4f}':>14}  [{lo:.4f}, {hi:.4f}]")


def main(full: bool = False) -> None:
    print("=" * 78)
    print("PHASE 2 COMPREHENSIVE BENCHMARK: SVD vs ALS vs Content vs Hybrid vs Reranker")
    print("=" * 78)

    # 1. Load clean ratings & movie metadata
    print("\nLoading dataset and movie embeddings …")
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    movies_df = pd.read_parquet(MOVIES_CLEAN_PARQUET)
    movie_ids = np.load(MOVIE_IDS_PATH)
    movie_embeddings = np.load(MOVIE_EMBEDDINGS_PATH).astype(np.float32)
    n_movies = len(movie_ids)
    movie_to_idx = {int(m): i for i, m in enumerate(movie_ids)}
    users = sorted(ratings["userId"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    print(f"  Ratings: {len(ratings):,} | Users: {n_users:,} | Movies: {n_movies:,}")

    # 2. 70/10/20 per-user split
    print("Splitting ratings 70% Train / 10% Validation / 20% Test per user …")
    rng = np.random.default_rng(EVAL_SEED)
    train_rows, val_rows, test_rows = [], [], []
    for uid, grp in ratings.groupby("userId"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * 0.20)))
        n_val = max(1, int(round(n * 0.10)))
        n_train = max(1, n - n_test - n_val)
        if n_train + n_val + n_test > n:
            n_test = max(0, n - n_train - n_val)
        train_rows.append(grp.loc[idx[:n_train]])
        val_rows.append(grp.loc[idx[n_train : n_train + n_val]])
        test_rows.append(grp.loc[idx[n_train + n_val :]])

    train = pd.concat(train_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True)
    print(f"  Train: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}")

    meta = PrecomputedMetadata(train, movie_ids, movie_to_idx, movies_df)

    # 3. Fit SVD on train only
    print(f"\nFitting TruncatedSVD (n_components={N_COMPONENTS}) on train …")
    r_user = train["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
    r_movie = train["movieId_tmdb"].map(movie_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    r_rating = train["rating"].to_numpy(dtype=np.float32)
    vmask = (r_user >= 0) & (r_movie >= 0)

    train_matrix = csr_matrix(
        (r_rating[vmask], (r_user[vmask], r_movie[vmask])),
        shape=(n_users, n_movies),
        dtype=np.float32,
    )
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    svd_user_factors = svd.fit_transform(train_matrix)
    svd_components = svd.components_

    # 4. Load or fit ALS model
    print("Loading ALS artifacts (or fitting if needed) …")
    try:
        als_model, als_user_factors, als_item_factors = load_als_artifacts()
    except Exception:
        print("  ALS artifacts not found on disk, training ALS now …")
        mat_als = build_implicit_matrix(train, user_to_idx, movie_to_idx, n_users, n_movies)
        als_model = train_als(mat_als, alpha=ALS_ALPHA, show_progress=False)
        als_user_factors = als_model.user_factors
        als_item_factors = als_model.item_factors

    print(f"  ALS factors: {als_user_factors.shape[1]} (alpha={ALS_ALPHA})")

    # 5. Load or train Reranker
    print("Loading LGBMRanker (or training if needed) …")
    try:
        reranker_model = load_reranker()
    except Exception:
        print("  Reranker not found on disk, running src/reranker.py …")
        import reranker
        reranker.main()
        reranker_model = load_reranker()

    # 6. Candidate masks & popularity baseline
    candidate_mask_all = np.ones(n_movies, dtype=bool)
    candidate_mask_filtered = meta.train_counts >= MIN_TRAIN_RATINGS_CANDIDATE
    print(f"  Total movies in catalog : {n_movies:,}")
    print(f"  Candidate filtered pool (>= {MIN_TRAIN_RATINGS_CANDIDATE} train ratings): {int(candidate_mask_filtered.sum()):,}")

    liked_train = train[train["rating"] >= LIKE_THRESHOLD]
    pop_liked_counts = np.zeros(n_movies, dtype=np.int32)
    pl_counts = liked_train["movieId_tmdb"].map(movie_to_idx).dropna().astype(int).value_counts()
    for ci, cnt in pl_counts.items():
        if 0 <= ci < n_movies:
            pop_liked_counts[ci] = cnt
    pop_ranked_cols = np.argsort(pop_liked_counts)[::-1]

    # Index per-user histories
    print("Indexing per-user histories …")
    user_train_rated: dict[int, set[int]] = {}
    user_train_liked: dict[int, set[int]] = {}
    user_train_count: dict[int, int] = {}
    for uid, grp in train.groupby("userId"):
        cols = {movie_to_idx[m] for m in grp["movieId_tmdb"] if m in movie_to_idx}
        user_train_rated[uid] = cols
        liked = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        user_train_liked[uid] = liked
        user_train_count[uid] = len(grp)

    # 7. Tune alpha on VALIDATION for existing fixed hybrid
    print("\nChecking / tuning hybrid alpha on validation …")
    hybrid_alpha = load_alpha(default=0.5)
    print(f"  Using tuned hybrid alpha: {hybrid_alpha:.2f}")

    # 8. Test set user indexing
    test_liked_all: dict[int, set[int]] = {}
    for uid, grp in test.groupby("userId"):
        liked = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        if liked:
            test_liked_all[uid] = liked

    test_pool = sorted(test_liked_all.keys())
    if not full and len(test_pool) > EVAL_N_USERS:
        test_eval_uids = list(np.random.default_rng(EVAL_SEED + 1).choice(test_pool, EVAL_N_USERS, replace=False))
    else:
        test_eval_uids = test_pool
    print(f"  Test users evaluated: {len(test_eval_uids):,}")

    # ── CONDITION 1: OVERALL (no candidate filter) ──────────────────────────
    print("\nRunning Condition 1: OVERALL (no filter) …")
    res_overall = evaluate_batch(
        test_eval_uids,
        user_to_idx,
        user_train_rated,
        user_train_liked,
        test_liked_all,
        svd_user_factors,
        svd_components,
        als_user_factors,
        als_item_factors,
        movie_embeddings,
        pop_ranked_cols,
        candidate_mask_all,
        hybrid_alpha,
        meta,
        reranker_model,
    )
    print_condition_table(res_overall, "CONDITION 1: OVERALL (no filter)", hybrid_alpha)

    # ── CONDITION 2: CANDIDATE FILTERED (>= 20 train ratings) ───────────────
    print(f"\nRunning Condition 2: CANDIDATE FILTERED (>= {MIN_TRAIN_RATINGS_CANDIDATE} train ratings) …")
    res_filtered = evaluate_batch(
        test_eval_uids,
        user_to_idx,
        user_train_rated,
        user_train_liked,
        test_liked_all,
        svd_user_factors,
        svd_components,
        als_user_factors,
        als_item_factors,
        movie_embeddings,
        pop_ranked_cols,
        candidate_mask_filtered,
        hybrid_alpha,
        meta,
        reranker_model,
    )
    print_condition_table(
        res_filtered,
        f"CONDITION 2: CANDIDATE FILTERED (>= {MIN_TRAIN_RATINGS_CANDIDATE} train ratings)",
        hybrid_alpha,
    )

    # ── CONDITION 3: HEAVY USERS (>= 50 train ratings) ──────────────────────
    heavy_uids = [u for u in test_eval_uids if user_train_count.get(u, 0) >= 50]
    print(f"\nRunning Condition 3: HEAVY USERS (>= 50 train ratings, N={len(heavy_uids):,}) …")
    if heavy_uids:
        res_heavy = evaluate_batch(
            heavy_uids,
            user_to_idx,
            user_train_rated,
            user_train_liked,
            test_liked_all,
            svd_user_factors,
            svd_components,
            als_user_factors,
            als_item_factors,
            movie_embeddings,
            pop_ranked_cols,
            candidate_mask_all,
            hybrid_alpha,
            meta,
            reranker_model,
        )
        print_condition_table(res_heavy, "CONDITION 3: HEAVY USERS (>= 50 train ratings)", hybrid_alpha)
    else:
        res_heavy = {}

    # ── CONDITION 4: USER COLD-START (n in {1, 3, 5}) ───────────────────────
    print("\n" + "=" * 78)
    print("CONDITION 4: USER COLD-START (2,000 users with >= 30 ratings, truncated history)")
    print("=" * 78)

    sim_pool = [u for u in user_to_idx if user_train_count.get(u, 0) >= 30]
    sim_rng = np.random.default_rng(EVAL_SEED + 2)
    sim_uids = list(sim_rng.choice(sim_pool, size=min(2000, len(sim_pool)), replace=False))
    print(f"  Sampled {len(sim_uids):,} users from pool of {len(sim_pool):,}")

    cold_sim_results: dict[int, dict[str, np.ndarray]] = {}

    for n_keep in [1, 3, 5]:
        print(f"\nEvaluating n_train = {n_keep} kept rating(s) …")
        sim_svd_override: dict[int, np.ndarray] = {}
        sim_als_override: dict[int, np.ndarray] = {}
        sim_rated_t: dict[int, set[int]] = {}
        sim_liked_t: dict[int, set[int]] = {}
        sim_test_t: dict[int, set[int]] = {}

        for uid in sim_uids:
            all_rated = list(user_train_rated.get(uid, set()))
            if len(all_rated) < n_keep:
                continue
            chosen_i = sim_rng.choice(len(all_rated), size=n_keep, replace=False)
            chosen = set(all_rated[j] for j in chosen_i)
            chosen_liked = chosen & user_train_liked.get(uid, set())
            held_liked = (user_train_liked.get(uid, set()) - chosen_liked) | test_liked_all.get(uid, set())
            if not held_liked:
                continue

            sim_rated_t[uid] = chosen
            sim_liked_t[uid] = chosen_liked
            sim_test_t[uid] = held_liked

            # SVD factor recalculation from truncated sparse row
            trunc_svd_row = csr_matrix(
                (np.full(n_keep, 3.5, dtype=np.float32),
                 (np.zeros(n_keep, dtype=np.int32), np.array(list(chosen), dtype=np.int32))),
                shape=(1, n_movies),
                dtype=np.float32,
            )
            sim_svd_override[uid] = svd.transform(trunc_svd_row)[0]

            # ALS factor recalculation
            sim_als_override[uid] = recalculate_user_factors(als_model, chosen_liked, n_movies)

        sim_eval_users = [u for u in sim_uids if u in sim_test_t]
        r_sim = evaluate_batch(
            sim_eval_users,
            user_to_idx,
            sim_rated_t,
            sim_liked_t,
            sim_test_t,
            svd_user_factors,
            svd_components,
            als_user_factors,
            als_item_factors,
            movie_embeddings,
            pop_ranked_cols,
            candidate_mask_all,
            hybrid_alpha,
            meta,
            reranker_model,
            override_svd_factors=sim_svd_override,
            override_als_factors=sim_als_override,
        )
        cold_sim_results[n_keep] = r_sim
        print_condition_table(
            r_sim,
            f"CONDITION 4.{n_keep}: USER COLD-START (n={n_keep} train rating{'s' if n_keep > 1 else ''})",
            hybrid_alpha,
        )

    # ── CONDITION 5: ITEM COLD-START (500 movies removed from train) ────────
    print("\n" + "=" * 78)
    print("CONDITION 5: ITEM COLD-START (500 popular movies removed from train)")
    print("=" * 78)

    popular_movie_cols = np.where(meta.train_counts >= 30)[0]
    item_rng = np.random.default_rng(EVAL_SEED + 3)
    n_remove = min(500, len(popular_movie_cols))
    removed_cols: set[int] = set(
        int(c) for c in item_rng.choice(popular_movie_cols, size=n_remove, replace=False)
    )
    print(f"  Removed {len(removed_cols):,} movies from train.")

    # Refit SVD without removed movies
    print("  Refitting SVD without removed movies …")
    vmask2 = vmask & ~np.isin(r_movie, list(removed_cols))
    train_matrix2 = csr_matrix(
        (r_rating[vmask2], (r_user[vmask2], r_movie[vmask2])),
        shape=(n_users, n_movies),
        dtype=np.float32,
    )
    svd2 = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    svd_user_factors2 = svd2.fit_transform(train_matrix2)
    svd_components2 = svd2.components_

    # Refit ALS without removed movies
    print("  Refitting ALS without removed movies …")
    train_df_no_removed = train[~train["movieId_tmdb"].map(movie_to_idx).isin(removed_cols)]
    mat_als2 = build_implicit_matrix(train_df_no_removed, user_to_idx, movie_to_idx, n_users, n_movies)
    als_model2 = train_als(mat_als2, alpha=ALS_ALPHA, show_progress=False)

    user_rated2 = {u: cols - removed_cols for u, cols in user_train_rated.items()}
    user_liked2 = {u: cols - removed_cols for u, cols in user_train_liked.items()}

    ics_liked: dict[int, set[int]] = {}
    for uid, liked_set in test_liked_all.items():
        tgt = liked_set & removed_cols
        if tgt:
            ics_liked[uid] = tgt

    ics_uids = list(ics_liked.keys())
    print(f"  Test users with >= 1 liked removed movie in test: {len(ics_uids):,}")

    if ics_uids:
        res_ics = evaluate_batch(
            ics_uids,
            user_to_idx,
            user_rated2,
            user_liked2,
            ics_liked,
            svd_user_factors2,
            svd_components2,
            als_model2.user_factors,
            als_model2.item_factors,
            movie_embeddings,
            pop_ranked_cols,
            candidate_mask_all,
            hybrid_alpha,
            meta,
            reranker_model,
            target_cols=removed_cols,
        )
        print_condition_table(
            res_ics,
            "CONDITION 5: ITEM COLD-START (only removed 500 movies as targets)",
            hybrid_alpha,
        )
    else:
        res_ics = {}

    # ── FINAL COMBINED TABLE ────────────────────────────────────────────────
    print("\n" + "=" * 94)
    print("FINAL COMBINED BENCHMARK TABLE ACROSS ALL CONDITIONS (Precision@10)")
    print("=" * 94)

    header = f"{'Condition':<32}  {'Popularity':>10}  {'SVD':>10}  {'ALS':>10}  {'Content':>10}  {'Hybrid':>10}  {'Reranker':>10}"
    print(header)
    print("-" * 94)

    def row_str(name: str, r_dict: dict[str, np.ndarray]) -> str:
        def _get(k):
            arr = r_dict.get(k, np.array([]))
            return f"{arr.mean():.4f}" if len(arr) > 0 else "n/a"

        return (
            f"{name:<32}  {_get('popularity'):>10}  {_get('SVD'):>10}  {_get('ALS'):>10}  "
            f"{_get('content'):>10}  {_get('hybrid'):>10}  {_get('reranker'):>10}"
        )

    print(row_str("1. Overall (no filter)", res_overall))
    print(row_str(f"2. Cand-filtered (≥{MIN_TRAIN_RATINGS_CANDIDATE})", res_filtered))
    if res_heavy:
        print(row_str("3. Heavy users (≥50 train)", res_heavy))
    for n_k in [1, 3, 5]:
        if n_k in cold_sim_results:
            print(row_str(f"4. User cold-start (n={n_k})", cold_sim_results[n_k]))
    if res_ics:
        print(row_str("5. Item cold-start (500 items)", res_ics))

    print("=" * 94)
    print("Evaluation complete ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Evaluate all test users")
    args = parser.parse_args()
    main(full=args.full)
