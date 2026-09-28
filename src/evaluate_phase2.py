"""
Phase 2 Evaluation: Popularity vs SVD vs Content-only vs Hybrid.

Fixes and extensions over previous version:
  1. NaN fix: standardize on UNMASKED candidate scores, blend, THEN apply -inf mask.
  2. Candidate filter: MIN_TRAIN_RATINGS_CANDIDATE — only movies with >= that many
     train ratings are candidates for all models.
  3. Cold-start simulation: 2000 users with >= 30 train ratings; for n in {1,3,5}
     keep only n random train ratings for profile / SVD; hold rest for test.
  4. Item cold-start: remove all train ratings for 500 random movies with >= 30 train
     ratings; evaluate only those movies as targets.
  5. 95% bootstrap CI (1000 resamples over users) for every precision headline;
     paired difference (hybrid - SVD) with CI.
  6. Popularity baseline confirmed to backfill to K unseen candidates per user.

Usage:
    python evaluate_phase2.py          # 5,000-user sample
    python evaluate_phase2.py --full   # all test users
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
    MIN_TRAIN_RATINGS_CANDIDATE,
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    N_COMPONENTS,
    RATINGS_CLEAN_PARQUET,
    SVD_SEED,
    TOP_K,
)
from content_model import build_user_profile
from hybrid import save_alpha

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(recs: list[int], liked: set[int], k: int = 10) -> float:
    if not recs or not liked:
        return 0.0
    return sum(1 for c in recs[:k] if c in liked) / k


def top_k_indices(scores: np.ndarray, k: int = 10) -> list[int]:
    """Return column indices of top-k finite scores."""
    finite = np.where(np.isfinite(scores))[0]
    if len(finite) == 0:
        return []
    k = min(k, len(finite))
    part = np.argpartition(scores[finite], -k)[-k:]
    sorted_part = part[np.argsort(scores[finite][part])[::-1]]
    return finite[sorted_part].tolist()


def z_score_unmasked(scores: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    """Standardize using statistics from CANDIDATE (unmasked) positions only.

    This avoids the NaN produced by 0 * -inf when alpha is 0 or 1.
    The mask (-inf) is NOT present in scores here; it is applied AFTER blending.
    """
    valid = candidate_mask & np.isfinite(scores)
    out = np.full_like(scores, 0.0, dtype=np.float64)
    if not np.any(valid):
        return out
    vals = scores[valid]
    mu = np.mean(vals)
    sigma = np.std(vals)
    if sigma < 1e-9:
        sigma = 1.0
    out[valid] = (vals - mu) / sigma
    return out


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lower_95%, upper_95%) via bootstrap over users."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    if rng is None:
        rng = np.random.default_rng(0)
    means = np.array([np.mean(rng.choice(values, size=len(values), replace=True))
                      for _ in range(n_resamples)])
    return float(np.mean(values)), float(np.percentile(means, 100*alpha/2)), float(np.percentile(means, 100*(1-alpha/2)))


def print_ci(label: str, values: np.ndarray, rng: np.random.Generator, width: int = 30) -> None:
    m, lo, hi = bootstrap_ci(values, rng=rng)
    print(f"  {label:<{width}} {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


def paired_diff_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b) using paired resamples."""
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = np.array([
        np.mean(rng.choice(a, size=n, replace=True)) - np.mean(rng.choice(b, size=n, replace=True))
        for _ in range(1000)
    ])
    return float(np.mean(a) - np.mean(b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation engine (reusable for main eval, cold-start sim, item cold-start)
# ──────────────────────────────────────────────────────────────────────────────

def compute_svd_user_vector(
    sparse_row: np.ndarray,  # shape (n_movies,)
    svd: TruncatedSVD,
) -> np.ndarray:
    """Project a single user row through the fitted SVD to get user factors."""
    from scipy.sparse import csr_matrix as _csr
    row_2d = _csr(sparse_row.reshape(1, -1), dtype=np.float32)
    return svd.transform(row_2d)[0]


def evaluate_users(
    user_ids: list[int],
    user_to_idx: dict[int, int],
    user_train_rated: dict[int, set[int]],   # col indices rated in train
    user_train_liked: dict[int, set[int]],   # col indices liked in train
    user_test_liked: dict[int, set[int]],    # col indices liked in test
    train_user_factors: np.ndarray,
    svd: TruncatedSVD,
    movie_embeddings: np.ndarray,
    pop_ranked_cols: np.ndarray,
    candidate_cols: np.ndarray,              # allowed recommendation columns
    candidate_mask: np.ndarray,              # bool mask of candidate_cols
    best_alpha: float,
    top_k: int = TOP_K,
    # Optional: override per-user factors (for cold-start simulation)
    override_factors: dict[int, np.ndarray] | None = None,
    # Optional: restrict test targets to specific col set
    target_cols: set[int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Evaluate popularity / SVD / content / hybrid on given users.
    Returns dict model_name -> array of per-user precision@k.
    """
    pop_p, svd_p, cnt_p, hyb_p = [], [], [], []

    for uid in user_ids:
        u_idx = user_to_idx[uid]
        rated = user_train_rated.get(uid, set())
        liked = user_train_liked.get(uid, set())
        test_liked_set = user_test_liked.get(uid, set())

        if target_cols is not None:
            test_liked_set = test_liked_set & target_cols

        if not test_liked_set:
            continue

        # Candidate set = allowed cols minus already rated
        available = candidate_mask.copy()
        if rated:
            available[list(rated)] = False

        avail_cols = np.where(available)[0]
        if len(avail_cols) == 0:
            pop_p.append(0.0); svd_p.append(0.0); cnt_p.append(0.0); hyb_p.append(0.0)
            continue

        # 1. Popularity — top-k from pop_ranked_cols that are available
        pop_recs = [c for c in pop_ranked_cols if available[c]][:top_k]
        pop_p.append(precision_at_k(pop_recs, test_liked_set, top_k))

        # 2. SVD
        if override_factors is not None and uid in override_factors:
            u_vec = override_factors[uid]
        else:
            u_vec = train_user_factors[u_idx]
        raw_svd = (u_vec @ svd.components_).astype(np.float64)  # shape (n_movies,)

        # z-score on candidates ONLY (before masking)
        z_svd = z_score_unmasked(raw_svd, available)
        # Now mask non-candidates with -inf for recommendation
        masked_svd = np.full(len(raw_svd), -np.inf)
        masked_svd[avail_cols] = z_svd[avail_cols]
        svd_recs = top_k_indices(masked_svd, top_k)
        svd_p.append(precision_at_k(svd_recs, test_liked_set, top_k))

        # 3. Content-only
        profile = build_user_profile(liked, movie_embeddings)
        if profile is not None:
            raw_cnt = (movie_embeddings @ profile).astype(np.float64)
            z_cnt = z_score_unmasked(raw_cnt, available)
            masked_cnt = np.full(len(raw_cnt), -np.inf)
            masked_cnt[avail_cols] = z_cnt[avail_cols]
            cnt_recs = top_k_indices(masked_cnt, top_k)
        else:
            z_cnt = np.zeros(len(raw_svd))
            cnt_recs = []
        cnt_p.append(precision_at_k(cnt_recs, test_liked_set, top_k))

        # 4. Hybrid: blend z-scores FIRST, then mask
        z_hyb_candidates = best_alpha * z_svd[avail_cols] + (1.0 - best_alpha) * z_cnt[avail_cols]
        masked_hyb = np.full(len(raw_svd), -np.inf)
        masked_hyb[avail_cols] = z_hyb_candidates
        hyb_recs = top_k_indices(masked_hyb, top_k)
        hyb_p.append(precision_at_k(hyb_recs, test_liked_set, top_k))

    return {
        "popularity":  np.array(pop_p),
        "SVD":         np.array(svd_p),
        "content":     np.array(cnt_p),
        "hybrid":      np.array(hyb_p),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(full: bool = False) -> None:
    print("=" * 70)
    print("PHASE 2 EVALUATION")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\nLoading ratings and movie embeddings …")
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    movie_embeddings = np.load(MOVIE_EMBEDDINGS_PATH).astype(np.float32)
    movie_ids = np.load(MOVIE_IDS_PATH)
    n_movies = len(movie_ids)
    movie_to_idx: dict[int, int] = {int(mid): i for i, mid in enumerate(movie_ids)}
    print(f"  Ratings: {len(ratings):,} | Catalog movies: {n_movies:,}")

    # ── 2. 70/10/20 split per user ────────────────────────────────────────────
    print("Splitting 70% Train / 10% Val / 20% Test per user …")
    rng = np.random.default_rng(EVAL_SEED)
    train_rows, val_rows, test_rows = [], [], []
    for uid, grp in ratings.groupby("userId"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test  = max(1, int(round(n * 0.20)))
        n_val   = max(1, int(round(n * 0.10)))
        n_train = max(1, n - n_test - n_val)
        # Rebalance if overconstrained
        total_allocated = n_train + n_val + n_test
        if total_allocated > n:
            n_test = max(0, n_test - (total_allocated - n))
        train_rows.append(grp.loc[idx[:n_train]])
        val_rows.append(grp.loc[idx[n_train:n_train+n_val]])
        test_rows.append(grp.loc[idx[n_train+n_val:]])

    train = pd.concat(train_rows, ignore_index=True)
    val   = pd.concat(val_rows,   ignore_index=True)
    test  = pd.concat(test_rows,  ignore_index=True)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ── 3. Index mappings ─────────────────────────────────────────────────────
    users = sorted(ratings["userId"].unique())
    user_to_idx: dict[int, int] = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    # ── 4. Build train sparse matrix & fit SVD ────────────────────────────────
    print(f"\nBuilding sparse matrix and fitting SVD (n_components={N_COMPONENTS}) on train …")
    r_user  = train["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
    r_movie = train["movieId_tmdb"].map(movie_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    r_rating = train["rating"].to_numpy(dtype=np.float32)
    vmask = (r_user >= 0) & (r_movie >= 0)
    train_matrix = csr_matrix(
        (r_rating[vmask], (r_user[vmask], r_movie[vmask])),
        shape=(n_users, n_movies), dtype=np.float32,
    )
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    train_user_factors = svd.fit_transform(train_matrix)

    # ── 5. Movie train rating counts & candidate mask ─────────────────────────
    print("Computing train movie counts …")
    train_movie_counts = np.zeros(n_movies, dtype=np.int32)
    col_counts = (
        train["movieId_tmdb"].map(movie_to_idx).dropna().astype(int).value_counts()
    )
    for col_idx, cnt in col_counts.items():
        if 0 <= col_idx < n_movies:
            train_movie_counts[col_idx] = cnt

    # Candidate mask: movies with >= MIN_TRAIN_RATINGS_CANDIDATE train ratings
    candidate_mask_filtered = train_movie_counts >= MIN_TRAIN_RATINGS_CANDIDATE
    candidate_mask_all      = np.ones(n_movies, dtype=bool)
    n_candidates_filtered = int(candidate_mask_filtered.sum())
    n_candidates_all      = n_movies
    print(f"  Candidate pool (all):      {n_candidates_all:,} movies")
    print(f"  Candidate pool (>= {MIN_TRAIN_RATINGS_CANDIDATE} train ratings): {n_candidates_filtered:,} movies")

    # Popularity baseline ranking (liked-count in train, descending)
    liked_train = train[train["rating"] >= LIKE_THRESHOLD]
    pop_liked_counts = np.zeros(n_movies, dtype=np.int32)
    pl_counts = liked_train["movieId_tmdb"].map(movie_to_idx).dropna().astype(int).value_counts()
    for col_idx, cnt in pl_counts.items():
        if 0 <= col_idx < n_movies:
            pop_liked_counts[col_idx] = cnt
    pop_ranked_cols = np.argsort(pop_liked_counts)[::-1]  # sorted desc

    # ── 6. Per-user train histories ───────────────────────────────────────────
    print("Indexing per-user train histories …")
    user_train_rated: dict[int, set[int]] = {}
    user_train_liked: dict[int, set[int]] = {}
    user_train_count: dict[int, int] = {}
    for uid, grp in train.groupby("userId"):
        cols = {movie_to_idx[m] for m in grp["movieId_tmdb"] if m in movie_to_idx}
        user_train_rated[uid] = cols
        liked_cols = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        user_train_liked[uid] = liked_cols
        user_train_count[uid] = len(grp)

    # ── 7. Alpha tuning on VALIDATION ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALPHA TUNING on VALIDATION (70/10 split, no candidate filter)")
    print("=" * 70)

    val_test_liked: dict[int, set[int]] = {}
    for uid, grp in val.groupby("userId"):
        liked = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        if liked:
            val_test_liked[uid] = liked

    val_users = list(val_test_liked.keys())
    if len(val_users) > EVAL_N_USERS:
        val_users = list(np.random.default_rng(EVAL_SEED).choice(val_users, EVAL_N_USERS, replace=False))
    print(f"  Validating with {len(val_users):,} users who have liked val items …")
    print(f"\n{'alpha':>6}  {'val_p@10':>10}")

    best_alpha = 0.5
    best_val_p10 = -1.0
    alpha_results = {}

    # Precompute raw scores for each val user (no masking yet)
    val_cache = []
    for uid in val_users:
        u_idx = user_to_idx[uid]
        rated = user_train_rated.get(uid, set())
        liked = user_train_liked.get(uid, set())
        avail = np.ones(n_movies, dtype=bool)
        if rated:
            avail[list(rated)] = False
        avail_cols = np.where(avail)[0]

        raw_svd = (train_user_factors[u_idx] @ svd.components_).astype(np.float64)
        z_svd = z_score_unmasked(raw_svd, avail)

        profile = build_user_profile(liked, movie_embeddings)
        if profile is not None:
            raw_cnt = (movie_embeddings @ profile).astype(np.float64)
            z_cnt = z_score_unmasked(raw_cnt, avail)
        else:
            z_cnt = np.zeros(n_movies)

        val_cache.append((uid, avail, avail_cols, z_svd, z_cnt, val_test_liked[uid]))

    candidate_alphas = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]
    for alpha in candidate_alphas:
        p_list = []
        for uid, avail, avail_cols, z_svd, z_cnt, liked_set in val_cache:
            # Blend THEN mask
            z_hyb_cands = alpha * z_svd[avail_cols] + (1.0 - alpha) * z_cnt[avail_cols]
            masked = np.full(n_movies, -np.inf)
            masked[avail_cols] = z_hyb_cands
            recs = top_k_indices(masked, TOP_K)
            p_list.append(precision_at_k(recs, liked_set, TOP_K))
        val_p10 = float(np.mean(p_list))
        alpha_results[alpha] = val_p10
        print(f"  {alpha:>4.1f}  {val_p10:>10.4f}")
        if val_p10 > best_val_p10:
            best_val_p10 = val_p10
            best_alpha = alpha

    print(f"\n→ Best alpha: {best_alpha:.1f}  (val precision@{TOP_K} = {best_val_p10:.4f})")
    save_alpha(best_alpha)

    # Sanity check: alpha=0 == content-only, alpha=1 == SVD-only
    cnt_only = alpha_results.get(0.0, -1)
    svd_only = alpha_results.get(1.0, -1)
    # Compute content-only and SVD-only separately on val_cache
    p_cnt_check = []
    p_svd_check = []
    for uid, avail, avail_cols, z_svd, z_cnt, liked_set in val_cache:
        m_cnt = np.full(n_movies, -np.inf); m_cnt[avail_cols] = z_cnt[avail_cols]
        m_svd = np.full(n_movies, -np.inf); m_svd[avail_cols] = z_svd[avail_cols]
        p_cnt_check.append(precision_at_k(top_k_indices(m_cnt, TOP_K), liked_set, TOP_K))
        p_svd_check.append(precision_at_k(top_k_indices(m_svd, TOP_K), liked_set, TOP_K))
    cnt_check = float(np.mean(p_cnt_check))
    svd_check = float(np.mean(p_svd_check))
    assert abs(cnt_only - cnt_check) < 1e-9, f"NaN bug! alpha=0 ({cnt_only:.6f}) != content-only ({cnt_check:.6f})"
    assert abs(svd_only - svd_check) < 1e-9, f"NaN bug! alpha=1 ({svd_only:.6f}) != SVD-only ({svd_check:.6f})"
    print(f"\n✓ Sanity: alpha=0.0 ({cnt_only:.4f}) == content-only ({cnt_check:.4f})")
    print(f"✓ Sanity: alpha=1.0 ({svd_only:.4f}) == SVD-only ({svd_check:.4f})")

    # ── 8. Test split user indexing ───────────────────────────────────────────
    test_liked_all: dict[int, set[int]] = {}
    for uid, grp in test.groupby("userId"):
        liked = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        if liked:
            test_liked_all[uid] = liked

    test_candidates_all = sorted(test_liked_all.keys())
    if not full and len(test_candidates_all) > EVAL_N_USERS:
        rng_eval = np.random.default_rng(EVAL_SEED + 1)
        test_eval_users = list(rng_eval.choice(test_candidates_all, EVAL_N_USERS, replace=False))
    else:
        test_eval_users = test_candidates_all
    print(f"\nTest users for main eval: {len(test_eval_users):,}")

    bs_rng = np.random.default_rng(EVAL_SEED + 777)

    def print_table(results: dict[str, np.ndarray], title: str, n_users: int) -> None:
        print(f"\n{'='*70}")
        print(f"{title} (N={n_users:,})")
        print(f"{'='*70}")
        print(f"  {'Model':<25}  {'P@10':>8}  {'95% CI':>22}")
        print(f"  {'-'*25}  {'-'*8}  {'-'*22}")
        for name, vals in results.items():
            if len(vals) == 0:
                print(f"  {name:<25}  {'n/a':>8}")
                continue
            m, lo, hi = bootstrap_ci(vals, rng=np.random.default_rng(EVAL_SEED+123))
            print(f"  {name:<25}  {m:>8.4f}  [{lo:.4f}, {hi:.4f}]")
        # Paired hybrid - SVD diff
        if "SVD" in results and "hybrid" in results and len(results["SVD"]) == len(results["hybrid"]):
            svd_v  = results["SVD"]
            hyb_v  = results["hybrid"]
            if len(svd_v) > 0:
                diff_mean, diff_lo, diff_hi = paired_diff_ci(hyb_v, svd_v, np.random.default_rng(EVAL_SEED+456))
                sign = "+" if diff_mean >= 0 else ""
                print(f"  {'hybrid - SVD (paired)':<25}  {sign+f'{diff_mean:.4f}':>8}  [{diff_lo:.4f}, {diff_hi:.4f}]")

    # ── 9. MAIN TEST EVAL — no candidate filter ───────────────────────────────
    print("\nRunning MAIN test evaluation (no candidate filter) …")
    r_all = evaluate_users(
        test_eval_users, user_to_idx,
        user_train_rated, user_train_liked, test_liked_all,
        train_user_factors, svd, movie_embeddings,
        pop_ranked_cols, np.where(candidate_mask_all)[0], candidate_mask_all,
        best_alpha,
    )
    model_names_all = {"popularity": "popularity", "SVD": "SVD", "content": "content-only", "hybrid": f"hybrid (α={best_alpha:.1f})"}
    r_all_named = {model_names_all[k]: v for k, v in r_all.items()}
    print_table(r_all_named, "OVERALL (no candidate filter)", len(r_all["SVD"]))

    # Breakdown by user type
    cold_users = [u for u in test_eval_users if user_train_count.get(u, 0) <= 5]
    heavy_users = [u for u in test_eval_users if user_train_count.get(u, 0) >= 50]

    if cold_users:
        r_cold = evaluate_users(cold_users, user_to_idx, user_train_rated, user_train_liked,
                                test_liked_all, train_user_factors, svd, movie_embeddings,
                                pop_ranked_cols, np.where(candidate_mask_all)[0], candidate_mask_all, best_alpha)
        print_table({model_names_all[k]: v for k, v in r_cold.items()},
                    f"COLD-START users (≤5 train ratings)", len(cold_users))
    else:
        print(f"\n  (No cold-start users ≤5 ratings after MIN_RATINGS={10} filter)")

    if heavy_users:
        r_heavy = evaluate_users(heavy_users, user_to_idx, user_train_rated, user_train_liked,
                                 test_liked_all, train_user_factors, svd, movie_embeddings,
                                 pop_ranked_cols, np.where(candidate_mask_all)[0], candidate_mask_all, best_alpha)
        print_table({model_names_all[k]: v for k, v in r_heavy.items()},
                    f"HEAVY users (≥50 train ratings)", len(heavy_users))

    # ── 10. CANDIDATE FILTER ──────────────────────────────────────────────────
    print(f"\nRunning with candidate filter (>= {MIN_TRAIN_RATINGS_CANDIDATE} train ratings) …")
    r_filt = evaluate_users(
        test_eval_users, user_to_idx,
        user_train_rated, user_train_liked, test_liked_all,
        train_user_factors, svd, movie_embeddings,
        pop_ranked_cols, np.where(candidate_mask_filtered)[0], candidate_mask_filtered,
        best_alpha,
    )
    r_filt_named = {model_names_all[k]: v for k, v in r_filt.items()}
    print_table(r_filt_named, f"OVERALL (candidate filter ≥ {MIN_TRAIN_RATINGS_CANDIDATE} train ratings)", len(r_filt["SVD"]))

    # ── 11. COLD-START SIMULATION ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COLD-START SIMULATION (2000 users with ≥30 train ratings, truncated history)")
    print("=" * 70)

    sim_candidates = [u for u in user_to_idx if user_train_count.get(u, 0) >= 30]
    sim_rng = np.random.default_rng(EVAL_SEED + 2)
    sim_users = list(sim_rng.choice(sim_candidates, size=min(2000, len(sim_candidates)), replace=False))
    print(f"  Pool: {len(sim_candidates):,} users with ≥30 train ratings → sampled {len(sim_users):,}")

    for n_keep in [1, 3, 5]:
        print(f"\n  n_train={n_keep} ratings kept:")
        sim_override: dict[int, np.ndarray] = {}
        sim_train_rated_trunc: dict[int, set[int]] = {}
        sim_train_liked_trunc: dict[int, set[int]] = {}
        sim_test_liked: dict[int, set[int]] = {}

        for uid in sim_users:
            all_rated_list = list(user_train_rated.get(uid, set()))
            all_liked_list = list(user_train_liked.get(uid, set()))
            if len(all_rated_list) < n_keep:
                continue

            # Randomly pick n_keep train ratings
            chosen_idx = sim_rng.choice(len(all_rated_list), size=n_keep, replace=False)
            chosen_cols = set(all_rated_list[i] for i in chosen_idx)
            chosen_liked = chosen_cols & user_train_liked.get(uid, set())

            # Use the REST (train - chosen) + actual test as the "test held-out" set
            held_out_liked = (user_train_liked.get(uid, set()) - chosen_liked) | test_liked_all.get(uid, set())
            if not held_out_liked:
                continue

            sim_train_rated_trunc[uid] = chosen_cols
            sim_train_liked_trunc[uid] = chosen_liked
            sim_test_liked[uid] = held_out_liked

            # Compute SVD user vector from truncated row
            trunc_row = np.zeros(n_movies, dtype=np.float32)
            for col_i in chosen_cols:
                trunc_row[col_i] = 3.5  # neutral rating for known items
            sim_override[uid] = compute_svd_user_vector(trunc_row, svd)

        sim_eval_users = [u for u in sim_users if u in sim_test_liked]
        r_sim = evaluate_users(
            sim_eval_users, user_to_idx,
            sim_train_rated_trunc, sim_train_liked_trunc, sim_test_liked,
            train_user_factors, svd, movie_embeddings,
            pop_ranked_cols, np.where(candidate_mask_all)[0], candidate_mask_all,
            best_alpha,
            override_factors=sim_override,
        )
        n_sim_eval = len(r_sim["SVD"])
        for mkey, mname in [("popularity","popularity"),("SVD","SVD"),("content","content-only"),("hybrid",f"hybrid (α={best_alpha:.1f})")]:
            if len(r_sim[mkey]) > 0:
                m, lo, hi = bootstrap_ci(r_sim[mkey], rng=np.random.default_rng(EVAL_SEED+n_keep))
                print(f"    {mname:<25}  {m:.4f}  [{lo:.4f}, {hi:.4f}]  (N={n_sim_eval})")

    # ── 12. ITEM COLD-START ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ITEM COLD-START (500 movies removed from train, only they are targets)")
    print("=" * 70)

    # Movies with >= 30 train ratings
    popular_movie_cols = np.where(train_movie_counts >= 30)[0]
    item_rng = np.random.default_rng(EVAL_SEED + 3)
    n_remove = min(500, len(popular_movie_cols))
    removed_movie_cols: set[int] = set(
        item_rng.choice(popular_movie_cols, size=n_remove, replace=False).tolist()
    )
    print(f"  Removed {len(removed_movie_cols):,} movies from train (retaining their test appearances as targets)")
    print(f"  Rebuilding SVD without those movies' ratings …")

    # Build a new train matrix without ratings for removed movies
    vmask2 = vmask & ~np.isin(r_movie, list(removed_movie_cols))
    train_matrix2 = csr_matrix(
        (r_rating[vmask2], (r_user[vmask2], r_movie[vmask2])),
        shape=(n_users, n_movies), dtype=np.float32,
    )
    svd2 = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    train_user_factors2 = svd2.fit_transform(train_matrix2)

    # Rebuild user train history without removed movies
    user_train_rated2 = {
        uid: cols - removed_movie_cols
        for uid, cols in user_train_rated.items()
    }
    user_train_liked2 = {
        uid: cols - removed_movie_cols
        for uid, cols in user_train_liked.items()
    }

    # Test: users who liked at least one removed movie in test
    item_cs_liked: dict[int, set[int]] = {}
    for uid, liked_set in test_liked_all.items():
        targets = liked_set & removed_movie_cols
        if targets:
            item_cs_liked[uid] = targets

    item_cs_users = list(item_cs_liked.keys())
    print(f"  Test users with ≥1 liked removed movie: {len(item_cs_users):,}")
    if not item_cs_users:
        print("  (No test users had liked removed movies — skipping item cold-start)")
    else:
        r_ics = evaluate_users(
            item_cs_users, user_to_idx,
            user_train_rated2, user_train_liked2, item_cs_liked,
            train_user_factors2, svd2, movie_embeddings,
            pop_ranked_cols, np.where(candidate_mask_all)[0], candidate_mask_all,
            best_alpha,
            target_cols=removed_movie_cols,
        )
        print_table(
            {model_names_all[k]: v for k, v in r_ics.items()},
            "ITEM COLD-START (only removed movies as targets)",
            len(r_ics["SVD"]),
        )

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Evaluate all test users instead of sample")
    args = parser.parse_args()
    main(full=args.full)
