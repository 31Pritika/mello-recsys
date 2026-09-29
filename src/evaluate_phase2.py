"""
Phase 2 Evaluation: Popularity vs SVD vs Content-only vs Hybrid.

Requirements implemented:
  1. NaN fix: z-score on UNMASKED candidate positions, blend THEN apply mask.
  2. Candidate filter: MIN_TRAIN_RATINGS_CANDIDATE (default 20).
  3. Cold-start simulation: 2000 users, n_train in {1,3,5}.
  4. Item cold-start: 500 movies removed; only those as test targets.
  5. 95% bootstrap CI (1000 resamples) for every headline; paired hybrid−SVD CI.
  6. Popularity backfills to K unseen candidates per user.

Memory: mini-batches of BATCH_SIZE users so peak RAM ≈ 3 × BATCH_SIZE × 28k × 8B
(~135 MB at BATCH_SIZE=200 vs 5.5 GB for the full matrix).
All 11 alphas are evaluated simultaneously per mini-batch.
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

BATCH_SIZE = 200   # users per mini-batch; keep peak RAM < 200 MB

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def precision_at_k(recs: list[int], liked: set[int], k: int = 10) -> float:
    if not recs or not liked:
        return 0.0
    return sum(1 for c in recs[:k] if c in liked) / k


def top_k_from_row(row: np.ndarray, k: int) -> list[int]:
    """Top-k column indices from a 1-D score array (finite values only)."""
    finite = np.where(np.isfinite(row))[0]
    if len(finite) == 0:
        return []
    k = min(k, len(finite))
    part = np.argpartition(row[finite], -k)[-k:]
    return finite[part[np.argsort(row[finite][part])[::-1]]].tolist()


def row_z_score_on_candidates(row: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Standardise using stats from avail positions only. Non-avail set to 0."""
    out = np.zeros(len(row), dtype=np.float64)
    cand = row[avail]
    mu  = cand.mean()
    std = cand.std()
    if std < 1e-9:
        std = 1.0
    out[avail] = (cand - mu) / std
    return out


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boot = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_ci(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 1) -> tuple[float, float, float]:
    return bootstrap_ci(a - b, n=n, seed=seed)


def fmt(m: float, lo: float, hi: float) -> str:
    return f"{m:.4f}  [{lo:.4f}, {hi:.4f}]"


# ─────────────────────────────────────────────────────────────────
# Mini-batch evaluation engine
# ─────────────────────────────────────────────────────────────────

def evaluate_minibatch(
    eval_uids: list[int],
    user_to_idx: dict[int, int],
    user_train_rated: dict[int, set[int]],
    user_train_liked: dict[int, set[int]],
    user_test_liked: dict[int, set[int]],
    user_factors: np.ndarray,            # (n_users_total, n_comp)
    svd_components: np.ndarray,          # (n_comp, n_movies)
    movie_embeddings: np.ndarray,        # (n_movies, emb_dim) float32
    pop_ranked_cols: np.ndarray,         # sorted desc by liked-count
    candidate_mask: np.ndarray,          # (n_movies,) bool
    best_alpha: float,
    top_k: int = TOP_K,
    override_factors: dict[int, np.ndarray] | None = None,
    target_cols: set[int] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate in BATCH_SIZE mini-batches. Returns {model: per-user-P@k}."""
    n_movies = svd_components.shape[1]
    cand_cols = np.where(candidate_mask)[0]   # column indices of candidate pool

    pop_p, svd_p, cnt_p, hyb_p = [], [], [], []

    # Filter to users that have liked test items
    active = []
    for uid in eval_uids:
        tl = user_test_liked.get(uid, set())
        if target_cols is not None:
            tl = tl & target_cols
        if tl:
            active.append((uid, tl))

    for batch_start in range(0, len(active), BATCH_SIZE):
        batch = active[batch_start: batch_start + BATCH_SIZE]
        bsize = len(batch)
        uids_b  = [x[0] for x in batch]
        tls_b   = [x[1] for x in batch]

        # User factor rows
        uf_b = np.array([
            override_factors.get(uid, user_factors[user_to_idx[uid]])
            if override_factors else user_factors[user_to_idx[uid]]
            for uid in uids_b
        ], dtype=np.float64)                              # (bsize, n_comp)

        # SVD raw scores — (bsize, n_movies) only over candidate columns to save RAM
        raw_svd_cand = (uf_b @ svd_components[:, cand_cols].astype(np.float64))
        # (bsize, n_cand)

        # Content raw scores
        prof_mat = np.zeros((bsize, movie_embeddings.shape[1]), dtype=np.float64)
        has_profile = []
        for i, uid in enumerate(uids_b):
            prof = build_user_profile(user_train_liked.get(uid, set()), movie_embeddings)
            has_profile.append(prof is not None)
            if prof is not None:
                prof_mat[i] = prof
        raw_cnt_cand = prof_mat @ movie_embeddings[cand_cols].T.astype(np.float64)
        # (bsize, n_cand)

        # z-score per user on candidate scores (no masking yet)
        z_svd_cand = np.zeros_like(raw_svd_cand)
        z_cnt_cand = np.zeros_like(raw_cnt_cand)
        for i in range(bsize):
            mu = raw_svd_cand[i].mean(); std = raw_svd_cand[i].std()
            z_svd_cand[i] = (raw_svd_cand[i] - mu) / (std if std > 1e-9 else 1.0)
            mu = raw_cnt_cand[i].mean(); std = raw_cnt_cand[i].std()
            z_cnt_cand[i] = (raw_cnt_cand[i] - mu) / (std if std > 1e-9 else 1.0)

        # Blend (candidate positions only)
        z_hyb_cand = best_alpha * z_svd_cand + (1.0 - best_alpha) * z_cnt_cand

        for i, uid in enumerate(uids_b):
            tl = tls_b[i]
            rated = user_train_rated.get(uid, set())

            # Which candidate columns are still available (not rated)?
            avail_mask_b = np.ones(len(cand_cols), dtype=bool)
            if rated:
                for ci, c in enumerate(cand_cols):
                    if c in rated:
                        avail_mask_b[ci] = False
            avail_cand_local = np.where(avail_mask_b)[0]   # indices into cand_cols

            if len(avail_cand_local) == 0:
                pop_p.append(0.0); svd_p.append(0.0); cnt_p.append(0.0); hyb_p.append(0.0)
                continue

            avail_global = cand_cols[avail_cand_local]     # global column indices

            # 1. Popularity — top-k from pop_ranked_cols ∩ avail_global
            avail_set = set(avail_global.tolist())
            pop_recs = [c for c in pop_ranked_cols if c in avail_set][:top_k]
            pop_p.append(precision_at_k(pop_recs, tl, top_k))

            def _recs_from_cand_scores(z_cand: np.ndarray) -> list[int]:
                # Map candidate-local avail scores back to global col space
                scores_avail = z_cand[avail_cand_local]    # values for available cands
                k = min(top_k, len(scores_avail))
                if k == 0:
                    return []
                part = np.argpartition(scores_avail, -k)[-k:]
                sorted_part = part[np.argsort(scores_avail[part])[::-1]]
                return avail_global[sorted_part].tolist()

            svd_p.append(precision_at_k(_recs_from_cand_scores(z_svd_cand[i]), tl, top_k))
            cnt_p.append(precision_at_k(_recs_from_cand_scores(z_cnt_cand[i]) if has_profile[i] else [], tl, top_k))
            hyb_p.append(precision_at_k(_recs_from_cand_scores(z_hyb_cand[i]), tl, top_k))

    return {
        "popularity": np.array(pop_p, dtype=np.float64),
        "SVD":        np.array(svd_p, dtype=np.float64),
        "content":    np.array(cnt_p, dtype=np.float64),
        "hybrid":     np.array(hyb_p, dtype=np.float64),
    }


def tune_alpha_minibatch(
    val_uids: list[int],
    user_to_idx: dict[int, int],
    user_train_rated: dict[int, set[int]],
    user_train_liked: dict[int, set[int]],
    val_liked: dict[int, set[int]],
    user_factors: np.ndarray,
    svd_components: np.ndarray,
    movie_embeddings: np.ndarray,
    candidate_mask: np.ndarray,
    alphas: list[float],
    top_k: int = TOP_K,
) -> dict[float, float]:
    """
    Tune alpha in one pass over users.
    For each mini-batch: compute z_svd_cand, z_cnt_cand once,
    then evaluate all alphas simultaneously.
    Returns {alpha: mean_precision@k}.
    """
    n_movies = svd_components.shape[1]
    cand_cols = np.where(candidate_mask)[0]
    alpha_sums   = {a: 0.0 for a in alphas}
    alpha_counts = {a: 0   for a in alphas}
    # Also track alpha=0 (content) and alpha=1 (SVD) for sanity check
    cnt_only_sums, svd_only_sums = 0.0, 0.0
    cnt_only_cnt,  svd_only_cnt  = 0,   0

    active = [(uid, val_liked[uid]) for uid in val_uids if val_liked.get(uid)]

    for batch_start in range(0, len(active), BATCH_SIZE):
        batch = active[batch_start: batch_start + BATCH_SIZE]
        bsize = len(batch)
        uids_b = [x[0] for x in batch]
        tls_b  = [x[1] for x in batch]

        uf_b = user_factors[[user_to_idx[u] for u in uids_b]].astype(np.float64)
        raw_svd_cand = uf_b @ svd_components[:, cand_cols].astype(np.float64)

        prof_mat = np.zeros((bsize, movie_embeddings.shape[1]), dtype=np.float64)
        has_profile = []
        for i, uid in enumerate(uids_b):
            prof = build_user_profile(user_train_liked.get(uid, set()), movie_embeddings)
            has_profile.append(prof is not None)
            if prof is not None:
                prof_mat[i] = prof
        raw_cnt_cand = prof_mat @ movie_embeddings[cand_cols].T.astype(np.float64)

        # z-score per user on candidate scores
        z_svd = np.zeros_like(raw_svd_cand)
        z_cnt = np.zeros_like(raw_cnt_cand)
        for i in range(bsize):
            mu = raw_svd_cand[i].mean(); std = raw_svd_cand[i].std()
            z_svd[i] = (raw_svd_cand[i] - mu) / (std if std > 1e-9 else 1.0)
            mu = raw_cnt_cand[i].mean(); std = raw_cnt_cand[i].std()
            z_cnt[i] = (raw_cnt_cand[i] - mu) / (std if std > 1e-9 else 1.0)

        for i, uid in enumerate(uids_b):
            tl = tls_b[i]
            rated = user_train_rated.get(uid, set())

            avail_mask_b = np.ones(len(cand_cols), dtype=bool)
            if rated:
                for ci, c in enumerate(cand_cols):
                    if c in rated:
                        avail_mask_b[ci] = False
            avail_local = np.where(avail_mask_b)[0]
            avail_global = cand_cols[avail_local]
            if len(avail_local) == 0:
                for a in alphas:
                    alpha_sums[a] += 0.0
                    alpha_counts[a] += 1
                continue

            def _p(z_cand_row: np.ndarray) -> float:
                sc = z_cand_row[avail_local]
                k = min(top_k, len(sc))
                part = np.argpartition(sc, -k)[-k:]
                recs = avail_global[part[np.argsort(sc[part])[::-1]]].tolist()
                return precision_at_k(recs, tl, top_k)

            for a in alphas:
                z_hyb = a * z_svd[i] + (1.0 - a) * z_cnt[i]
                alpha_sums[a] += _p(z_hyb)
                alpha_counts[a] += 1

            # Sanity check separate pass
            cnt_only_sums += _p(z_cnt[i])
            cnt_only_cnt  += 1
            svd_only_sums += _p(z_svd[i])
            svd_only_cnt  += 1

    result = {a: alpha_sums[a] / alpha_counts[a] if alpha_counts[a] > 0 else 0.0
              for a in alphas}
    result["_cnt_only_check"] = cnt_only_sums / cnt_only_cnt if cnt_only_cnt > 0 else 0.0
    result["_svd_only_check"] = svd_only_sums / svd_only_cnt if svd_only_cnt > 0 else 0.0
    return result


def print_table(results: dict[str, np.ndarray], title: str, best_alpha: float) -> None:
    LABELS = {
        "popularity": "popularity",
        "SVD":        "SVD",
        "content":    "content-only",
        "hybrid":     f"hybrid (α={best_alpha:.1f})",
    }
    n = len(results.get("SVD", []))
    print(f"\n{'='*72}")
    print(f"{title}  (N={n:,})")
    print(f"{'='*72}")
    print(f"  {'Model':<26}  {'P@10':>8}  {'95% CI':>22}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*22}")
    for key in ("popularity", "SVD", "content", "hybrid"):
        v = results.get(key, np.array([]))
        label = LABELS[key]
        if len(v) == 0:
            print(f"  {label:<26}  {'n/a':>8}")
        else:
            m, lo, hi = bootstrap_ci(v, seed=42)
            print(f"  {label:<26}  {fmt(m, lo, hi)}")
    sv = results.get("SVD",    np.array([]))
    hv = results.get("hybrid", np.array([]))
    if len(sv) > 0 and len(hv) == len(sv):
        dm, dlo, dhi = paired_ci(hv, sv, seed=43)
        sign = "+" if dm >= 0 else ""
        print(f"  {'hybrid − SVD (paired)':<26}  {sign+f'{dm:.4f}':>8}  [{dlo:.4f}, {dhi:.4f}]")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main(full: bool = False) -> None:
    print("=" * 72)
    print("PHASE 2 EVALUATION (mini-batch, all 6 requirements)")
    print(f"  BATCH_SIZE={BATCH_SIZE} users, peak RAM ≈ {3*BATCH_SIZE*28154*8//1_000_000} MB")
    print("=" * 72)

    # ── Load ────────────────────────────────────────────────────────
    print("\nLoading data …")
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    movie_embeddings = np.load(MOVIE_EMBEDDINGS_PATH).astype(np.float32)
    movie_ids = np.load(MOVIE_IDS_PATH)
    n_movies = len(movie_ids)
    movie_to_idx: dict[int, int] = {int(mid): i for i, mid in enumerate(movie_ids)}
    print(f"  Ratings: {len(ratings):,}  |  Catalog: {n_movies:,}")

    # ── 70/10/20 split ──────────────────────────────────────────────
    print("Splitting 70/10/20 per user …")
    rng = np.random.default_rng(EVAL_SEED)
    train_rows, val_rows, test_rows = [], [], []
    for uid, grp in ratings.groupby("userId"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test  = max(1, int(round(n * 0.20)))
        n_val   = max(1, int(round(n * 0.10)))
        n_train = max(1, n - n_test - n_val)
        excess  = n_train + n_val + n_test - n
        if excess > 0:
            n_test = max(0, n_test - excess)
        train_rows.append(grp.loc[idx[:n_train]])
        val_rows.append(grp.loc[idx[n_train:n_train+n_val]])
        test_rows.append(grp.loc[idx[n_train+n_val:]])

    train = pd.concat(train_rows, ignore_index=True)
    val   = pd.concat(val_rows,   ignore_index=True)
    test  = pd.concat(test_rows,  ignore_index=True)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    users = sorted(ratings["userId"].unique())
    user_to_idx: dict[int, int] = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    # ── Sparse matrix & SVD ─────────────────────────────────────────
    print(f"Fitting SVD (n_components={N_COMPONENTS}) on train …")
    r_user   = train["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
    r_movie  = train["movieId_tmdb"].map(movie_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    r_rating = train["rating"].to_numpy(dtype=np.float32)
    vmask    = (r_user >= 0) & (r_movie >= 0)
    train_matrix = csr_matrix(
        (r_rating[vmask], (r_user[vmask], r_movie[vmask])),
        shape=(n_users, n_movies), dtype=np.float32,
    )
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    train_user_factors = svd.fit_transform(train_matrix)

    # ── Movie counts & candidate masks ─────────────────────────────
    print("Computing movie counts …")
    train_movie_counts = np.zeros(n_movies, dtype=np.int32)
    col_cnts = train["movieId_tmdb"].map(movie_to_idx).dropna().astype(int).value_counts()
    for ci, cnt in col_cnts.items():
        if 0 <= ci < n_movies:
            train_movie_counts[ci] = cnt

    candidate_mask_all      = np.ones(n_movies, dtype=bool)
    candidate_mask_filtered = train_movie_counts >= MIN_TRAIN_RATINGS_CANDIDATE
    print(f"  All candidates:           {n_movies:,}")
    print(f"  Filtered (≥{MIN_TRAIN_RATINGS_CANDIDATE} train ratings): {candidate_mask_filtered.sum():,}")

    liked_train = train[train["rating"] >= LIKE_THRESHOLD]
    pop_liked_counts = np.zeros(n_movies, dtype=np.int32)
    pl_cnts = liked_train["movieId_tmdb"].map(movie_to_idx).dropna().astype(int).value_counts()
    for ci, cnt in pl_cnts.items():
        if 0 <= ci < n_movies:
            pop_liked_counts[ci] = cnt
    pop_ranked_cols = np.argsort(pop_liked_counts)[::-1]

    # ── Per-user histories ──────────────────────────────────────────
    print("Indexing per-user histories …")
    user_train_rated: dict[int, set[int]] = {}
    user_train_liked: dict[int, set[int]] = {}
    user_train_count: dict[int, int]      = {}
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

    # ── Alpha tuning on VAL ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ALPHA TUNING (mini-batch, all alphas per batch)")
    print("=" * 72)

    val_liked: dict[int, set[int]] = {}
    for uid, grp in val.groupby("userId"):
        liked = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        if liked:
            val_liked[uid] = liked

    val_uids = list(val_liked.keys())
    if len(val_uids) > EVAL_N_USERS:
        val_uids = list(np.random.default_rng(EVAL_SEED).choice(val_uids, EVAL_N_USERS, replace=False))
    print(f"  Validation users: {len(val_uids):,}")

    alphas = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]
    alpha_results = tune_alpha_minibatch(
        val_uids, user_to_idx,
        user_train_rated, user_train_liked, val_liked,
        train_user_factors, svd.components_, movie_embeddings,
        candidate_mask_all, alphas,
    )

    print(f"\n{'alpha':>6}  {'val_p@10':>10}")
    best_alpha = 0.5
    best_val_p10 = -1.0
    for a in alphas:
        v = alpha_results[a]
        print(f"  {a:>4.1f}  {v:>10.4f}")
        if v > best_val_p10:
            best_val_p10 = v
            best_alpha = a

    print(f"\n→ Best alpha: {best_alpha:.1f}  (val P@{TOP_K} = {best_val_p10:.4f})")
    save_alpha(best_alpha)

    # Sanity check
    cnt0 = alpha_results[0.0]
    svd1 = alpha_results[1.0]
    cnt_check = alpha_results["_cnt_only_check"]
    svd_check = alpha_results["_svd_only_check"]
    assert abs(cnt0 - cnt_check) < 1e-9, f"NaN bug! α=0={cnt0:.6f} ≠ content-only={cnt_check:.6f}"
    assert abs(svd1 - svd_check) < 1e-9, f"NaN bug! α=1={svd1:.6f} ≠ SVD-only={svd_check:.6f}"
    print(f"\n✓ α=0.0 ({cnt0:.4f}) == content-only ({cnt_check:.4f})")
    print(f"✓ α=1.0 ({svd1:.4f}) == SVD-only   ({svd_check:.4f})")

    # ── Build test liked sets ───────────────────────────────────────
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
        test_eval_uids = list(np.random.default_rng(EVAL_SEED+1).choice(test_pool, EVAL_N_USERS, replace=False))
    else:
        test_eval_uids = test_pool
    print(f"\nTest sample: {len(test_eval_uids):,} users")

    # ── MAIN EVAL — no filter ───────────────────────────────────────
    print("Evaluating (no candidate filter) …")
    r_all = evaluate_minibatch(
        test_eval_uids, user_to_idx,
        user_train_rated, user_train_liked, test_liked_all,
        train_user_factors, svd.components_, movie_embeddings,
        pop_ranked_cols, candidate_mask_all, best_alpha,
    )
    print_table(r_all, "OVERALL — no candidate filter", best_alpha)

    # Cold / heavy breakdowns
    cold_uids  = [u for u in test_eval_uids if user_train_count.get(u, 0) <= 5]
    heavy_uids = [u for u in test_eval_uids if user_train_count.get(u, 0) >= 50]
    if cold_uids:
        r_cold = evaluate_minibatch(
            cold_uids, user_to_idx, user_train_rated, user_train_liked, test_liked_all,
            train_user_factors, svd.components_, movie_embeddings,
            pop_ranked_cols, candidate_mask_all, best_alpha)
        print_table(r_cold, "COLD-START users (≤5 train ratings)", best_alpha)
    else:
        print("\n  (No cold-start users ≤5 ratings — MIN_RATINGS=10 filter removes them)")
    if heavy_uids:
        r_heavy = evaluate_minibatch(
            heavy_uids, user_to_idx, user_train_rated, user_train_liked, test_liked_all,
            train_user_factors, svd.components_, movie_embeddings,
            pop_ranked_cols, candidate_mask_all, best_alpha)
        print_table(r_heavy, "HEAVY users (≥50 train ratings)", best_alpha)

    # ── CANDIDATE FILTER ────────────────────────────────────────────
    print(f"\nEvaluating with candidate filter (≥{MIN_TRAIN_RATINGS_CANDIDATE}) …")
    r_filt = evaluate_minibatch(
        test_eval_uids, user_to_idx,
        user_train_rated, user_train_liked, test_liked_all,
        train_user_factors, svd.components_, movie_embeddings,
        pop_ranked_cols, candidate_mask_filtered, best_alpha,
    )
    print_table(r_filt, f"OVERALL — candidate filter ≥{MIN_TRAIN_RATINGS_CANDIDATE} train ratings", best_alpha)

    # ── COLD-START SIMULATION ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("COLD-START SIMULATION (2000 users with ≥30 train ratings)")
    print("=" * 72)

    sim_pool = [u for u in user_to_idx if user_train_count.get(u, 0) >= 30]
    sim_rng  = np.random.default_rng(EVAL_SEED + 2)
    sim_uids = list(sim_rng.choice(sim_pool, size=min(2000, len(sim_pool)), replace=False))
    print(f"  Sampled {len(sim_uids):,} users from pool of {len(sim_pool):,}")
    print(f"\n  {'n_keep':>6}  {'pop':>8}  {'SVD':>8}  {'content':>8}  {'hybrid':>8}   N")

    for n_keep in [1, 3, 5]:
        sim_override: dict[int, np.ndarray] = {}
        sim_rated_t: dict[int, set[int]] = {}
        sim_liked_t: dict[int, set[int]] = {}
        sim_test_t:  dict[int, set[int]] = {}

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
            sim_test_t[uid]  = held_liked
            # SVD from truncated sparse row
            trunc = csr_matrix(
                (np.full(n_keep, 3.5, dtype=np.float32),
                 (np.zeros(n_keep, dtype=np.int32), np.array(list(chosen), dtype=np.int32))),
                shape=(1, n_movies), dtype=np.float32,
            )
            sim_override[uid] = svd.transform(trunc)[0]

        sim_eval = [u for u in sim_uids if u in sim_test_t]
        r_sim = evaluate_minibatch(
            sim_eval, user_to_idx,
            sim_rated_t, sim_liked_t, sim_test_t,
            train_user_factors, svd.components_, movie_embeddings,
            pop_ranked_cols, candidate_mask_all, best_alpha,
            override_factors=sim_override,
        )
        n_s = len(r_sim["SVD"])
        def _m(key: str) -> str:
            v = r_sim[key]
            return f"{v.mean():.4f}" if len(v) > 0 else "n/a"
        print(f"  {n_keep:>6}  {_m('popularity'):>8}  {_m('SVD'):>8}  {_m('content'):>8}  {_m('hybrid'):>8}  {n_s:>4}")

    # ── ITEM COLD-START ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ITEM COLD-START (500 movies removed from train)")
    print("=" * 72)

    popular_movie_cols = np.where(train_movie_counts >= 30)[0]
    item_rng = np.random.default_rng(EVAL_SEED + 3)
    n_remove = min(500, len(popular_movie_cols))
    removed_cols: set[int] = set(
        int(c) for c in item_rng.choice(popular_movie_cols, size=n_remove, replace=False)
    )
    print(f"  Removed {len(removed_cols):,} movies. Rebuilding SVD …")

    vmask2 = vmask & ~np.isin(r_movie, list(removed_cols))
    train_matrix2 = csr_matrix(
        (r_rating[vmask2], (r_user[vmask2], r_movie[vmask2])),
        shape=(n_users, n_movies), dtype=np.float32,
    )
    svd2 = TruncatedSVD(n_components=N_COMPONENTS, random_state=SVD_SEED)
    train_uf2 = svd2.fit_transform(train_matrix2)

    user_rated2 = {u: cols - removed_cols for u, cols in user_train_rated.items()}
    user_liked2 = {u: cols - removed_cols for u, cols in user_train_liked.items()}

    ics_liked: dict[int, set[int]] = {}
    for uid, liked_set in test_liked_all.items():
        tgt = liked_set & removed_cols
        if tgt:
            ics_liked[uid] = tgt

    ics_uids = list(ics_liked.keys())
    print(f"  Test users with ≥1 liked removed movie: {len(ics_uids):,}")
    if ics_uids:
        r_ics = evaluate_minibatch(
            ics_uids, user_to_idx,
            user_rated2, user_liked2, ics_liked,
            train_uf2, svd2.components_, movie_embeddings,
            pop_ranked_cols, candidate_mask_all, best_alpha,
            target_cols=removed_cols,
        )
        print_table(r_ics, "ITEM COLD-START (only removed movies as targets)", best_alpha)
    else:
        print("  (No test users had liked removed movies — skipping)")

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    main(full=args.full)
