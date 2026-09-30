"""
Learning-to-Rank (LTR) Reranker using LightGBM.

Features:
- Built strictly on VALIDATION users (never touches test set).
- Positives: items liked (rating >= 4.0) by the user in validation.
- Negatives: ~20 unrated items per user, sampled with popularity weighting
  (proportional to train rating count).
- 6 features per (user, movie) pair:
  1. als_score: dot product of user ALS factor and movie ALS factor.
  2. content_cosine_score: cosine similarity between user profile and movie embedding.
  3. log1p(movie_train_rating_count): log1p of total train ratings for movie.
  4. mean_train_rating_for_movie: average rating of movie in train (or global mean).
  5. genre_overlap_count: count of shared genres between user's liked-genre profile and movie.
  6. is_item_cold: 1.0 if train rating count < 20, else 0.0.
- Trained with LGBMRanker (objective="lambdarank", metric="ndcg@10") and early stopping.
- Saves model to model_artifacts/reranker_lgbm.pkl.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

import lightgbm as lgb
import numpy as np
import pandas as pd

from als_model import load_als_artifacts
from config import (
    EVAL_SEED,
    LIKE_THRESHOLD,
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    MOVIES_CLEAN_PARQUET,
    RATINGS_CLEAN_PARQUET,
    RERANKER_MODEL_PATH,
)
from content_model import build_user_profile

FEATURE_NAMES = [
    "als_score",
    "content_cosine_score",
    "log1p_train_count",
    "mean_train_rating",
    "genre_overlap",
    "is_item_cold",
]


class PrecomputedMetadata:
    """Cached global movie features and genre lookups to avoid repeated compute."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        movie_ids: np.ndarray,
        movie_to_idx: dict[int, int],
        movies_df: pd.DataFrame,
    ) -> None:
        self.n_movies = len(movie_ids)
        self.movie_ids = movie_ids
        self.movie_to_idx = movie_to_idx

        # 1. Movie train counts & log1p
        train_counts = np.zeros(self.n_movies, dtype=np.float64)
        c_series = train_df["movieId_tmdb"].map(movie_to_idx).dropna().astype(int).value_counts()
        for ci, cnt in c_series.items():
            if 0 <= ci < self.n_movies:
                train_counts[ci] = cnt
        self.train_counts = train_counts
        self.log1p_train_counts = np.log1p(train_counts)
        self.is_item_cold = (train_counts < 20).astype(np.float64)

        # 2. Movie mean train rating
        global_mean = float(train_df["rating"].mean()) if len(train_df) > 0 else 3.5
        mean_ratings = np.full(self.n_movies, global_mean, dtype=np.float64)
        m_grouped = train_df.groupby("movieId_tmdb")["rating"].mean()
        for mid, m_val in m_grouped.items():
            if mid in movie_to_idx:
                mean_ratings[movie_to_idx[mid]] = float(m_val)
        self.mean_train_ratings = mean_ratings
        self.global_mean_rating = global_mean

        # 3. Movie genres
        self.movie_genres: list[set[str]] = [set() for _ in range(self.n_movies)]
        for _, row in movies_df.iterrows():
            mid = int(row["id"])
            if mid in movie_to_idx:
                c_idx = movie_to_idx[mid]
                g_str = str(row["genres"]) if pd.notna(row["genres"]) else ""
                self.movie_genres[c_idx] = {g.strip().lower() for g in g_str.split(",") if g.strip()}


def build_user_liked_genres(
    liked_movie_col_indices: set[int] | list[int],
    movie_genres: list[set[str]],
) -> set[str]:
    """Union of genres for movies the user liked in train."""
    res: set[str] = set()
    for col in liked_movie_col_indices:
        if 0 <= col < len(movie_genres):
            res.update(movie_genres[col])
    return res


def extract_pair_features(
    candidate_cols: list[int] | np.ndarray,
    user_als_factor: Optional[np.ndarray],
    user_profile: Optional[np.ndarray],
    user_liked_genres: set[str],
    als_item_factors: np.ndarray,
    movie_embeddings: np.ndarray,
    meta: PrecomputedMetadata,
) -> np.ndarray:
    """
    Extract (n_candidates, 6) feature matrix for given user and candidate movie column indices.
    """
    cand = np.asarray(candidate_cols, dtype=np.int32)
    n_c = len(cand)
    if n_c == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)

    # 1. als_score
    if user_als_factor is not None:
        cand_item_factors = als_item_factors[cand].astype(np.float64)
        als_scores = cand_item_factors @ user_als_factor.astype(np.float64)
    else:
        als_scores = np.zeros(n_c, dtype=np.float64)

    # 2. content_cosine_score
    if user_profile is not None:
        cand_embs = movie_embeddings[cand].astype(np.float64)
        cnt_scores = cand_embs @ user_profile.astype(np.float64)
    else:
        cnt_scores = np.zeros(n_c, dtype=np.float64)

    # 3. log1p_train_count
    log1p_cnt = meta.log1p_train_counts[cand]

    # 4. mean_train_rating
    mean_rat = meta.mean_train_ratings[cand]

    # 5. genre_overlap_count
    overlap = np.array([
        len(user_liked_genres & meta.movie_genres[c]) for c in cand
    ], dtype=np.float64)

    # 6. is_item_cold
    cold = meta.is_item_cold[cand]

    return np.column_stack([als_scores, cnt_scores, log1p_cnt, mean_rat, overlap, cold])


def build_validation_training_data(
    val_users: list[int],
    val_liked: dict[int, set[int]],
    user_train_rated: dict[int, set[int]],
    user_val_rated: dict[int, set[int]],
    user_train_liked: dict[int, set[int]],
    user_to_idx: dict[int, int],
    als_user_factors: np.ndarray,
    als_item_factors: np.ndarray,
    movie_embeddings: np.ndarray,
    meta: PrecomputedMetadata,
    n_negatives_per_user: int = 20,
    seed: int = EVAL_SEED,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Construct (X, y, groups) dataset from validation users.
    Negatives are sampled with popularity weighting from movies unrated by each user.
    """
    rng = np.random.default_rng(seed)
    n_movies = meta.n_movies

    # Sampling weights: popularity + 1
    weights = (meta.train_counts + 1.0).astype(np.float64)
    probs = weights / weights.sum()

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    groups: list[int] = []

    for uid in val_users:
        pos_cols = list(val_liked.get(uid, set()))
        if not pos_cols:
            continue

        rated_all = user_train_rated.get(uid, set()) | user_val_rated.get(uid, set())

        # Sample ~20 unrated negatives weighted by popularity
        negs: list[int] = []
        while len(negs) < n_negatives_per_user:
            draw = rng.choice(n_movies, size=n_negatives_per_user * 2, p=probs, replace=True)
            for c in draw:
                if c not in rated_all and c not in negs:
                    negs.append(int(c))
                    if len(negs) == n_negatives_per_user:
                        break

        all_candidates = pos_cols + negs
        n_pos = len(pos_cols)
        n_neg = len(negs)
        y_user = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=np.float32)

        # User factors and profiles
        u_idx = user_to_idx[uid]
        u_als_vec = als_user_factors[u_idx]
        liked_train = user_train_liked.get(uid, set())
        u_profile = build_user_profile(liked_train, movie_embeddings)
        u_genres = build_user_liked_genres(liked_train, meta.movie_genres)

        X_user = extract_pair_features(
            all_candidates,
            u_als_vec,
            u_profile,
            u_genres,
            als_item_factors,
            movie_embeddings,
            meta,
        )

        X_list.append(X_user)
        y_list.append(y_user)
        groups.append(n_pos + n_neg)

    X = np.vstack(X_list) if X_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=np.float32)
    return X, y, groups


def train_reranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: list[int],
    random_state: int = EVAL_SEED,
) -> lgb.LGBMRanker:
    """Train LGBMRanker with early stopping disabled (n_estimators=100)."""
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
    )
    ranker.fit(
        X_train,
        y_train,
        group=train_groups,
    )
    return ranker


def save_reranker(model: lgb.LGBMRanker, path: Path = RERANKER_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=5)
    print(f"Saved trained LGBMRanker to {path}")


def load_reranker(path: Path = RERANKER_MODEL_PATH) -> lgb.LGBMRanker:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    print("=" * 60)
    print("TRAINING LEARNING-TO-RANK (LTR) RERANKER")
    print("=" * 60)

    # 1. Load data & artifacts
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    movies_df = pd.read_parquet(MOVIES_CLEAN_PARQUET)
    movie_ids = np.load(MOVIE_IDS_PATH)
    movie_embeddings = np.load(MOVIE_EMBEDDINGS_PATH).astype(np.float32)
    movie_to_idx = {int(m): i for i, m in enumerate(movie_ids)}
    users = sorted(ratings["userId"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}

    # 2. 70/10/20 split
    rng = np.random.default_rng(EVAL_SEED)
    train_rows, val_rows = [], []
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

    train = pd.concat(train_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True)
    print(f"Train ratings: {len(train):,} | Validation ratings: {len(val):,}")

    meta = PrecomputedMetadata(train, movie_ids, movie_to_idx, movies_df)

    # 3. Load ALS artifacts
    als_model, als_user_factors, als_item_factors = load_als_artifacts()
    print("Loaded ALS model and factor matrices.")

    # 4. Build per-user train/val histories
    user_train_rated: dict[int, set[int]] = {}
    user_train_liked: dict[int, set[int]] = {}
    for uid, grp in train.groupby("userId"):
        cols = {movie_to_idx[m] for m in grp["movieId_tmdb"] if m in movie_to_idx}
        user_train_rated[uid] = cols
        liked = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        user_train_liked[uid] = liked

    user_val_rated: dict[int, set[int]] = {}
    val_liked: dict[int, set[int]] = {}
    for uid, grp in val.groupby("userId"):
        cols = {movie_to_idx[m] for m in grp["movieId_tmdb"] if m in movie_to_idx}
        user_val_rated[uid] = cols
        liked = {
            movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in movie_to_idx
        }
        if liked:
            val_liked[uid] = liked

    candidate_val_users = sorted(val_liked.keys())
    print(f"Validation users with liked items: {len(candidate_val_users):,}")

    # 5. Split validation users into train-ranker (85%) and val-ranker (15%) for early stopping
    split_rng = np.random.default_rng(EVAL_SEED + 10)
    shuffled_val_users = candidate_val_users.copy()
    split_rng.shuffle(shuffled_val_users)

    print(f"Building training examples ({len(candidate_val_users):,} validation users, ~20 negatives/user) …")
    X_train, y_train, train_groups = build_validation_training_data(
        candidate_val_users,
        val_liked,
        user_train_rated,
        user_val_rated,
        user_train_liked,
        user_to_idx,
        als_user_factors,
        als_item_factors,
        movie_embeddings,
        meta,
        n_negatives_per_user=20,
        seed=EVAL_SEED + 20,
    )

    print(f"  X_train: {X_train.shape} in {len(train_groups):,} groups")

    # 6. Fit LGBMRanker (n_estimators=100, no early stopping)
    print("\nFitting LGBMRanker (objective=lambdarank, n_estimators=100) …")
    ranker = train_reranker(X_train, y_train, train_groups)

    # 7. Report feature importances
    print("\n" + "=" * 45)
    print("FEATURE IMPORTANCES (LGBMRanker)")
    print("=" * 45)
    importances = ranker.feature_importances_
    total_imp = max(1, sum(importances))
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        pct = (imp / total_imp) * 100
        print(f"  {name:<28}: {imp:>5} ({pct:>5.1f}%)")
    print("=" * 45)

    # 8. Save model
    save_reranker(ranker)
    print("Reranker training complete ✓")


if __name__ == "__main__":
    main()
