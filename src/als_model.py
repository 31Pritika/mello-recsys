"""
Implicit Alternating Least Squares (ALS) Collaborative Filtering Recommender.

Features:
- Builds a binary implicit feedback matrix: 1 where rating >= LIKE_THRESHOLD (4.0) in TRAIN,
  unobserved (0/unset) otherwise (negative/low ratings are left unobserved).
- Fits implicit.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15).
- Confidence weighting alpha=40 (configurable via config.py or CLI).
- Saves model, user factors, and item factors to model_artifacts/.
- Provides utilities for user score prediction and factor recalculation for cold-start users.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config import (
    ALS_ALPHA,
    ALS_FACTORS,
    ALS_ITEM_FACTORS_PATH,
    ALS_ITERATIONS,
    ALS_MODEL_PATH,
    ALS_REGULARIZATION,
    ALS_SEED,
    ALS_USER_FACTORS_PATH,
    EVAL_SEED,
    LIKE_THRESHOLD,
    MOVIE_IDS_PATH,
    RATINGS_CLEAN_PARQUET,
)


def build_implicit_matrix(
    train_df: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
    n_users: int,
    n_movies: int,
    like_threshold: float = LIKE_THRESHOLD,
) -> csr_matrix:
    """
    Build binary implicit matrix (users x items):
    1.0 where rating >= like_threshold in train_df.
    Unobserved entries are left as 0.
    """
    positives = train_df[train_df["rating"] >= like_threshold]
    r_user = positives["userId"].map(user_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    r_movie = positives["movieId_tmdb"].map(movie_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    valid = (r_user >= 0) & (r_movie >= 0)

    data = np.ones(int(valid.sum()), dtype=np.float32)
    matrix = csr_matrix(
        (data, (r_user[valid], r_movie[valid])),
        shape=(n_users, n_movies),
        dtype=np.float32,
    )
    return matrix


def train_als(
    train_matrix: csr_matrix,
    factors: int = ALS_FACTORS,
    regularization: float = ALS_REGULARIZATION,
    iterations: int = ALS_ITERATIONS,
    alpha: float = ALS_ALPHA,
    random_state: int = ALS_SEED,
    show_progress: bool = True,
) -> implicit.cpu.als.AlternatingLeastSquares:
    """Train implicit ALS model on binary train matrix."""
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        random_state=random_state,
    )
    model.fit(train_matrix, show_progress=show_progress)
    return model


def recalculate_user_factors(
    model: implicit.cpu.als.AlternatingLeastSquares,
    liked_movie_col_indices: list[int] | set[int],
    n_movies: int,
) -> np.ndarray:
    """
    Calculate ALS user factor vector (shape: (factors,)) for a user from their
    liked items, without updating the full model. Useful for cold-start users.
    """
    liked_cols = [c for c in liked_movie_col_indices if 0 <= c < n_movies]
    if not liked_cols:
        return np.zeros(model.factors, dtype=np.float32)

    row = csr_matrix(
        (np.ones(len(liked_cols), dtype=np.float32),
         (np.zeros(len(liked_cols), dtype=np.int32), np.array(liked_cols, dtype=np.int32))),
        shape=(1, n_movies),
        dtype=np.float32,
    )
    return model.recalculate_user(0, row).astype(np.float32)


def predict_als_scores(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    rated_col_indices: Optional[set[int]] = None,
) -> np.ndarray:
    """Compute dot product user_factors @ item_factors.T and mask already-rated movies."""
    scores = (user_factors @ item_factors.T).astype(np.float64)
    if rated_col_indices:
        scores[list(rated_col_indices)] = -np.inf
    return scores


def save_als_artifacts(
    model: implicit.cpu.als.AlternatingLeastSquares,
    model_path: Path = ALS_MODEL_PATH,
    user_factors_path: Path = ALS_USER_FACTORS_PATH,
    item_factors_path: Path = ALS_ITEM_FACTORS_PATH,
) -> None:
    """Save model pickle and factor arrays to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=5)
    np.save(user_factors_path, model.user_factors.astype(np.float32))
    np.save(item_factors_path, model.item_factors.astype(np.float32))
    print(f"Saved ALS model to {model_path}")
    print(f"Saved user factors to {user_factors_path} (shape={model.user_factors.shape})")
    print(f"Saved item factors to {item_factors_path} (shape={model.item_factors.shape})")


def load_als_artifacts(
    model_path: Path = ALS_MODEL_PATH,
    user_factors_path: Path = ALS_USER_FACTORS_PATH,
    item_factors_path: Path = ALS_ITEM_FACTORS_PATH,
) -> tuple[implicit.cpu.als.AlternatingLeastSquares, np.ndarray, np.ndarray]:
    """Load model pickle and factor arrays from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    user_factors = np.load(user_factors_path).astype(np.float32)
    item_factors = np.load(item_factors_path).astype(np.float32)
    return model, user_factors, item_factors


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save implicit ALS recommender model.")
    parser.add_argument("--alpha", type=float, default=ALS_ALPHA, help=f"Confidence weighting alpha (default: {ALS_ALPHA})")
    parser.add_argument("--factors", type=int, default=ALS_FACTORS, help=f"Latent factors (default: {ALS_FACTORS})")
    parser.add_argument("--reg", type=float, default=ALS_REGULARIZATION, help=f"Regularization (default: {ALS_REGULARIZATION})")
    parser.add_argument("--iterations", type=int, default=ALS_ITERATIONS, help=f"Iterations (default: {ALS_ITERATIONS})")
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING IMPLICIT ALS RECOMMENDER")
    print(f"  factors={args.factors}, reg={args.reg}, iter={args.iterations}, alpha={args.alpha}")
    print("=" * 60)

    # 1. Load data
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    movie_ids = np.load(MOVIE_IDS_PATH)
    n_movies = len(movie_ids)
    movie_to_idx = {int(m): i for i, m in enumerate(movie_ids)}
    users = sorted(ratings["userId"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    # 2. 70/10/20 train split
    rng = np.random.default_rng(EVAL_SEED)
    train_rows = []
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

    train = pd.concat(train_rows, ignore_index=True)
    print(f"Train split ratings: {len(train):,}")

    # 3. Build implicit matrix
    mat = build_implicit_matrix(train, user_to_idx, movie_to_idx, n_users, n_movies, like_threshold=LIKE_THRESHOLD)
    print(f"Binary implicit matrix: {mat.shape[0]:,} users x {mat.shape[1]:,} items with {mat.nnz:,} positive ratings (>= {LIKE_THRESHOLD})")

    # 4. Fit ALS
    print("Fitting model …")
    model = train_als(
        mat,
        factors=args.factors,
        regularization=args.reg,
        iterations=args.iterations,
        alpha=args.alpha,
        random_state=ALS_SEED,
        show_progress=True,
    )

    # 5. Save artifacts
    save_als_artifacts(model)
    print("ALS training complete ✓")


if __name__ == "__main__":
    main()
