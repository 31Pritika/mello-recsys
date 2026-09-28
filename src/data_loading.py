"""
Data loading utilities.

Pass 1: stream ratings.csv, collect unique userId column only → sample N_USERS.
Pass 2: stream chunks again, keep only rows for sampled users.
Also loads + cleans movies_metadata.csv and links.csv.
"""
from __future__ import annotations

import gc
import random
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config import (
    CHUNK_SIZE,
    LINKS_CSV,
    MIN_RATINGS,
    MOVIES_CSV,
    N_USERS,
    RATINGS_CSV,
    SAMPLE_SEED,
)


# ── Movies ─────────────────────────────────────────────────────────────────────

def load_movies_clean() -> pd.DataFrame:
    """Load and clean movies_metadata.csv.

    Keeps only numeric ids, drops duplicate ids, returns relevant columns.
    Parses genres list to a comma-joined string.
    """
    print("Loading movies_metadata.csv …")
    movies = pd.read_csv(
        MOVIES_CSV,
        usecols=["id", "title", "genres", "overview", "vote_average", "vote_count"],
        low_memory=False,
    )
    # Keep only rows where id is numeric
    movies = movies[pd.to_numeric(movies["id"], errors="coerce").notna()].copy()
    movies["id"] = movies["id"].astype(int)
    # Drop duplicate ids (keep first occurrence)
    movies = movies.drop_duplicates(subset="id", keep="first")
    # Parse genres from list-of-dicts string → comma-separated names
    movies["genres"] = movies["genres"].apply(_parse_genres)
    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0.0).astype("float32")
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0).astype("int32")
    movies = movies.reset_index(drop=True)
    print(f"  → {len(movies):,} movies after cleaning")
    return movies


def _parse_genres(val: object) -> str:
    """Convert a stringified list-of-dicts to comma-joined genre names."""
    if not isinstance(val, str) or val.strip() in ("", "[]"):
        return ""
    try:
        import ast
        items = ast.literal_eval(val)
        return ", ".join(item["name"] for item in items if "name" in item)
    except Exception:
        return ""


# ── Links ──────────────────────────────────────────────────────────────────────

def load_links() -> pd.DataFrame:
    """Load links.csv; drop rows without tmdbId; cast to int."""
    print("Loading links.csv …")
    links = pd.read_csv(LINKS_CSV, usecols=["movieId", "tmdbId"])
    links = links.dropna(subset=["tmdbId"]).copy()
    links["movieId"] = links["movieId"].astype("int32")
    links["tmdbId"] = links["tmdbId"].astype(int)
    links = links.rename(columns={"tmdbId": "movieId_tmdb"})
    print(f"  → {len(links):,} links")
    return links


# ── Ratings ────────────────────────────────────────────────────────────────────

def sample_user_ids(n: int = N_USERS, seed: int = SAMPLE_SEED) -> set[int]:
    """Pass 1: stream ratings.csv reading only userId column; pick n random users."""
    print(f"Pass 1: collecting unique userIds from {RATINGS_CSV} …")
    all_ids: set[int] = set()
    for chunk in pd.read_csv(
        RATINGS_CSV,
        usecols=["userId"],
        dtype={"userId": "int32"},
        chunksize=CHUNK_SIZE,
    ):
        all_ids.update(chunk["userId"].tolist())

    all_ids_list = sorted(all_ids)
    print(f"  → {len(all_ids_list):,} unique users found")
    rng = random.Random(seed)
    sampled = set(rng.sample(all_ids_list, min(n, len(all_ids_list))))
    print(f"  → sampled {len(sampled):,} users")
    return sampled


def load_ratings_for_users(
    user_ids: set[int],
    valid_tmdb_ids: Optional[set[int]] = None,
) -> pd.DataFrame:
    """Pass 2: stream chunks; keep rows for sampled users.

    Optionally filter to only rows whose movieId_tmdb is in valid_tmdb_ids
    (done after the links join in the caller).
    """
    print("Pass 2: streaming ratings for sampled users …")
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(
        RATINGS_CSV,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        chunksize=CHUNK_SIZE,
    ):
        filtered = chunk[chunk["userId"].isin(user_ids)]
        if len(filtered):
            chunks.append(filtered)
        total_rows += len(chunk)

    print(f"  → scanned {total_rows:,} rows")
    df = pd.concat(chunks, ignore_index=True)
    print(f"  → {len(df):,} ratings for sampled users")
    return df


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_clean_dataset(
    n_users: int = N_USERS,
    seed: int = SAMPLE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ratings_clean, movies_clean) after all cleaning steps."""
    movies = load_movies_clean()
    links = load_links()

    # Bridge: movieId (ML) → movieId_tmdb (TMDB)
    valid_movie_ids = set(links["movieId"].tolist())
    valid_tmdb_ids = set(movies["id"].tolist())

    # Pass 1: sample users
    user_ids = sample_user_ids(n=n_users, seed=seed)

    # Pass 2: stream ratings for those users
    ratings = load_ratings_for_users(user_ids)

    # Join links to get tmdb id
    print("Joining links …")
    ratings = ratings.merge(links, on="movieId", how="inner")
    # Rename for clarity
    ratings = ratings.rename(columns={"movieId_tmdb": "movieId_tmdb"})

    # Drop ratings whose tmdb movie has no entry in movies_clean
    print("Filtering to known movies …")
    before = len(ratings)
    ratings = ratings[ratings["movieId_tmdb"].isin(valid_tmdb_ids)].copy()
    print(f"  → dropped {before - len(ratings):,} ratings for unknown movies")

    # Drop users with < MIN_RATINGS ratings
    print(f"Dropping users with < {MIN_RATINGS} ratings …")
    counts = ratings["userId"].value_counts()
    good_users = counts[counts >= MIN_RATINGS].index
    before = len(ratings)
    ratings = ratings[ratings["userId"].isin(good_users)].copy()
    print(f"  → dropped {before - len(ratings):,} ratings (sparse users); {ratings['userId'].nunique():,} users remain")

    # Keep only columns we need
    ratings = ratings[["userId", "movieId_tmdb", "rating"]].copy()
    ratings = ratings.reset_index(drop=True)

    # Align movies to those actually present in ratings
    present_tmdb = set(ratings["movieId_tmdb"].unique())
    movies_clean = movies[movies["id"].isin(present_tmdb)].copy().reset_index(drop=True)

    print(f"\nFinal dataset: {len(ratings):,} ratings | {ratings['userId'].nunique():,} users | {ratings['movieId_tmdb'].nunique():,} movies")
    avg = len(ratings) / ratings["userId"].nunique()
    print(f"Average ratings per user: {avg:.1f}")

    return ratings, movies_clean


# ── Sparse matrix ──────────────────────────────────────────────────────────────

def build_sparse_matrix(
    ratings: pd.DataFrame,
) -> tuple[csr_matrix, dict, dict, dict, np.ndarray]:
    """Build a users × movies CSR matrix from ratings DataFrame.

    Returns:
        matrix: (n_users, n_movies) csr_matrix of ratings
        user_to_idx: userId → row index
        idx_to_user: row index → userId
        movie_to_idx: movieId_tmdb → col index
        movie_ids: array of movieId_tmdb in column order
    """
    users = sorted(ratings["userId"].unique())
    movies = sorted(ratings["movieId_tmdb"].unique())

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    movie_to_idx = {m: j for j, m in enumerate(movies)}
    movie_ids = np.array(movies, dtype=np.int32)

    row = ratings["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
    col = ratings["movieId_tmdb"].map(movie_to_idx).to_numpy(dtype=np.int32)
    data = ratings["rating"].to_numpy(dtype=np.float32)

    matrix = csr_matrix(
        (data, (row, col)),
        shape=(len(users), len(movies)),
        dtype=np.float32,
    )
    print(f"Sparse matrix shape: {matrix.shape}, nnz={matrix.nnz:,}, density={matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.5%}")
    return matrix, user_to_idx, idx_to_user, movie_to_idx, movie_ids
