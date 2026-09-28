"""
FastAPI recommendation service.

Loads all artifacts ONCE at startup. Never retrains.

Endpoints:
  GET /                                                → status + counts
  GET /recommendations/{user_id}?k=10&method=svd|content|hybrid (default svd)
  GET /users/{user_id}/compatibility/{other}          → cosine similarity
  GET /users/{user_id}/cluster                        → cluster_id + cluster_size
  GET /movies/{tmdb_id}/similar?k=10                  → item-to-item similar movies
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config import (
    CLUSTER_LABELS_PATH,
    HYBRID_ALPHA_PATH,
    IDX_TO_USER_PATH,
    KMEANS_K,
    LIKE_THRESHOLD,
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    MOVIE_TO_IDX_PATH,
    MOVIES_CLEAN_PARQUET,
    RATINGS_CLEAN_PARQUET,
    SVD_PATH,
    USER_FACTORS_PATH,
    USER_TO_IDX_PATH,
)
from content_model import (
    build_user_profile,
    compute_content_scores,
    get_item_similarities,
    recommend_content,
)
from hybrid import compute_hybrid_scores, load_alpha, recommend_hybrid


# ── Global state ───────────────────────────────────────────────────────────────

class AppState:
    svd: Any
    user_factors: np.ndarray          # (n_users, n_components)
    cluster_labels: np.ndarray        # (n_users,)
    user_to_idx: dict[int, int]
    idx_to_user: dict[int, int]
    movie_to_idx: dict[int, int]
    movie_ids: np.ndarray             # (n_movies,) TMDB ids
    id_to_title: dict[int, str]
    id_to_genres: dict[int, str]
    rated_by_user: dict[int, set[int]]  # userId → set of movieId_tmdb column indices
    liked_by_user: dict[int, set[int]]  # userId → set of movieId_tmdb col indices (rating >= 4.0)
    n_users: int
    n_movies: int
    cluster_sizes: dict[int, int]
    movie_embeddings: Optional[np.ndarray] = None
    hybrid_alpha: float = 0.5


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts at startup."""
    print("Loading artifacts …")
    with open(SVD_PATH, "rb") as f:
        state.svd = pickle.load(f)
    state.user_factors = np.load(USER_FACTORS_PATH)
    state.cluster_labels = np.load(CLUSTER_LABELS_PATH)
    with open(USER_TO_IDX_PATH, "rb") as f:
        state.user_to_idx = pickle.load(f)
    with open(IDX_TO_USER_PATH, "rb") as f:
        state.idx_to_user = pickle.load(f)
    with open(MOVIE_TO_IDX_PATH, "rb") as f:
        state.movie_to_idx = pickle.load(f)
    state.movie_ids = np.load(MOVIE_IDS_PATH)
    state.n_users, state.n_movies = state.user_factors.shape[0], len(state.movie_ids)

    movies = pd.read_parquet(MOVIES_CLEAN_PARQUET)
    state.id_to_title  = dict(zip(movies["id"].astype(int), movies["title"]))
    state.id_to_genres = dict(zip(movies["id"].astype(int), movies["genres"].fillna("")))

    # Build per-user rated and liked column-index sets
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    state.rated_by_user = {}
    state.liked_by_user = {}
    for uid, grp in ratings.groupby("userId"):
        uid_int = int(uid)
        cols = set(
            state.movie_to_idx[m]
            for m in grp["movieId_tmdb"]
            if m in state.movie_to_idx
        )
        state.rated_by_user[uid_int] = cols

        liked_cols = set(
            state.movie_to_idx[m]
            for m in grp.loc[grp["rating"] >= LIKE_THRESHOLD, "movieId_tmdb"]
            if m in state.movie_to_idx
        )
        state.liked_by_user[uid_int] = liked_cols

    state.cluster_sizes = {
        int(cid): int((state.cluster_labels == cid).sum())
        for cid in range(KMEANS_K)
    }

    # Phase 2: Load neural movie embeddings if present
    if MOVIE_EMBEDDINGS_PATH.exists():
        state.movie_embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
        print(f"  Loaded movie embeddings: {state.movie_embeddings.shape}")
    else:
        state.movie_embeddings = None
        print("  Notice: movie_embeddings.npy not found. Neural content endpoints will require running embed_movies.py.")

    # Phase 2: Load tuned hybrid alpha
    state.hybrid_alpha = load_alpha(HYBRID_ALPHA_PATH, default=0.5)
    print(f"  Loaded hybrid alpha: {state.hybrid_alpha:.2f}")

    print(f"  {state.n_users:,} users | {state.n_movies:,} movies — ready ✓")
    yield
    # (no cleanup needed)


app = FastAPI(title="Mello RecSys", lifespan=lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_user_idx(user_id: int) -> int:
    try:
        return state.user_to_idx[user_id]
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found. Known user ids are integers in the training set.",
        )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _recommend_svd(user_id: int, k: int) -> list[tuple[int, float]]:
    """Return top-k (movieId_tmdb, score) via SVD user factors, masking rated movies."""
    u_idx = _get_user_idx(user_id)
    scores = (state.user_factors[u_idx] @ state.svd.components_).copy()  # (n_movies,)
    rated_cols = state.rated_by_user.get(user_id, set())
    if rated_cols:
        scores[list(rated_cols)] = -np.inf
    k = min(k, len(scores))
    top_cols = np.argpartition(scores, -k)[-k:]
    top_cols = top_cols[np.argsort(scores[top_cols])[::-1]]
    return [(int(state.movie_ids[c]), float(scores[c])) for c in top_cols if np.isfinite(scores[c])]


def _recommend_content(user_id: int, k: int) -> list[tuple[int, float]]:
    """Return top-k recommendations via neural content profile."""
    _get_user_idx(user_id)
    if state.movie_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Movie embeddings not loaded. Run embed_movies.py first.",
        )
    liked_cols = state.liked_by_user.get(user_id, set())
    if not liked_cols:
        # Fallback to SVD if user has no liked movies
        return _recommend_svd(user_id, k)

    profile = build_user_profile(liked_cols, state.movie_embeddings)
    rated_cols = state.rated_by_user.get(user_id, set())
    return recommend_content(profile, state.movie_embeddings, state.movie_ids, rated_cols, top_k=k)


def _recommend_hybrid(user_id: int, k: int) -> list[tuple[int, float]]:
    """Return top-k recommendations via standardized hybrid model."""
    u_idx = _get_user_idx(user_id)
    if state.movie_embeddings is None:
        return _recommend_svd(user_id, k)

    rated_cols = state.rated_by_user.get(user_id, set())
    liked_cols = state.liked_by_user.get(user_id, set())

    # SVD score vector
    svd_scores = (state.user_factors[u_idx] @ state.svd.components_).copy()
    if rated_cols:
        svd_scores[list(rated_cols)] = -np.inf

    # Content score vector
    profile = build_user_profile(liked_cols, state.movie_embeddings)
    content_scores = compute_content_scores(profile, state.movie_embeddings, rated_cols)

    return recommend_hybrid(svd_scores, content_scores, state.movie_ids, alpha=state.hybrid_alpha, top_k=k)


# ── Response models ────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    n_users: int
    n_movies: int
    embeddings_loaded: bool = False
    hybrid_alpha: float = 0.5


class MovieRec(BaseModel):
    movieId_tmdb: int
    title: str
    genres: str
    score: float


class RecommendationsResponse(BaseModel):
    user_id: int
    method: str
    recommendations: list[MovieRec]


class CompatibilityResponse(BaseModel):
    user_id: int
    other_user_id: int
    cosine_similarity: float


class ClusterResponse(BaseModel):
    user_id: int
    cluster_id: int
    cluster_size: int


class SimilarMovieRec(BaseModel):
    movieId_tmdb: int
    title: str
    genres: str
    score: float


class SimilarMoviesResponse(BaseModel):
    movieId_tmdb: int
    similar_movies: list[SimilarMovieRec]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=StatusResponse)
def root() -> StatusResponse:
    return StatusResponse(
        status="ok",
        n_users=state.n_users,
        n_movies=state.n_movies,
        embeddings_loaded=(state.movie_embeddings is not None),
        hybrid_alpha=state.hybrid_alpha,
    )


@app.get("/recommendations/{user_id}", response_model=RecommendationsResponse)
def recommend(
    user_id: int,
    k: int = 10,
    method: str = Query("svd", description="Recommendation method: svd, content, or hybrid"),
) -> RecommendationsResponse:
    k = max(1, min(k, 50))
    method_lower = method.lower().strip()

    if method_lower == "svd":
        recs = _recommend_svd(user_id, k)
    elif method_lower == "content":
        recs = _recommend_content(user_id, k)
    elif method_lower == "hybrid":
        recs = _recommend_hybrid(user_id, k)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown recommendation method '{method}'. Valid options are: svd, content, hybrid.",
        )

    items = [
        MovieRec(
            movieId_tmdb=mid,
            title=state.id_to_title.get(mid, "Unknown"),
            genres=state.id_to_genres.get(mid, ""),
            score=round(score, 6),
        )
        for mid, score in recs
    ]
    return RecommendationsResponse(user_id=user_id, method=method_lower, recommendations=items)


@app.get("/users/{user_id}/compatibility/{other_user_id}", response_model=CompatibilityResponse)
def compatibility(user_id: int, other_user_id: int) -> CompatibilityResponse:
    u_idx = _get_user_idx(user_id)
    v_idx = _get_user_idx(other_user_id)
    sim = _cosine(state.user_factors[u_idx], state.user_factors[v_idx])
    return CompatibilityResponse(
        user_id=user_id,
        other_user_id=other_user_id,
        cosine_similarity=round(sim, 6),
    )


@app.get("/users/{user_id}/cluster", response_model=ClusterResponse)
def cluster(user_id: int) -> ClusterResponse:
    u_idx = _get_user_idx(user_id)
    cid = int(state.cluster_labels[u_idx])
    return ClusterResponse(
        user_id=user_id,
        cluster_id=cid,
        cluster_size=state.cluster_sizes[cid],
    )


@app.get("/movies/{tmdb_id}/similar", response_model=SimilarMoviesResponse)
def similar_movies(tmdb_id: int, k: int = 10) -> SimilarMoviesResponse:
    k = max(1, min(k, 50))
    if state.movie_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Movie embeddings not loaded. Run embed_movies.py first.",
        )
    if tmdb_id not in state.movie_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Movie {tmdb_id} not found in catalog. Must be a valid TMDB ID from movies_clean.",
        )

    sims = get_item_similarities(tmdb_id, state.movie_to_idx, state.movie_ids, state.movie_embeddings, top_k=k)
    items = [
        SimilarMovieRec(
            movieId_tmdb=mid,
            title=state.id_to_title.get(mid, "Unknown"),
            genres=state.id_to_genres.get(mid, ""),
            score=round(score, 6),
        )
        for mid, score in sims
    ]
    return SimilarMoviesResponse(movieId_tmdb=tmdb_id, similar_movies=items)
