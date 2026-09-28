"""
FastAPI recommendation service.

Loads all artifacts ONCE at startup. Never retrains.

Endpoints:
  GET /                                       → status + counts
  GET /recommendations/{user_id}?k=10        → top-k movies
  GET /users/{user_id}/compatibility/{other} → cosine similarity
  GET /users/{user_id}/cluster               → cluster_id + cluster_size
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    CLUSTER_LABELS_PATH,
    IDX_TO_USER_PATH,
    KMEANS_K,
    MOVIE_IDS_PATH,
    MOVIE_TO_IDX_PATH,
    MOVIES_CLEAN_PARQUET,
    RATINGS_CLEAN_PARQUET,
    SVD_PATH,
    USER_FACTORS_PATH,
    USER_TO_IDX_PATH,
)


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
    n_users: int
    n_movies: int
    cluster_sizes: dict[int, int]


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

    # Build per-user rated column-index sets (for masking)
    ratings = pd.read_parquet(RATINGS_CLEAN_PARQUET)
    state.rated_by_user = {}
    for uid, grp in ratings.groupby("userId"):
        cols = set(
            state.movie_to_idx[m]
            for m in grp["movieId_tmdb"]
            if m in state.movie_to_idx
        )
        state.rated_by_user[int(uid)] = cols

    state.cluster_sizes = {
        int(cid): int((state.cluster_labels == cid).sum())
        for cid in range(KMEANS_K)
    }

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
            detail=f"User {user_id} not found. "
                   f"Known user ids are integers in the training set.",
        )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _recommend(user_id: int, k: int) -> list[tuple[int, float]]:
    """Return top-k (movieId_tmdb, score) excluding already-rated."""
    u_idx = _get_user_idx(user_id)
    scores = state.user_factors[u_idx] @ state.svd.components_  # (n_movies,)
    rated_cols = state.rated_by_user.get(user_id, set())
    if rated_cols:
        scores[list(rated_cols)] = -np.inf
    top_cols = np.argpartition(scores, -k)[-k:]
    top_cols = top_cols[np.argsort(scores[top_cols])[::-1]]
    result = [(int(state.movie_ids[c]), float(scores[c])) for c in top_cols]
    return result


# ── Response models ────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    n_users: int
    n_movies: int


class MovieRec(BaseModel):
    movieId_tmdb: int
    title: str
    genres: str
    score: float


class RecommendationsResponse(BaseModel):
    user_id: int
    recommendations: list[MovieRec]


class CompatibilityResponse(BaseModel):
    user_id: int
    other_user_id: int
    cosine_similarity: float


class ClusterResponse(BaseModel):
    user_id: int
    cluster_id: int
    cluster_size: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=StatusResponse)
def root() -> StatusResponse:
    return StatusResponse(
        status="ok",
        n_users=state.n_users,
        n_movies=state.n_movies,
    )


@app.get("/recommendations/{user_id}", response_model=RecommendationsResponse)
def recommend(user_id: int, k: int = 10) -> RecommendationsResponse:
    k = max(1, min(k, 50))
    recs = _recommend(user_id, k)
    items = [
        MovieRec(
            movieId_tmdb=mid,
            title=state.id_to_title.get(mid, "Unknown"),
            genres=state.id_to_genres.get(mid, ""),
            score=score,
        )
        for mid, score in recs
    ]
    return RecommendationsResponse(user_id=user_id, recommendations=items)


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
