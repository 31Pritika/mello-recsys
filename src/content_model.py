"""
Content-based recommender model using neural movie embeddings.

Features:
- User profile = L2-normalized mean embedding of movies rated >= 4.0 in train only.
- Content score = user_profile @ movie_embeddings.T, computed per user on demand.
- Item-to-item similarity = top-k most similar movies by cosine distance.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


def build_user_profile(
    liked_movie_col_indices: list[int] | set[int],
    movie_embeddings: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute L2-normalized mean embedding of liked movies.

    Returns None if user has no liked items.
    """
    if not liked_movie_col_indices:
        return None

    indices = list(liked_movie_col_indices)
    liked_embs = movie_embeddings[indices]
    profile = np.mean(liked_embs, axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return profile.astype(np.float32)


def compute_content_scores(
    user_profile: Optional[np.ndarray],
    movie_embeddings: np.ndarray,
    rated_col_indices: Optional[set[int]] = None,
) -> Optional[np.ndarray]:
    """Compute content scores for all catalog movies on demand.

    Scores are cosine similarities in [-1, 1].
    Already-rated movies are masked with -np.inf.
    Returns None if user_profile is None.
    """
    if user_profile is None:
        return None

    # movie_embeddings is (n_movies, dim), user_profile is (dim,)
    scores = movie_embeddings @ user_profile  # shape (n_movies,)
    scores = scores.copy()

    if rated_col_indices:
        masked_indices = list(rated_col_indices)
        scores[masked_indices] = -np.inf

    return scores


def get_item_similarities(
    movie_id: int,
    movie_to_idx: dict[int, int],
    movie_ids: np.ndarray,
    movie_embeddings: np.ndarray,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Return top-k most similar movies to movie_id by cosine similarity.

    Excludes the queried movie itself.
    """
    if movie_id not in movie_to_idx:
        raise KeyError(f"Movie ID {movie_id} not found in catalog")

    m_idx = movie_to_idx[movie_id]
    query_emb = movie_embeddings[m_idx]

    sims = (movie_embeddings @ query_emb).copy()
    sims[m_idx] = -np.inf  # exclude self

    top_k = min(top_k, len(sims) - 1)
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    return [(int(movie_ids[i]), float(sims[i])) for i in top_indices]


def recommend_content(
    user_profile: Optional[np.ndarray],
    movie_embeddings: np.ndarray,
    movie_ids: np.ndarray,
    rated_col_indices: Optional[set[int]] = None,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Return top-k recommendations using content model."""
    scores = compute_content_scores(user_profile, movie_embeddings, rated_col_indices)
    if scores is None:
        return []

    top_k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        (int(movie_ids[i]), float(scores[i]))
        for i in top_indices
        if np.isfinite(scores[i])
    ]
