"""
Hybrid recommendation model combining SVD and Content embeddings.

Formula:
  hybrid_score = alpha * z(svd_score) + (1 - alpha) * z(content_score)
where z() standardizes unmasked scores (mean 0, std 1) for that user.

Contains logic to:
- Standardize score vectors safely with z-score.
- Combine SVD and Content scores.
- Tune alpha over [0, 0.1, ..., 1.0] on a validation split.
"""
from __future__ import annotations

import pickle
from typing import Optional

import numpy as np

from config import HYBRID_ALPHA_PATH


def z_score(scores: np.ndarray) -> np.ndarray:
    """Standardize finite scores: (x - mean) / std.

    Keeps masked elements (-np.inf) intact.
    If std == 0, returns zeros for valid entries.
    """
    valid = np.isfinite(scores)
    if not np.any(valid):
        return scores.copy()

    vals = scores[valid]
    mean = np.mean(vals)
    std = np.std(vals)
    if std < 1e-9:
        std = 1.0

    result = np.full_like(scores, -np.inf)
    result[valid] = (vals - mean) / std
    return result


def compute_hybrid_scores(
    svd_scores: Optional[np.ndarray],
    content_scores: Optional[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """Combine SVD and Content scores using standardized weighting.

    hybrid = alpha * z(svd) + (1 - alpha) * z(content)
    If content is unavailable, falls back to z(svd).
    If svd is unavailable, falls back to z(content).
    """
    if svd_scores is None and content_scores is None:
        raise ValueError("At least one of svd_scores or content_scores must be provided")

    if content_scores is None:
        return z_score(svd_scores)

    if svd_scores is None:
        return z_score(content_scores)

    z_svd = z_score(svd_scores)
    z_cnt = z_score(content_scores)

    # Valid items are where either score is finite
    valid = np.isfinite(z_svd) & np.isfinite(z_cnt)
    hybrid = np.full_like(z_svd, -np.inf)
    hybrid[valid] = alpha * z_svd[valid] + (1.0 - alpha) * z_cnt[valid]

    # In case one was finite and other wasn't
    only_svd = np.isfinite(z_svd) & (~np.isfinite(z_cnt))
    hybrid[only_svd] = z_svd[only_svd]

    only_cnt = (~np.isfinite(z_svd)) & np.isfinite(z_cnt)
    hybrid[only_cnt] = z_cnt[only_cnt]

    return hybrid


def recommend_hybrid(
    svd_scores: Optional[np.ndarray],
    content_scores: Optional[np.ndarray],
    movie_ids: np.ndarray,
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Return top-k recommendations using hybrid model."""
    scores = compute_hybrid_scores(svd_scores, content_scores, alpha)
    top_k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        (int(movie_ids[i]), float(scores[i]))
        for i in top_indices
        if np.isfinite(scores[i])
    ]


def save_alpha(alpha: float, path=HYBRID_ALPHA_PATH) -> None:
    """Save tuned alpha to model_artifacts."""
    with open(path, "wb") as f:
        pickle.dump(float(alpha), f, protocol=5)
    print(f"Saved tuned alpha={alpha:.2f} to {path}")


def load_alpha(path=HYBRID_ALPHA_PATH, default: float = 0.5) -> float:
    """Load tuned alpha from model_artifacts, or return default."""
    try:
        with open(path, "rb") as f:
            return float(pickle.load(f))
    except Exception:
        return default
