"""
Embed movies using SentenceTransformer.

Generates dense text embeddings for all movies in movies_clean.parquet.
Saves movie_embeddings.npy aligned with movie_ids.npy (column order of the user-item matrix).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    ARTIFACTS_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    MOVIES_CLEAN_PARQUET,
)


def build_movie_text(title: object, genres: object, overview: object) -> str:
    """Build text representation: '<title>. Genres: <genres>. <overview>'.

    If overview is missing/NaN, use title + genres only.
    Never drops a movie for missing text.
    """
    t = str(title).strip() if pd.notna(title) and str(title).strip() else "Untitled"
    g = str(genres).strip() if pd.notna(genres) and str(genres).strip() else "Unknown"
    o = str(overview).strip() if pd.notna(overview) and str(overview).strip() else ""

    text = f"{t}. Genres: {g}."
    if o:
        text = f"{text} {o}"
    return text


def main() -> None:
    print("=" * 60)
    print("EMBED MOVIES (Phase 2)")
    print("=" * 60)

    # 1. Load movies and movie_ids
    print("Loading movies and movie_ids …")
    movies_df = pd.read_parquet(MOVIES_CLEAN_PARQUET)
    movie_ids = np.load(MOVIE_IDS_PATH)
    n_movies = len(movie_ids)
    print(f"  Total catalog movies in matrix: {n_movies:,}")
    print(f"  Total records in movies_clean: {len(movies_df):,}")

    # Build mapping from movie_id -> row data
    # Ensure movie_ids are integers
    movies_df["id"] = movies_df["id"].astype(int)
    movie_map = movies_df.set_index("id").to_dict(orient="index")

    # 2. Build aligned texts in the exact order of movie_ids
    texts: list[str] = []
    missing_count = 0
    for mid in movie_ids:
        mid_int = int(mid)
        if mid_int in movie_map:
            row = movie_map[mid_int]
            txt = build_movie_text(row.get("title"), row.get("genres"), row.get("overview"))
        else:
            missing_count += 1
            txt = "Untitled. Genres: Unknown."
        texts.append(txt)

    if missing_count:
        print(f"  Warning: {missing_count} movie IDs had no record in movies_clean metadata.")

    print(f"Sample text for first movie (id={movie_ids[0]}):")
    print(f"  \"{texts[0][:120]}...\"")

    # 3. Load model and encode
    print(f"\nLoading SentenceTransformer model '{EMBEDDING_MODEL_NAME}' on CPU …")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    # Sort texts by length for high batch efficiency on CPU (minimizes padding)
    print("Sorting texts by length for efficient CPU batching …")
    sort_idx = np.argsort([len(t) for t in texts])
    sorted_texts = [texts[i] for i in sort_idx]

    print(f"Encoding {len(texts):,} movies (batch_size={EMBEDDING_BATCH_SIZE}, normalize_embeddings=True) …")
    sorted_embeddings = model.encode(
        sorted_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Invert sorting to restore exact original order of movie_ids
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    embeddings = sorted_embeddings[inv_idx]

    # 4. Assert alignment and shapes
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    assert embeddings.shape[0] == n_movies, (
        f"Mismatch: embeddings shape {embeddings.shape[0]} != movie_ids length {n_movies}"
    )

    # 5. Save artifacts
    print(f"Saving embeddings to {MOVIE_EMBEDDINGS_PATH} …")
    np.save(MOVIE_EMBEDDINGS_PATH, embeddings)

    # Also save verification file of movie_ids order used
    order_path = ARTIFACTS_DIR / "movie_ids_embedding_order.npy"
    np.save(order_path, movie_ids)
    print(f"Saved movie IDs order verification to {order_path}")

    print("Embedding complete ✓")


if __name__ == "__main__":
    main()
