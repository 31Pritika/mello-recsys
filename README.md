# Mello RecSys — Local Movie Recommender

A memory-safe, hybrid (SVD + neural content) movie recommendation service built on the MovieLens + TMDB dataset.
Streams 26M ratings without ever loading them all at once.

## Architecture

```
data/
├── movies_metadata.csv      # TMDB metadata
├── ratings.csv              # 26M MovieLens ratings
└── links.csv                # MovieLens → TMDB id bridge

src/
├── config.py                # All constants & paths
├── data_loading.py          # Two-pass streaming loader + sparse matrix builder
├── train_and_save.py        # Phase 1: clean → SVD → KMeans → artifacts
├── evaluate.py              # Phase 1: precision@10 vs popularity baseline
├── cluster.py               # Elbow analysis + cluster introspection
├── embed_movies.py          # Phase 2: encode movies with sentence-transformers
├── content_model.py         # Phase 2: content scoring (user profile @ embeddings)
├── hybrid.py                # Phase 2: z-score hybrid (alpha * SVD + (1-alpha) * content)
├── evaluate_phase2.py       # Phase 2: full eval table (all 4 models, cold/heavy split)
└── api.py                   # FastAPI service (all artifacts loaded once at startup)

model_artifacts/             # Saved after training
smoke_test.py                # End-to-end API smoke test
```

## RAM Safety Guarantees

| Constraint | Implementation |
|---|---|
| Never load all ratings at once | `pd.read_csv(..., chunksize=500_000)` in two passes |
| Sample by user, not row | Pass 1 collects unique userIds; pass 2 keeps all rows for sampled users |
| No users×users similarity matrix | Cosine computed one user vs all (dot product, not full matrix) |
| No dense predictions matrix | `user_factors[u] @ svd.components_` per request, on demand |
| Sparse user-item matrix | `scipy.sparse.csr_matrix` throughout |
| Embeddings on demand | Content scores computed per user (`profile @ embeddings.T`), never full matrix |

## Quick Start

### Phase 1 — SVD Recommender

```bash
# 1. Activate the venv
source venv/bin/activate

# 2. Train (streams ratings, fits SVD + KMeans, saves artifacts)
python src/train_and_save.py

# 3. Evaluate (precision@10 vs popularity baseline)
python src/evaluate.py

# 4. Cluster analysis (elbow k=2..10 + per-cluster sample)
python src/cluster.py
```

### Phase 2 — Neural Embeddings + Hybrid Recommender

```bash
# 5. Encode all movies with all-MiniLM-L6-v2 (downloads model once, ~30 min on CPU)
python src/embed_movies.py

# 6. Evaluate all 4 models; tunes and saves best alpha (10-15 min on CPU)
python src/evaluate_phase2.py
#    add --full to evaluate on all test users instead of 5,000

# 7. Start API (loads all artifacts once; serves SVD / content / hybrid)
uvicorn src.api:app --host 127.0.0.1 --port 8000

# 8. Smoke test (in another terminal)
python smoke_test.py
```

> **Note**: run steps in order. `train_and_save.py` must complete before everything else;
> `embed_movies.py` must complete before `evaluate_phase2.py`.

## Configuration (`src/config.py`)

| Variable | Default | Description |
|---|---|---|
| `N_USERS` | 15 000 | Users to sample from ratings |
| `SAMPLE_SEED` | 42 | Random seed for user sampling |
| `CHUNK_SIZE` | 500 000 | Rows per streaming chunk |
| `MIN_RATINGS` | 10 | Drop users with fewer ratings |
| `N_COMPONENTS` | 20 | SVD latent factors |
| `KMEANS_K` | 4 | Final number of clusters |
| `EVAL_N_USERS` | 5 000 | Users to evaluate on |
| `TOP_K` | 10 | Recommendations per user |
| `LIKE_THRESHOLD` | 4.0 | Min rating to count as a "like" |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `EMBEDDING_BATCH_SIZE` | 64 | Encoding batch size |

## API Endpoints

```
GET /
GET /recommendations/{user_id}?k=10&method=svd|content|hybrid   (default: svd)
GET /users/{user_id}/compatibility/{other_user_id}
GET /users/{user_id}/cluster
GET /movies/{tmdb_id}/similar?k=10
```

### Example curl calls

```bash
# Status
curl http://127.0.0.1:8000/

# Top-10 SVD recommendations for user 5
curl http://127.0.0.1:8000/recommendations/5

# Top-5 hybrid recommendations
curl "http://127.0.0.1:8000/recommendations/5?k=5&method=hybrid"

# Content-only recommendations
curl "http://127.0.0.1:8000/recommendations/5?k=5&method=content"

# Compatibility between two users
curl http://127.0.0.1:8000/users/5/compatibility/19

# Cluster membership
curl http://127.0.0.1:8000/users/5/cluster

# Movies similar to Saw (tmdb_id=176)
curl "http://127.0.0.1:8000/movies/176/similar?k=5"
```

### Response shapes

```json
// GET /
{"status": "ok", "n_users": 12992, "n_movies": 28154, "embeddings_loaded": true, "hybrid_alpha": 0.5}

// GET /recommendations/{user_id}?method=svd
{
  "user_id": 5,
  "method": "svd",
  "recommendations": [
    {"movieId_tmdb": 240, "title": "The Godfather: Part II", "genres": "Drama, Crime", "score": 1.45},
    ...
  ]
}

// GET /users/{user_id}/compatibility/{other_user_id}
{"user_id": 5, "other_user_id": 19, "cosine_similarity": 0.298584}

// GET /users/{user_id}/cluster
{"user_id": 5, "cluster_id": 0, "cluster_size": 8640}

// GET /movies/{tmdb_id}/similar
{
  "movieId_tmdb": 176,
  "similar_movies": [
    {"movieId_tmdb": 105114, "title": "So Sweet, So Dead", "genres": "Crime, Drama, Horror...", "score": 0.72},
    ...
  ]
}
```

Unknown user or movie IDs return **HTTP 404** with a clear error message.

## Measured Metrics (real runs)

### Dataset

| Metric | Value |
|---|---|
| Users sampled | 15,000 (from 270,896 total) |
| Users after MIN_RATINGS=10 filter | 12,992 |
| Movies in cleaned dataset | 28,154 |
| Ratings in cleaned dataset | 1,416,164 |
| **Avg ratings per user** | **109.0** |
| SVD explained variance (20 components) | 31.4% |

### Phase 1 — 80/20 split (evaluate.py)

| Model | Precision@10 |
|---|---|
| Popularity baseline | 0.0544 |
| **SVD** | **0.1839** |
| Lift | +238.1% |

### Phase 2 — 70/10/20 split, best alpha=0.5 (evaluate_phase2.py)

Overall results (N = 5,000 test users):

| Model | Precision@10 | Pop Skew (log10) | Catalog Coverage |
|---|---|---|---|
| Popularity | 0.0822 | 3.424 | 48 (0.2%) |
| SVD | 0.1514 | 3.196 | 530 (1.9%) |
| Content-only | 0.0019 | 0.758 | 3,497 (12.4%) |
| **Hybrid (α=0.5)** | **0.1537** | **3.192** | **560 (2.0%)** |

Heavy users (≥ 50 train ratings, N = 1,868):

| Model | Precision@10 |
|---|---|
| Popularity | 0.1575 |
| SVD | 0.2727 |
| Content-only | 0.0021 |
| **Hybrid (α=0.5)** | **0.2757** |

> **Content-only note**: low precision@10 but highest catalog diversity (12.4% vs 2% for SVD).
> The hybrid combines SVD's accuracy with content's diversity signal.
>
> **Cold-start note**: no cold-start users exist in this dataset after the MIN_RATINGS=10 filter.

### Cluster sizes (k=4)

| Cluster | Users | % | Character |
|---|---|---|---|
| 0 | 8,640 | 66.5% | Mainstream (large, as expected) |
| 3 | 2,260 | 17.4% | Dramas / classics |
| 2 | 1,409 | 10.8% | Family / lighter fare |
| 1 | 683 | 5.3% | Art-house / prestige cinema |

## Data Flow

```
ratings.csv (26M rows)
  │
  ├─ Pass 1: collect unique userIds → sample 15k
  └─ Pass 2: stream chunks → keep sampled users

                 ↓
    links.csv → join → filter to known TMDB movies
                 ↓
    movies_metadata.csv → clean (numeric id, dedup, parse genres)
                 ↓
    Drop users < 10 ratings
                 ↓
    scipy.sparse csr_matrix (users × movies)
                 ↓
    TruncatedSVD (20 components) → user_factors.npy
                 ↓
    KMeans (k=4) → cluster_labels.npy
                 ↓ ─────────────────────────────── Phase 2
    SentenceTransformer (all-MiniLM-L6-v2)
    → movie_embeddings.npy (28154 × 384, L2-normalized)
                 ↓
    70/10/20 split → tune alpha on val → hybrid_alpha.pkl
                 ↓
    FastAPI loads all artifacts once at startup
```
