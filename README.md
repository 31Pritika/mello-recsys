# Mello RecSys — Local Movie Recommender

A memory-safe, SVD-based movie recommendation service built on the MovieLens + TMDB dataset.
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
├── train_and_save.py        # Full pipeline: clean → SVD → KMeans → artifacts
├── evaluate.py              # precision@10 vs popularity baseline
├── cluster.py               # Elbow analysis + cluster introspection
└── api.py                   # FastAPI service (artifacts loaded once at startup)

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

## Quick Start

```bash
# 1. Activate the venv
source venv/bin/activate

# 2. Train (streams ratings, fits SVD + KMeans, saves artifacts)
python src/train_and_save.py

# 3. Evaluate (precision@10 vs popularity baseline)
python src/evaluate.py

# 4. Cluster analysis (elbow k=2..10 + per-cluster sample)
python src/cluster.py

# 5. Start API
uvicorn src.api:app --host 127.0.0.1 --port 8000

# 6. Smoke test (in another terminal)
python smoke_test.py
```

> **Note**: `train_and_save.py` must complete before the other scripts.

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

## API Endpoints

```
GET /
GET /recommendations/{user_id}?k=10
GET /users/{user_id}/compatibility/{other_user_id}
GET /users/{user_id}/cluster
```

### Example curl calls

```bash
# Status
curl http://127.0.0.1:8000/

# Top-10 recommendations for user 1
curl http://127.0.0.1:8000/recommendations/1

# Top-5 recommendations
curl "http://127.0.0.1:8000/recommendations/1?k=5"

# Compatibility between two users
curl http://127.0.0.1:8000/users/1/compatibility/2

# Cluster membership
curl http://127.0.0.1:8000/users/1/cluster
```

### Response shapes

```json
// GET /
{"status": "ok", "n_users": 13842, "n_movies": 8241}

// GET /recommendations/{user_id}
{
  "user_id": 1,
  "recommendations": [
    {"movieId_tmdb": 862, "title": "Toy Story", "genres": "Animation, Comedy", "score": 4.31},
    ...
  ]
}

// GET /users/{user_id}/compatibility/{other_user_id}
{"user_id": 1, "other_user_id": 2, "cosine_similarity": 0.714285}

// GET /users/{user_id}/cluster
{"user_id": 1, "cluster_id": 0, "cluster_size": 9241}
```

Unknown user IDs return **HTTP 404** with a clear error message.

## Measured Metrics (real run)

| Metric | Value |
|---|---|
| Users sampled | 15,000 (from 270,896 total) |
| Users after MIN_RATINGS=10 filter | 12,992 |
| Movies in cleaned dataset | 28,154 |
| Ratings in cleaned dataset | 1,416,164 |
| **Avg ratings per user** | **109.0** |
| SVD explained variance (20 components) | 31.4% |
| **SVD precision@10** | **0.1839** |
| **Popularity precision@10** | **0.0544** |
| **Lift** | **+238.1%** |

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
                 ↓
    Saved artifacts → FastAPI loads once at startup
```
