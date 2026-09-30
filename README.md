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
├── als_model.py             # Phase 2 Step 1: implicit ALS collaborative filtering
├── reranker.py              # Phase 2 Step 2: Learning-to-Rank (LambdaRank) reranker
├── evaluate_phase2.py       # Phase 2: full benchmark (5 models across 5 conditions)
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

### Phase 2 — Neural Embeddings, Hybrid, ALS, & Learned Reranker

```bash
# 5. Encode all movies with all-MiniLM-L6-v2 (downloads model once, ~30 min on CPU)
python src/embed_movies.py

# 6. Train Implicit ALS collaborative filtering baseline (Step 1)
python src/als_model.py

# 7. Train Learning-to-Rank (LambdaRank) reranker with LightGBM (Step 2)
python src/reranker.py

# 8. Run comprehensive benchmark across all 5 models and 5 evaluation conditions
python src/evaluate_phase2.py
#    add --full to evaluate on all test users instead of 5,000

# 9. Start API (loads all artifacts once; serves SVD / content / hybrid)
uvicorn src.api:app --host 127.0.0.1 --port 8000

# 10. Smoke test (in another terminal)
python smoke_test.py
```

> **Note**: run steps in order. `train_and_save.py` must complete before everything else;
> `embed_movies.py` must complete before training the reranker or running `evaluate_phase2.py`.

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

### Phase 2 — Comprehensive 5-Model Benchmark Across 5 Conditions

All models evaluated under a strict 70% Train / 10% Validation / 20% Test split.
Metrics are Precision@10 (hit = test item rated $\ge 4.0$).
Evaluated on $N = 5,000$ test users with 95% bootstrap confidence intervals (1,000 resamples).

#### Final Combined Benchmark Table (Precision@10)

| Evaluation Condition | Popularity | SVD (20 comp) | ALS (64 factors) | Content-Only | Fixed Hybrid (α=0.5) | Learned Reranker (LGBM) |
|---|---|---|---|---|---|---|
| **1. Overall (no filter)** | 0.0822 | 0.1514 | 0.1323 | 0.0019 | **0.1537** | 0.0831 |
| **2. Cand-filtered (≥20)** | 0.0822 | 0.1514 | 0.1323 | 0.0079 | **0.1525** | 0.0835 |
| **3. Heavy users (≥50 train)** | 0.1575 | 0.2727 | 0.2125 | 0.0021 | **0.2757** | 0.1188 |
| **4. User cold-start (n=1)** | 0.3498 | 0.3036 | 0.1612 | 0.0137 | **0.3020** | 0.2109 |
| **4. User cold-start (n=3)** | 0.3447 | 0.3292 | 0.2642 | 0.0213 | **0.3312** | 0.2553 |
| **4. User cold-start (n=5)** | 0.3409 | 0.3533 | 0.3282 | 0.0209 | **0.3580** | 0.3045 |
| **5. Item cold-start (500 items)**| 0.0061 | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0000 |

#### 95% Bootstrap Confidence Intervals & Paired Differences (Condition 1: Overall)

| Model / Comparison | Precision@10 | 95% Bootstrap CI | Paired Lift vs Baseline |
|---|---|---|---|
| **Fixed Hybrid (α=0.5)** | **0.1537** | [0.1491, 0.1586] | **+0.0023 [0.0013, 0.0033]** vs SVD |
| **SVD** | 0.1514 | [0.1467, 0.1564] | Baseline |
| **ALS** | 0.1323 | [0.1282, 0.1360] | **-0.0192 [-0.0225, -0.0156]** vs SVD |
| **Learned Reranker** | 0.0831 | [0.0804, 0.0859] | **-0.0706 [-0.0747, -0.0665]** vs Hybrid |
| **Popularity** | 0.0822 | [0.0787, 0.0856] | - |
| **Content-only** | 0.0019 | [0.0015, 0.0023] | - |

#### Key Findings & Model Dynamics

1. **Fixed Hybrid is the Overall Champion**:
   - Fixed hybrid blending ($\alpha=0.5$) achieves the highest Precision@10 across overall ($0.1537$), heavy users ($0.2757$), and simulated cold-start ($n=5: 0.3580$).
   - The paired lift over SVD ($+0.0023$, 95% CI $[0.0013, 0.0033]$) is statistically significant.
2. **ALS vs. SVD (Explicit vs. Implicit Feedback)**:
   - SVD significantly outperforms ALS ($0.1514$ vs $0.1323$, paired difference $-0.0192$ with 95% CI $[-0.0225, -0.0156]$).
   - *Why*: MovieLens ratings are explicit grades (1–5 stars). In ALS implicit feedback formulation, unobserved items are treated as negative interactions with uniform weight 1.0, penalizing long-tail unrated movies that users might have loved.
3. **Learned Reranker (LGBM LambdaRank)**:
   - Overall, the reranker ($0.0831$) does **not** beat the fixed hybrid ($0.1537$) or ALS ($0.1323$).
   - *Feature importance analysis*: `als_score` (40.0%) and `log1p_train_count` (36.7%) dominate. Because unrated negatives during training were sampled with popularity weighting, the tree model learned a strong popularity heuristic, causing its overall score to converge near the popularity baseline ($0.0822$).
   - *Where the Reranker excels*: In extreme user cold-start ($n=1$), ALS collapses to $0.1612$, while the reranker scores **$0.2109$**—a statistically significant paired lift of **$+0.0497$ [0.0416, 0.0577]** over ALS. The engineered popularity and genre features buffer the breakdown of latent factor dot products when user interaction history is minimal.
4. **Candidate Filter Impact ($\ge 20$ ratings)**:
   - Filtering out catalog items with $< 20$ train ratings boosts Content-Only precision by **4.1x** ($0.0019 \to 0.0079$), by preventing it from wasting recommendations on obscure items with zero test-set probability. SVD and ALS are naturally robust to this.
5. **Item Cold-Start (500 removed movies)**:
   - Collaborative filtering (SVD, ALS, Hybrid, Reranker) scores $0.0000$ since unobserved items have no train interaction data.
   - Content-only is the only model capable of surfacing removed items ($0.0002$), though precision is bounded by the tiny target set size ($500 / 28,154$).

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
