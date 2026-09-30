"""Central configuration for mello-recsys."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "model_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RATINGS_CSV   = DATA_DIR / "ratings.csv"
MOVIES_CSV    = DATA_DIR / "movies_metadata.csv"
LINKS_CSV     = DATA_DIR / "links.csv"

RATINGS_CLEAN_PARQUET = ARTIFACTS_DIR / "ratings_clean.parquet"
MOVIES_CLEAN_PARQUET  = ARTIFACTS_DIR / "movies_clean.parquet"
SVD_PATH              = ARTIFACTS_DIR / "svd.joblib"
KMEANS_PATH           = ARTIFACTS_DIR / "kmeans.joblib"
USER_FACTORS_PATH     = ARTIFACTS_DIR / "user_factors.npy"
CLUSTER_LABELS_PATH   = ARTIFACTS_DIR / "cluster_labels.npy"
USER_TO_IDX_PATH      = ARTIFACTS_DIR / "user_to_idx.joblib"
IDX_TO_USER_PATH      = ARTIFACTS_DIR / "idx_to_user.joblib"
MOVIE_TO_IDX_PATH     = ARTIFACTS_DIR / "movie_to_idx.joblib"
MOVIE_IDS_PATH        = ARTIFACTS_DIR / "movie_ids.npy"

# ── Sampling ───────────────────────────────────────────────────────────────────
N_USERS       = 15_000   # how many random users to sample
SAMPLE_SEED   = 42
CHUNK_SIZE    = 500_000  # rows per chunk when streaming ratings.csv
MIN_RATINGS   = 10       # drop users with fewer ratings

# ── Model ──────────────────────────────────────────────────────────────────────
N_COMPONENTS  = 20
SVD_SEED      = 42
KMEANS_K      = 4
KMEANS_SEED   = 42

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_N_USERS  = 5_000
EVAL_SEED     = 42
LIKE_THRESHOLD = 4.0
TOP_K          = 10

# ── Phase 2: Content & Hybrid ──────────────────────────────────────────────────
MOVIE_EMBEDDINGS_PATH          = ARTIFACTS_DIR / "movie_embeddings.npy"
HYBRID_ALPHA_PATH              = ARTIFACTS_DIR / "hybrid_alpha.pkl"
EMBEDDING_MODEL_NAME           = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE           = 64
MIN_TRAIN_RATINGS_CANDIDATE    = 20   # candidate filter: min train ratings for a movie to be recommendable

# ── Phase 2 Upgrades: Implicit ALS & LGBMRanker ────────────────────────────────
ALS_MODEL_PATH                 = ARTIFACTS_DIR / "als_model.pkl"
ALS_USER_FACTORS_PATH          = ARTIFACTS_DIR / "als_user_factors.npy"
ALS_ITEM_FACTORS_PATH          = ARTIFACTS_DIR / "als_item_factors.npy"
ALS_FACTORS                    = 64
ALS_REGULARIZATION             = 0.05
ALS_ITERATIONS                 = 15
ALS_ALPHA                      = 40.0
ALS_SEED                       = 42

RERANKER_MODEL_PATH            = ARTIFACTS_DIR / "reranker_lgbm.pkl"
