from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "resume_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
SEARCH_TYPE = "similarity_score_threshold"  # Options: "similarity", "similarity_score_threshold", "mmr"
SIMILARITY_THRESHOLD = 0.1

### mmr - Maximum Marginal Relevance.
### similarity - Similarity Search.
### similarity_score_threshold - Similarity Search with Score Threshold.
