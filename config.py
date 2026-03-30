"""
RAG alkalmazás konfigurációja.
Mac Mini M4 24GB-re optimalizálva.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Útvonalak ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
TRACKING_DB_PATH = BASE_DIR / "tracking.db"

# --- Ollama beállítások ---
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# LLM modell chunkoláshoz és válaszgeneráláshoz
LLM_MODEL = "qwen2.5:14b"
LLM_TEMPERATURE = 0.1  # Alacsony hőmérséklet a determinisztikus chunkoláshoz
LLM_NUM_CTX = 8192  # Kontextus ablak méret (RAM-barát)

# --- OpenRouter beállítások ---
# Ha OpenRouter modelleket használnál (pl. a gyorsabb google/gemini-2.5-flash-free), töltsd ki ezt a .env fájlban!
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Embedding modell
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIMENSIONS = 1024  # bge-m3 output dimenzió

# --- Chunkolási beállítások ---
# LLM-alapú chunkolás
PRE_SEGMENT_SIZE = 3000  # Karakterek száma az előszegmentáláshoz
PRE_SEGMENT_OVERLAP = 300  # Átfedés az előszegmentek között

# Fallback chunkolás (ha az LLM nem elérhető)
FALLBACK_CHUNK_SIZE = 2000  # Karakterek
FALLBACK_CHUNK_OVERLAP = 200  # Átfedés

# --- ChromaDB beállítások ---
CHROMA_COLLECTION_NAME = "documents"

# --- Keresési beállítások ---
SEARCH_TOP_K = 5  # Hány releváns chunkot kérjünk
SEARCH_MIN_SIMILARITY = 0.3  # Minimum hasonlóság (0-1, alacsonyabb = több eredmény)

# --- Támogatott fájlformátumok ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
