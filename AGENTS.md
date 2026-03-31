# AGENTS.md — AI Agent Context

## Project Overview

This is a fully local RAG (Retrieval-Augmented Generation) application optimized for **Mac Mini M4 24GB**. It processes PDF, DOCX, and TXT documents using LLM-based semantic chunking and stores them in a portable ChromaDB vector database. All processing runs locally via Ollama — no data leaves the machine.

**Primary language of documents and queries: Hungarian** (secondary: English).

## Architecture

```
User Documents (PDF/DOCX/TXT)
    ↓
Document Loader (pymupdf4llm / python-docx)
    ↓
Raw Text (Markdown format)
    ↓
LLM Chunker (qwen2.5:14b via Ollama) → semantic chunks with summaries
    ↓
Embedder (bge-m3 via Ollama) → 1024-dim vectors
    ↓
ChromaDB (persistent, file-based) + SQLite tracker
    ↓
Search: query → embed → vector search → LLM answer generation
```

## Tech Stack

| Component | Technology | Notes |
|:--|:--|:--|
| LLM | `qwen2.5:14b` via Ollama | Chosen for strong Hungarian multilingual support. ~10GB RAM. |
| Embedding | `bge-m3` via Ollama | Multilingual dense/sparse/multi-vector retrieval. 1024-dim. |
| Vector DB | ChromaDB (PersistentClient) | File-based, portable. Stored in `chroma_db/` directory. |
| Tracking | SQLite (`tracking.db`) | SHA-256 hash-based incremental indexing. |
| PDF | `pymupdf4llm` | Outputs Markdown preserving headers/tables. |
| DOCX | `python-docx` | Paragraph-level extraction with heading styles. |
| Markdown | Direct file read | `.md` files read as-is, no conversion needed. |
| Package mgr | `uv` | `pyproject.toml` is the source of truth for dependencies. |
| CLI output | `rich` | Formatted tables, progress bars, streaming output. |
| Env config | `python-dotenv` | Secrets and API keys loaded from `.env`. |

## Directory Structure

```
rag/
├── config.py                  # All configuration (models, paths, parameters)
├── pyproject.toml             # uv project file + dependencies
├── requirements.txt           # Legacy reference (pyproject.toml is canonical)
├── setup.sh                   # One-time setup: Ollama models + uv sync
├── ollama_client.py           # Shared Ollama client with explicit loopback + timeout
├── benchmark.py               # Chunking benchmark tool for model comparison
├── .env                       # Local secrets (e.g. OpenRouter API key, ignored by git)
├── .env.example               # Template for the .env file
├── .gitignore
│
├── indexer/                   # Document processing pipeline
│   ├── __init__.py
│   ├── document_loader.py     # PDF/DOCX/TXT/.md → Document dataclass
│   ├── chunker.py             # LLM-based semantic chunking (Hungarian prompts)
│   ├── embedder.py            # bge-m3 embedding generation via Ollama
│   ├── vectorstore.py         # ChromaDB persistent storage wrapper
│   └── tracker.py             # SQLite-based incremental indexing tracker
│
├── search/                    # Query and answer pipeline
│   ├── __init__.py
│   ├── retriever.py           # Embedding query + ChromaDB vector search
│   ├── generator.py           # RAG prompt + LLM answer generation (streaming)
│   └── cli.py                 # Interactive chat, single query, source listing
│
├── index.py                   # Entry point: indexing
├── search.py                  # Entry point: search/chat
│
├── data/                      # Source documents (gitignored, user-provided)
├── chroma_db/                 # ChromaDB persistent storage (gitignored, portable)
└── tracking.db                # SQLite indexing log (gitignored, portable)
```

## Key Design Decisions

### LLM-Based Chunking (`indexer/chunker.py`)
- Documents are pre-segmented into ~3000-char blocks at paragraph boundaries
- Each block is sent to `qwen2.5:14b` with a Hungarian prompt asking it to identify logical semantic units
- The LLM returns JSON: `[{"summary": "...", "content": "..."}, ...]`
- **Robust JSON Parsing**: Custom `_fix_json_strings()` handles invalid escapes (`\(`, `\[`), literal newlines in strings, and unescaped interior quotes
- **LLM Retry Logic**: Each segment has up to 3 retries with a 2-second delay before falling back to chunk-based chunking
- **OpenRouter Support**: Chunking supports external cloud models (e.g., `openrouter/google/gemini-2.5-flash-free`) via API if configured in `.env`. Also supports `openrouter/free` and `openrouter/auto` for auto-selecting free models.
- **Fallback**: If LLM fails or returns unparseable output after all retries, recursive character-based chunking (2000 chars, 200 overlap) is used
- Summaries are prepended to chunk text before embedding for better retrieval

### Ollama Client (`ollama_client.py`)
- Uses explicit `127.0.0.1` loopback IP to avoid DNS resolution issues (works offline / no WiFi)
- Configures `300` second timeout (5 minutes) to handle slow chunking of large segments with 14B models
- Shared client instance used by both chunking and search pipelines

### Incremental Indexing (`indexer/tracker.py`)
- Each file's SHA-256 hash is stored in SQLite
- On re-run: only new files or files with changed hashes are processed
- Changed files: old chunks are deleted from ChromaDB, then re-indexed
- `--force` flag clears the tracker to re-index everything

### ChromaDB Portability (`indexer/vectorstore.py`)
- Uses `PersistentClient` — all data in `chroma_db/` directory
- To migrate: stop app → copy `chroma_db/` + `tracking.db` → done
- Cosine similarity metric for vector search

### Search Pipeline (`search/`)
- Query → bge-m3 embedding → ChromaDB top-5 search → similarity filtering (≥0.3)
- Retrieved chunks + query → qwen2.5:14b with Hungarian RAG system prompt
- Streaming token output in CLI for responsive UX

## Working With This Codebase

### Running commands
```bash
# Indexing
uv run python index.py                    # Incremental indexing
uv run python index.py --model modelname  # Override chunking model (e.g., qwen2.5:7b or openrouter/...)
uv run python index.py --source-dir /path # Custom source directory
uv run python index.py --status           # Show indexed files detail
uv run python index.py --force            # Force full re-index
uv run python index.py --file path/to/doc # Force re-index a single document
uv run python index.py --remove path/doc  # Remove a single document from the index
uv run python index.py --verbose          # Detailed logging

# Search
uv run python search.py "question"        # Single query
uv run python search.py --chat            # Interactive chat
uv run python search.py --list-sources    # List indexed sources

# Benchmark (model comparison)
uv run python benchmark.py                # Compare default models on all data/ files
uv run python benchmark.py --file f.txt   # Single file benchmark
uv run python benchmark.py --models a,b   # Custom model list (comma-separated)
uv run python benchmark.py --search-test  # Add embedding similarity relevance test
uv run python benchmark.py --save         # Save results to JSON
uv run python benchmark.py --show-summaries # Show summaries per model
uv run python benchmark.py --verbose      # Detailed logging
```

### Configuration
All tuneable parameters are in `config.py`. Key ones:
- `LLM_MODEL` / `EMBEDDING_MODEL` — Ollama model names
- `PRE_SEGMENT_SIZE` — Pre-chunking block size (chars)
- `SEARCH_TOP_K` — Number of results to retrieve
- `SEARCH_MIN_SIMILARITY` — Minimum cosine similarity threshold
- `LLM_NUM_CTX` — Context window size (affects RAM usage)

### Adding new document formats
1. Add a `load_xxx()` function in `indexer/document_loader.py` returning a `Document` dataclass
2. Register the extension in `load_document()` dispatcher
3. Add the extension to `SUPPORTED_EXTENSIONS` in `config.py`

### Modifying chunking behavior
- The chunking prompt is in `indexer/chunker.py` (`CHUNKING_USER_PROMPT`)
- Adjust `PRE_SEGMENT_SIZE` in `config.py` to control how much text the LLM sees at once
- The fallback chunker params are `FALLBACK_CHUNK_SIZE` / `FALLBACK_CHUNK_OVERLAP`

### Modifying search/answer behavior
- The RAG system prompt is in `search/generator.py` (`RAG_SYSTEM_PROMPT`)
- Answer temperature is hardcoded at 0.3 in `generator.py`
- Chunking temperature is 0.1 (in `config.py`) for deterministic output

### Benchmark Tool (`benchmark.py`)
The benchmark tool compares chunking models across multiple dimensions:
- **Speed**: Total time, token/s, speedup/slowdown vs reference model
- **Quality**: Chunk count, word statistics (avg/median/min/max), fallback percentage, summary coverage
- **Search Relevance**: Embedding-based cosine similarity test with configurable Hungarian queries (requires `--search-test`)
- **Cost**: OpenRouter API cost tracking (`$` per model)

Results are displayed in Rich comparison tables and can be saved to `benchmark_results/` (add this directory to `.gitignore`).

## Dependencies (from pyproject.toml)
- `chromadb>=0.5.0` — Vector database
- `pymupdf4llm>=0.0.10` — PDF extraction (depends on pymupdf)
- `python-docx>=1.1.0` — DOCX extraction
- `ollama>=0.4.0` — Ollama Python SDK
- `rich>=13.0` — Terminal formatting
- `python-dotenv>=1.2.2` — Environment variable loading

## External Dependencies (not pip)
- **Ollama** must be installed and running (`https://ollama.ai`)
- Models must be pulled: `ollama pull qwen2.5:14b` and `ollama pull bge-m3`
- **OpenRouter API key** (optional) — configured via `OPENROUTER_API_KEY` in `.env` for cloud-based chunking acceleration
