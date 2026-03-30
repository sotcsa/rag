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
| Package mgr | `uv` | `pyproject.toml` is the source of truth for dependencies. |
| CLI output | `rich` | Formatted tables, progress bars, streaming output. |

## Directory Structure

```
rag/
├── config.py                  # All configuration (models, paths, parameters)
├── pyproject.toml             # uv project file + dependencies
├── requirements.txt           # Legacy reference (pyproject.toml is canonical)
├── setup.sh                   # One-time setup: Ollama models + uv sync
├── .gitignore
│
├── indexer/                   # Document processing pipeline
│   ├── __init__.py
│   ├── document_loader.py     # PDF/DOCX/TXT → Document dataclass
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
- Robust JSON parsing handles markdown code blocks, extra text around JSON
- **Fallback**: If LLM fails or returns unparseable output, recursive character-based chunking (2000 chars, 200 overlap) is used
- Summaries are prepended to chunk text before embedding for better retrieval

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
uv run python index.py                    # Incremental indexing
uv run python index.py --status           # Show indexed files detail
uv run python index.py --force            # Force full re-index
uv run python search.py "question"        # Single query
uv run python search.py --chat            # Interactive chat
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

## Dependencies (from pyproject.toml)
- `chromadb>=0.5.0` — Vector database
- `pymupdf4llm>=0.0.10` — PDF extraction (depends on pymupdf)
- `python-docx>=1.1.0` — DOCX extraction
- `ollama>=0.4.0` — Ollama Python SDK
- `rich>=13.0` — Terminal formatting

## External Dependencies (not pip)
- **Ollama** must be installed and running (`https://ollama.ai`)
- Models must be pulled: `ollama pull qwen2.5:14b` and `ollama pull bge-m3`
