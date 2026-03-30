#!/bin/bash
set -e

echo "=========================================="
echo "  RAG Alkalmazás — Setup"
echo "  Mac Mini M4 24GB-re optimalizálva"
echo "=========================================="
echo ""

# --- uv ellenőrzés ---
if ! command -v uv &> /dev/null; then
    echo "📥 uv telepítése..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo ""
fi

echo "✅ uv megtalálva: $(uv --version)"
echo ""

# --- Ollama ellenőrzés ---
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama nincs telepítve!"
    echo "   Telepítsd innen: https://ollama.ai/download"
    exit 1
fi

echo "✅ Ollama megtalálva: $(ollama --version)"
echo ""

# --- Modellek letöltése ---
echo "📥 LLM modell letöltése: qwen2.5:14b (~9 GB)..."
echo "   Ez az első alkalommal hosszabb ideig tarthat."
ollama pull qwen2.5:14b

echo ""
echo "📥 Embedding modell letöltése: bge-m3 (~1.2 GB)..."
ollama pull bge-m3

echo ""
echo "✅ Ollama modellek telepítve:"
ollama list | grep -E "qwen2.5:14b|bge-m3"
echo ""

# --- Python környezet uv-vel ---
echo "🐍 Python környezet létrehozása uv-vel..."
uv sync

echo ""
echo "=========================================="
echo "  ✅ Setup kész!"
echo ""
echo "  Használat:"
echo "    uv run python index.py                  # Indexelés"
echo "    uv run python search.py \"kérdésed\"      # Keresés"
echo "    uv run python search.py --chat          # Interaktív chat"
echo "=========================================="
