# 🔍 Lokális RAG — Dokumentumkeresés AI-val

Teljesen lokálisan futó, **magyar nyelvű** dokumentumkereső rendszer, amely a fájljaidból (PDF, DOCX, TXT) intelligens tudásbázist épít. Kérdéseket tehetsz fel magyarul, és a rendszer a dokumentumaid alapján válaszol — mindezt az adatok megosztása nélkül.

## ✨ Jellemzők

- **100% lokális** — Minden a saját gépeden fut, az adatok soha nem hagyják el a gépet
- **Ingyenes** — Ollama open-source modelleket használ, nincs API költség
- **Magyar nyelvű** — Dokumentumok feldolgozása és keresés magyarul (angol is támogatott)
- **Intelligens chunkolás** — LLM-alapú szövegszegmentálás a legjobb keresési minőségért
- **Inkrementális indexelés** — Csak az új/módosult fájlokat dolgozza fel
- **Hordozható adatbázis** — Az egész tudásbázis egy mappa, ami USB-n átvihető másik gépre
- **Felhős gyorsítás** — OpenRouter integráció a tízszer gyorsabb, ingyenes chunkoláshoz (opcionális, `.env` fájlból)

## 🖥️ Rendszerkövetelmények

- **Mac Mini M4 24GB** (vagy hasonló Apple Silicon / 16GB+ RAM)
- **Ollama** telepítve ([letöltés](https://ollama.ai/download))
- **Python 3.11+**
- **uv** package manager ([letöltés](https://docs.astral.sh/uv/getting-started/installation/))
- ~10 GB szabad hely (Ollama modellek)

## 🚀 Telepítés

```bash
# 1. Klónozd a repót
git clone <repo-url>
cd rag

# 2. Futtasd a setup scriptet (Ollama modellek + Python függőségek)
chmod +x setup.sh
./setup.sh
```

Ez letölti a szükséges AI modelleket (~10 GB összesen):
- `qwen2.5:14b` — LLM a chunkoláshoz és válaszgeneráláshoz
- `bge-m3` — Embedding modell a kereséshez

### Beállítás (.env)

Hozd létre a `.env` fájlt a gyökérkönyvtárban a `.env.example` alapján:
```env
OPENROUTER_API_KEY="sk-or-v1-saját-kulcs"
```
Erre akkor van szükség, ha gyorsabb, felhős AI modellt szeretnél használni a feldolgozáshoz (pl. ingyenes Gemini). Minden egyéb konfiguráció (pl. mappák, default modellek) a `config.py` fájlban található.

## 📖 Használat

### 1. Dokumentumok elhelyezése

Másold a feldolgozandó fájlokat a `data/` mappába:

```bash
cp ~/dokumentumok/*.pdf data/
cp ~/dokumentumok/*.docx data/
cp ~/dokumentumok/*.txt data/
```

Almappák is támogatottak — a rendszer rekurzívan bejárja a könyvtárat.

### 2. Indexelés

```bash
# Inkrementális indexelés (csak új/módosult fájlok)
uv run python index.py

# Gyorsabb lokális modellel
uv run python index.py --model qwen2.5:7b

# Ingyenes, villámgyors felhős modellel (OpenRouter)
uv run python index.py --model openrouter/google/gemini-2.5-flash-free

# Egy adott fájl újraindexelése
uv run python index.py --file data/minta.pdf

# Egy fájl törlése az adatbázisból és az indexből
uv run python index.py --remove data/torlendo.pdf

# Egyedi forráskönyvtár megadása
uv run python index.py --source-dir /path/to/documents

# Teljes újraindexelés (mindent elölről)
uv run python index.py --force

# Indexelési állapot megtekintése (mely fájlok, hány chunk, stb.)
uv run python index.py --status
```

### 3. Keresés

```bash
# Egyszeri kérdés
uv run python search.py "Mi az a RAG rendszer?"

# Interaktív chat mód (folyamatos kérdezés)
uv run python search.py --chat

# Indexelt források listázása
uv run python search.py --list-sources
```

## 📊 Status kimenete

```
       📊 Indexelési Összesítés
┌───────────────────────────┬────────┐
│ Feldolgozott fájlok       │      3 │
│ Hibás fájlok              │      — │
│ Összes chunk              │     42 │
│ Összes forrásfájl méret   │ 1.2 MB │
│ ChromaDB elemek           │     42 │
└───────────────────────────┴────────┘

                📁 Indexelt Fájlok Részletei
┏━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃  # ┃ Fájlnév          ┃  Méret ┃ Chunkok ┃ Indexelve           ┃ Státusz ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│  1 │ szerzodes.pdf    │ 850 KB │      28 │ 2026-03-30 07:18:22 │  ✅ OK  │
│  2 │ szabalyzat.docx  │ 320 KB │      11 │ 2026-03-30 07:20:45 │  ✅ OK  │
│  3 │ jegyzetek.txt    │  12 KB │       3 │ 2026-03-30 07:21:01 │  ✅ OK  │
└────┴──────────────────┴────────┴─────────┴─────────────────────┴─────────┘
```

## 🔄 Átvitel másik gépre

A teljes tudásbázis internet nélkül átvihető:

```bash
# --- Forrásgépen ---
# Állítsd le az alkalmazást, majd csomagold be:
tar -czf rag_backup.tar.gz chroma_db/ tracking.db

# --- Célgépen ---
# 1. Telepítsd a kódot + függőségeket
uv sync

# 2. Telepítsd az Ollama modelleket
ollama pull qwen2.5:14b
ollama pull bge-m3

# 3. Csomagold ki a tudásbázist
tar -xzf rag_backup.tar.gz

# Kész! A keresés azonnal működik:
uv run python search.py --chat
```

> ⚠️ **Fontos**: A `chroma_db/` és `tracking.db` másolása előtt mindig állítsd le az alkalmazást!

## ⚙️ Konfiguráció

A `config.py` fájlban módosítható:

| Beállítás | Alapérték | Leírás |
|:--|:--|:--|
| `OPENROUTER_API_KEY` | `.env`-ből | API kulcs a felhős modellekhez (opcionális) |
| `LLM_MODEL` | `qwen2.5:14b` | LLM modell neve |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding modell neve |
| `DATA_DIR` | `./data` | Dokumentumok forráskönyvtára |
| `SEARCH_TOP_K` | `5` | Keresési eredmények száma |
| `PRE_SEGMENT_SIZE` | `3000` | Előszegmentálási méret (karakter) |

## 🏗️ Projekt felépítés

```
rag/
├── .env                   # Jelszavak, kulcsok (lokális fájl)
├── .env.example           # Template a .env-hez
├── config.py              # Konfiguráció
├── pyproject.toml         # Függőségek (uv)
├── setup.sh               # Telepítő script
├── index.py               # 📥 Indexelés belépési pont
├── search.py              # 🔍 Keresés belépési pont
│
├── indexer/               # Dokumentum feldolgozás
│   ├── document_loader.py #   PDF/DOCX/TXT beolvasás
│   ├── chunker.py         #   LLM-alapú szemantikus chunkolás
│   ├── embedder.py        #   Embedding generálás (bge-m3)
│   ├── vectorstore.py     #   ChromaDB kezelés
│   └── tracker.py         #   Inkrementális indexelés nyilvántartás
│
├── search/                # Keresés és válaszgenerálás
│   ├── retriever.py       #   Vektor keresés
│   ├── generator.py       #   LLM válaszgenerálás
│   └── cli.py             #   CLI felület (chat + egyszeri kérdés)
│
├── data/                  # 📁 Dokumentumok helye
├── chroma_db/             # 💾 Vektor adatbázis (hordozható)
└── tracking.db            # 📋 Feldolgozási napló
```

## 📝 Licensz

Privát felhasználásra.
