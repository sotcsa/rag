"""
RAG Indexelés — Főprogram

Használat:
    uv run python index.py                        # Inkrementális indexelés (./data/)
    uv run python index.py --source-dir /path     # Egyedi forráskönyvtár
    uv run python index.py --force                # Minden fájl újraindexelése
    uv run python index.py --status               # Indexelési állapot kiírása
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

import config
from indexer.document_loader import load_document
from indexer.chunker import chunk_document_with_llm, Chunk
from indexer.embedder import generate_embeddings
from indexer.vectorstore import VectorStore
from indexer.tracker import Tracker

console = Console()


def setup_logging(verbose: bool = False):
    """Logging beállítás Rich formázással."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def show_status(tracker: Tracker, vector_store: VectorStore):
    """Indexelési állapot kiírása."""
    stats = tracker.get_stats()

    table = Table(title="📊 Indexelési Állapot")
    table.add_column("Metrika", style="cyan")
    table.add_column("Érték", style="green", justify="right")

    table.add_row("Feldolgozott fájlok", str(stats["processed_files"]))
    table.add_row("Hibás fájlok", str(stats["error_files"]))
    table.add_row("Összes chunk", str(stats["total_chunks"]))
    table.add_row(
        "Összes méret",
        f"{stats['total_size_bytes'] / 1024 / 1024:.1f} MB",
    )
    table.add_row("ChromaDB elemek", str(vector_store.get_count()))

    console.print(table)

    # Részletes lista
    processed = tracker.get_all_processed()
    if processed:
        console.print("\n[bold]Fájlok:[/]")
        for f in processed:
            status_icon = "✅" if f["status"] == "ok" else "❌"
            console.print(
                f"  {status_icon} {f['file_path']} "
                f"[dim]({f['chunk_count']} chunk, {f['indexed_at']})[/]"
            )


def index_file(
    file_path: Path,
    vector_store: VectorStore,
    tracker: Tracker,
    logger: logging.Logger,
) -> int:
    """
    Egyetlen fájl feldolgozása és indexelése.
    
    Returns:
        Létrehozott chunkok száma
    """
    filename = file_path.name

    # 1. Dokumentum betöltés
    logger.info("📄 Betöltés: %s", filename)
    doc = load_document(file_path)

    if not doc.content.strip():
        logger.warning("⚠ Üres dokumentum: %s", filename)
        tracker.mark_processed(file_path, chunk_count=0)
        return 0

    # 2. Régi chunkok törlése (ha újraindexelés)
    vector_store.delete_by_source(str(file_path))

    # 3. LLM-alapú chunkolás
    logger.info("🧠 Chunkolás: %s", filename)
    chunks = chunk_document_with_llm(doc.content, str(file_path))

    if not chunks:
        logger.warning("⚠ Nem sikerült chunk-olni: %s", filename)
        tracker.mark_error(file_path, "no chunks")
        return 0

    # 4. Embedding-hez szöveg előkészítés
    # Az összefoglalót is belefűzzük a chunk szövegébe a jobb kereshetőség érdekében
    texts_for_embedding = []
    for chunk in chunks:
        if chunk.summary:
            texts_for_embedding.append(f"{chunk.summary}\n\n{chunk.content}")
        else:
            texts_for_embedding.append(chunk.content)

    # 5. Embedding generálás
    logger.info("🔢 Embedding: %d chunk", len(chunks))
    embeddings = generate_embeddings(texts_for_embedding)

    # 6. ChromaDB-be mentés
    ids = [f"{file_path.stem}_chunk_{c.chunk_index}" for c in chunks]
    documents = [c.content for c in chunks]
    metadatas = [
        {
            "source": c.source_file,
            "filename": file_path.name,
            "chunk_index": c.chunk_index,
            "summary": c.summary,
        }
        for c in chunks
    ]

    vector_store.add_chunks(ids, documents, embeddings, metadatas)

    # 7. Tracker frissítés
    tracker.mark_processed(file_path, chunk_count=len(chunks))

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Indexelés — Dokumentumok feldolgozása és indexelése"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(config.DATA_DIR),
        help=f"Dokumentumok forráskönyvtára (alapértelmezett: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Minden fájl újraindexelése (tracker törlése)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Indexelési állapot kiírása",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Részletes naplózás",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("index")

    tracker = Tracker()
    vector_store = VectorStore()

    # Állapot kiírás
    if args.status:
        show_status(tracker, vector_store)
        return

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        source_dir.mkdir(parents=True)
        console.print(
            f"[yellow]📁 Forráskönyvtár létrehozva: {source_dir}[/]\n"
            f"Másold ide a dokumentumokat, majd futtasd újra."
        )
        return

    # Force mód: tracker törlése
    if args.force:
        console.print("[yellow]⚠ Force mód: minden fájl újraindexelése[/]")
        tracker.clear()

    # Feldolgozandó fájlok meghatározása
    unprocessed = tracker.get_unprocessed_files(source_dir)

    if not unprocessed:
        console.print("[green]✅ Minden fájl naprakész, nincs új feldolgozandó dokumentum.[/]")
        show_status(tracker, vector_store)
        return

    console.print(
        f"\n[bold]📋 Feldolgozandó fájlok: {len(unprocessed)}[/]\n"
    )
    for f in unprocessed:
        console.print(f"  • {f.name}")
    console.print()

    # Feldolgozás
    total_chunks = 0
    errors = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexelés...", total=len(unprocessed))

        for file_path in unprocessed:
            progress.update(task, description=f"[cyan]{file_path.name}[/]")
            try:
                chunk_count = index_file(file_path, vector_store, tracker, logger)
                total_chunks += chunk_count
                console.print(
                    f"  [green]✅ {file_path.name}[/] → {chunk_count} chunk"
                )
            except Exception as e:
                errors += 1
                tracker.mark_error(file_path, str(e))
                console.print(f"  [red]❌ {file_path.name}[/]: {e}")
                logger.exception("Hiba a fájl feldolgozásakor: %s", file_path)

            progress.advance(task)

    elapsed = time.time() - start_time

    # Összesítés
    console.print(f"\n[bold]📊 Összesítés[/]")
    console.print(f"  Idő: {elapsed:.1f} mp")
    console.print(f"  Feldolgozott fájlok: {len(unprocessed) - errors}")
    console.print(f"  Hibák: {errors}")
    console.print(f"  Új chunkok: {total_chunks}")
    console.print(f"  Összes chunk a DB-ben: {vector_store.get_count()}")
    console.print()


if __name__ == "__main__":
    main()
