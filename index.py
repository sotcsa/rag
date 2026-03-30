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
from indexer.chunker import chunk_document_with_llm, Chunk, ChunkingPerformance
from indexer.embedder import generate_embeddings
from indexer.vectorstore import VectorStore
from indexer.tracker import Tracker

console = Console()


def setup_logging(verbose: bool = False):
    """Logging beállítás Rich formázással, explicit időbélyegekkel."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
    )


def _human_size(size_bytes: int) -> str:
    """Fájlméret ember-olvasható formátumban."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


def show_status(tracker: Tracker, vector_store: VectorStore):
    """Indexelési állapot részletes kiírása."""
    stats = tracker.get_stats()

    # --- Összesítő táblázat ---
    summary = Table(title="📊 Indexelési Összesítés", show_header=False, border_style="cyan")
    summary.add_column("Metrika", style="bold cyan", min_width=25)
    summary.add_column("Érték", style="green", justify="right")

    summary.add_row("Feldolgozott fájlok", str(stats["processed_files"]))
    summary.add_row("Hibás fájlok", str(stats["error_files"]) if stats["error_files"] else "—")
    summary.add_row("Összes chunk", str(stats["total_chunks"]))
    summary.add_row("Összes forrásfájl méret", _human_size(stats["total_size_bytes"]))
    summary.add_row("ChromaDB elemek", str(vector_store.get_count()))

    console.print()
    console.print(summary)

    # --- Fájlonkénti részletes táblázat ---
    processed = tracker.get_all_processed()
    if not processed:
        console.print("\n[yellow]Nincs indexelt dokumentum.[/]")
        console.print("Futtasd: [bold]uv run python index.py[/]")
        return

    detail = Table(title="\n📁 Indexelt Fájlok Részletei", border_style="dim")
    detail.add_column("#", style="dim", justify="right", width=4)
    detail.add_column("Fájlnév", style="cyan", max_width=40)
    detail.add_column("Méret", justify="right", style="green")
    detail.add_column("Chunkok", justify="right", style="yellow")
    detail.add_column("Indexelve", style="dim")
    detail.add_column("Státusz", justify="center")
    detail.add_column("Hash (SHA-256)", style="dim", max_width=16)

    for i, f in enumerate(processed, 1):
        status = f["status"]
        if status == "ok":
            status_display = "[green]✅ OK[/]"
        elif status.startswith("error"):
            error_msg = status.replace("error: ", "").replace("error", "hiba")
            status_display = f"[red]❌ {error_msg[:20]}[/]"
        else:
            status_display = f"[yellow]⚠ {status}[/]"

        # Fájlnév: csak a basename
        filepath = Path(f["file_path"])
        filename = filepath.name

        # Dátum formázás: csak dátum + idő (UTC string rövidítés)
        indexed_at = f["indexed_at"] or "—"
        if "T" in indexed_at:
            indexed_at = indexed_at.replace("T", " ")[:19]

        # Hash rövidítés
        file_hash = f["file_hash"][:12] + "…" if f["file_hash"] and len(f["file_hash"]) > 12 else (f["file_hash"] or "—")

        detail.add_row(
            str(i),
            filename,
            _human_size(f["file_size"] or 0),
            str(f["chunk_count"] or 0),
            indexed_at,
            status_display,
            file_hash,
        )

    console.print(detail)


def index_file(
    file_path: Path,
    vector_store: VectorStore,
    tracker: Tracker,
    logger: logging.Logger,
    progress_callback=None,
    model: str = None,
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
    selected_model = model or config.LLM_MODEL
    logger.info("🧠 Chunkolás: %s (Modell: %s)", filename, selected_model)
    chunks, perf = chunk_document_with_llm(
        doc.content, str(file_path), model=model, progress_callback=progress_callback
    )

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

    return len(chunks), perf


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
        "--file",
        type=str,
        help="Csak egy adott fájl (újra)indexelése (pl. data/minta.pdf)",
    )
    parser.add_argument(
        "--remove",
        type=str,
        help="Adott fájl törlése az indexből és a nyilvántartásból",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Indexelési állapot kiírása",
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"LLM modell a chunkoláshoz (alapértelmezett, vagy Ollama model name, vagy openrouter/model_name)",
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

    # Fájl törlése
    if args.remove:
        file_path = Path(args.remove).resolve()
        console.print(f"[yellow]🗑 Fájl törlése a DB-ből és nyilvántartásból: {file_path}[/]")
        vector_store.delete_by_source(str(file_path))
        tracker.remove_file(file_path)
        console.print("[green]✅ Sikeresen törölve.[/]")
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
    if args.file:
        target_file = Path(args.file).resolve()
        if not target_file.is_file():
            console.print(f"[red]❌ Hiba: A fájl nem található: {target_file}[/]")
            return
        # Csak ezt a fájlt dolgozzuk fel, kényszerítve az újraindexelést
        unprocessed = [target_file]
    else:
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
    processed_count = 0
    start_time = time.time()
    interrupted = False
    all_perf = ChunkingPerformance()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_files = progress.add_task("Fájlok", total=len(unprocessed))
        task_segments = progress.add_task("Szegmensek (LLM)", total=1, visible=False)

        for file_path in unprocessed:
            progress.update(task_files, description=f"[cyan]{file_path.name}[/]")
            progress.update(task_segments, visible=False)
            
            def segment_callback(current, total):
                if not progress.tasks[task_segments].visible:
                    progress.update(task_segments, visible=True)
                # +1 mert a current 0-indexált
                progress.update(
                    task_segments, 
                    description=f"[magenta]  Szegmensek ({current}/{total})[/]", 
                    completed=current, 
                    total=total
                )

            try:
                chunk_count, file_perf = index_file(
                    file_path, vector_store, tracker, logger, segment_callback, model=args.model
                )
                total_chunks += chunk_count
                processed_count += 1
                # Accumulate performance stats
                if file_perf.total_llm_calls > 0:
                    all_perf.total_elapsed += file_perf.total_elapsed
                    all_perf.total_input_tokens += file_perf.total_input_tokens
                    all_perf.total_output_tokens += file_perf.total_output_tokens
                    all_perf.total_llm_calls += file_perf.total_llm_calls
                    all_perf.total_cost_usd += file_perf.total_cost_usd
                    all_perf.segment_stats.extend(file_perf.segment_stats)
                
                speed_info = ""
                if file_perf.total_llm_calls > 0:
                    cost_part = f", ${file_perf.total_cost_usd:.6f}" if file_perf.total_cost_usd > 0 else ""
                    speed_info = f" ({file_perf.total_elapsed:.1f}s, {file_perf.avg_tokens_per_second:.1f} tok/s{cost_part})"
                console.print(
                    f"  [green]✅ {file_path.name}[/] → {chunk_count} chunk{speed_info}"
                )
            except KeyboardInterrupt:
                console.print(
                    f"\n  [yellow]⏭ {file_path.name} — megszakítva (Ctrl+C)[/]"
                )
                console.print(
                    "[yellow]  Nyomj még egy Ctrl+C-t a teljes leálláshoz, "
                    "vagy várd meg a következő fájlt.[/]\n"
                )
                try:
                    # Adjunk esélyt a második Ctrl+C-nek
                    import signal
                    signal.alarm(0)  # reset
                except Exception:
                    pass
                continue
            except Exception as e:
                errors += 1
                tracker.mark_error(file_path, str(e))
                console.print(f"  [red]❌ {file_path.name}[/]: {e}")
                logger.exception("Hiba a fájl feldolgozásakor: %s", file_path)

            progress.advance(task_files)

    elapsed = time.time() - start_time

    # Összesítés
    console.print(f"\n[bold]📊 Összesítés[/]")
    console.print(f"  Idő: {elapsed:.1f} mp")
    console.print(f"  Feldolgozott fájlok: {processed_count}")
    console.print(f"  Hibák: {errors}")
    console.print(f"  Új chunkok: {total_chunks}")
    console.print(f"  Összes chunk a DB-ben: {vector_store.get_count()}")
    
    # LLM sebesség összefoglaló
    if all_perf.total_llm_calls > 0:
        console.print(f"\n[bold]⏱ LLM Sebesség[/]")
        console.print(f"  LLM hívások: {all_perf.total_llm_calls}")
        console.print(f"  LLM összidő: {all_perf.total_elapsed:.1f} mp")
        console.print(f"  Átl. hívás: {all_perf.avg_call_duration:.1f} mp")
        console.print(f"  Input tokenek: {all_perf.total_input_tokens:,}")
        console.print(f"  Output tokenek: {all_perf.total_output_tokens:,}")
        console.print(f"  Átl. sebesség: [bold cyan]{all_perf.avg_tokens_per_second:.1f} tok/s[/]")
        if all_perf.total_cost_usd > 0:
            console.print(f"  [bold yellow]💰 Összes költség: ${all_perf.total_cost_usd:.6f}[/]")
        else:
            console.print(f"  💰 Költség: $0 (lokális)")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print("\n[yellow]👋 Indexelés megszakítva.[/]\n")

