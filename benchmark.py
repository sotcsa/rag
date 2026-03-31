"""
Chunkolási Benchmark — Modellek összehasonlítása

Használat:
    uv run python benchmark.py                              # Benchmark az összes data/ fájlon
    uv run python benchmark.py --file data/teszt.txt        # Egyetlen fájl
    uv run python benchmark.py --models qwen2.5:14b,qwen3.5 # Modellek megadása
    uv run python benchmark.py --search-test                # Keresési relevancia teszt is
    uv run python benchmark.py --save                       # Eredmények mentése JSON-be

A script:
1. Betölti a dokumentumokat
2. Mindegyik modellel lefuttatja a chunkolást
3. Összehasonlító táblázatot ír ki (sebesség, minőség metrikák)
4. Opcionálisan keresési relevancia tesztet is futtat
"""

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

import config
from indexer.document_loader import load_document
from indexer.chunker import chunk_document_with_llm, Chunk, ChunkingPerformance

console = Console()


# --- Alapértelmezett tesztelendő modellek ---
# A referencia modell az első (qwen2.5:14b), a többi az összehasonlítandó
DEFAULT_MODELS = [
    "qwen2.5:14b",    # Referencia (jelenlegi)
    "qwen3.5",        # Telepített, gyorsabb
    "gemma3:12b",     # Telepített, hasonló méret
]

# Keresési relevancia tesztkérdések (magyar)
# Ezeket a kérdéseket a benchmark automatikusan felteszi minden modell chunk-jaira
DEFAULT_TEST_QUERIES = [
    "Mi a dokumentum fő témája?",
    "Milyen fontos részleteket tartalmaz a szöveg?",
    "Milyen összefüggéseket mutat be a dokumentum?",
]


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
    )


def get_available_models(requested_models: list[str]) -> list[str]:
    """Ellenőrzi, hogy a kért modellek elérhetők-e az Ollama-ban."""
    from ollama_client import client

    available = []
    try:
        model_list = client.list()
        installed = {m["model"].split(":")[0] for m in model_list.get("models", [])}
        # Pontos névvel is ellenőrizzük (pl. qwen2.5:14b)
        installed_full = {m["model"] for m in model_list.get("models", [])}
    except Exception as e:
        console.print(f"[red]❌ Ollama nem elérhető: {e}[/]")
        return []

    for model in requested_models:
        if model.startswith("openrouter/"):
            available.append(model)
            continue
        base_name = model.split(":")[0]
        if model in installed_full or base_name in installed:
            available.append(model)
        else:
            console.print(f"[yellow]⚠ Modell nem telepített, kihagyom: {model}[/]")
            console.print(f"  Telepítés: [bold]ollama pull {model}[/]")

    return available


def analyze_chunks(chunks: list[Chunk]) -> dict:
    """Chunk minőségi metrikák kiszámítása."""
    if not chunks:
        return {
            "count": 0,
            "avg_words": 0,
            "median_words": 0,
            "min_words": 0,
            "max_words": 0,
            "std_words": 0,
            "has_summary_pct": 0,
            "avg_summary_len": 0,
            "total_chars": 0,
        }

    word_counts = [len(c.content.split()) for c in chunks]
    summaries = [c.summary for c in chunks if c.summary]
    summary_lens = [len(s.split()) for s in summaries]

    return {
        "count": len(chunks),
        "avg_words": statistics.mean(word_counts),
        "median_words": statistics.median(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "std_words": statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
        "has_summary_pct": len(summaries) / len(chunks) * 100,
        "avg_summary_len": statistics.mean(summary_lens) if summary_lens else 0,
        "total_chars": sum(len(c.content) for c in chunks),
    }


def run_search_relevance_test(
    chunks: list[Chunk],
    queries: list[str],
    model_name: str,
) -> list[dict]:
    """
    Keresési relevancia teszt: embed-eli a chunkokat és a kérdéseket,
    majd cosine similarity-vel kiértékeli a top-K találatokat.
    Nem ír a valódi ChromaDB-be, hanem in-memory-ben dolgozik.
    """
    from indexer.embedder import generate_embeddings, generate_single_embedding
    import math

    # Chunk szövegek (summary + content, mint a valódi pipeline-ban)
    texts = []
    for c in chunks:
        if c.summary:
            texts.append(f"{c.summary}\n\n{c.content}")
        else:
            texts.append(c.content)

    if not texts:
        return []

    # Embedding generálás a chunkokhoz
    chunk_embeddings = generate_embeddings(texts)

    results = []
    for query in queries:
        query_embedding = generate_single_embedding(query)

        # Cosine similarity
        scores = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            dot = sum(a * b for a, b in zip(query_embedding, chunk_emb))
            norm_q = math.sqrt(sum(a * a for a in query_embedding))
            norm_c = math.sqrt(sum(a * a for a in chunk_emb))
            sim = dot / (norm_q * norm_c) if norm_q > 0 and norm_c > 0 else 0
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top5 = scores[:5]

        results.append({
            "query": query,
            "top5_similarities": [s[1] for s in top5],
            "avg_top5_sim": statistics.mean([s[1] for s in top5]),
            "top1_sim": top5[0][1] if top5 else 0,
            "top1_chunk_preview": chunks[top5[0][0]].content[:100] if top5 else "",
        })

    return results


def run_benchmark_for_file(
    file_path: Path,
    models: list[str],
    run_search: bool = False,
    test_queries: list[str] = None,
) -> dict:
    """Egy fájl benchmarkja az összes modellel."""
    console.print(f"\n[bold cyan]📄 {file_path.name}[/] ({file_path.stat().st_size / 1024:.1f} KB)")

    # Dokumentum betöltés (egyszer)
    doc = load_document(file_path)
    text_len = len(doc.content)
    word_count = len(doc.content.split())
    console.print(f"   {word_count:,} szó, {text_len:,} karakter\n")

    file_results = {
        "file": str(file_path),
        "filename": file_path.name,
        "text_length": text_len,
        "word_count": word_count,
        "models": {},
    }

    for model in models:
        console.print(f"  [bold]🧠 {model}[/] ... ", end="")

        try:
            start = time.time()
            chunks, perf = chunk_document_with_llm(
                doc.content, str(file_path), model=model
            )
            elapsed = time.time() - start

            metrics = analyze_chunks(chunks)
            fallback_chunks = sum(1 for c in chunks if not c.summary)

            result = {
                "model": model,
                "elapsed_total": elapsed,
                "llm_elapsed": perf.total_elapsed,
                "chunks": metrics,
                "fallback_count": fallback_chunks,
                "fallback_pct": fallback_chunks / max(len(chunks), 1) * 100,
                "perf": {
                    "llm_calls": perf.total_llm_calls,
                    "input_tokens": perf.total_input_tokens,
                    "output_tokens": perf.total_output_tokens,
                    "avg_tok_per_sec": perf.avg_tokens_per_second,
                    "avg_call_duration": perf.avg_call_duration,
                    "cost_usd": perf.total_cost_usd,
                },
                "raw_chunks": [
                    {"summary": c.summary, "content": c.content, "index": c.chunk_index}
                    for c in chunks
                ],
            }

            # Keresési relevancia teszt
            if run_search and chunks:
                console.print("[dim]embedding...[/] ", end="")
                search_results = run_search_relevance_test(
                    chunks, test_queries or DEFAULT_TEST_QUERIES, model
                )
                result["search_relevance"] = search_results
                avg_sim = statistics.mean([r["avg_top5_sim"] for r in search_results])
                result["avg_search_similarity"] = avg_sim

            file_results["models"][model] = result

            speed_str = f"{perf.avg_tokens_per_second:.1f} tok/s" if perf.total_llm_calls > 0 else "N/A"
            console.print(
                f"[green]✅[/] {elapsed:.1f}s | "
                f"{metrics['count']} chunk | "
                f"átl. {metrics['avg_words']:.0f} szó | "
                f"fallback: {result['fallback_pct']:.0f}% | "
                f"{speed_str}"
            )

        except Exception as e:
            console.print(f"[red]❌ Hiba: {e}[/]")
            file_results["models"][model] = {"model": model, "error": str(e)}

    return file_results


def print_comparison_table(all_results: list[dict], models: list[str]):
    """Összehasonlító táblázat megjelenítése."""
    console.print()

    # --- Sebesség táblázat ---
    speed_table = Table(
        title="⏱ Sebesség Összehasonlítás",
        border_style="cyan",
        show_lines=True,
    )
    speed_table.add_column("Fájl", style="cyan", max_width=25)
    for model in models:
        speed_table.add_column(model, justify="center", min_width=18)

    for result in all_results:
        row = [result["filename"]]
        ref_elapsed = None
        for model in models:
            m = result["models"].get(model, {})
            if "error" in m:
                row.append("[red]HIBA[/]")
                continue
            elapsed = m.get("elapsed_total", 0)
            tok_s = m.get("perf", {}).get("avg_tok_per_sec", 0)
            if ref_elapsed is None:
                ref_elapsed = elapsed
                speedup = ""
            else:
                ratio = ref_elapsed / elapsed if elapsed > 0 else 0
                if ratio > 1.1:
                    speedup = f"\n[green]▲ {ratio:.1f}x gyorsabb[/]"
                elif ratio < 0.9:
                    speedup = f"\n[red]▼ {1/ratio:.1f}x lassabb[/]"
                else:
                    speedup = "\n[dim]≈ hasonló[/]"
            row.append(f"{elapsed:.1f}s\n{tok_s:.1f} tok/s{speedup}")
        speed_table.add_row(*row)

    console.print(speed_table)

    # --- Minőség táblázat ---
    quality_table = Table(
        title="📊 Chunkolás Minőség",
        border_style="green",
        show_lines=True,
    )
    quality_table.add_column("Fájl", style="cyan", max_width=25)
    for model in models:
        quality_table.add_column(model, justify="center", min_width=22)

    for result in all_results:
        row = [result["filename"]]
        ref_count = None
        for model in models:
            m = result["models"].get(model, {})
            if "error" in m:
                row.append("[red]HIBA[/]")
                continue
            chunks = m.get("chunks", {})
            fb = m.get("fallback_pct", 0)

            count = chunks.get("count", 0)
            avg_w = chunks.get("avg_words", 0)
            med_w = chunks.get("median_words", 0)
            min_w = chunks.get("min_words", 0)
            max_w = chunks.get("max_words", 0)
            sum_pct = chunks.get("has_summary_pct", 0)

            # Chunk szám összehasonlítás
            if ref_count is None:
                ref_count = count
                diff_str = "[dim](referencia)[/]"
            else:
                diff = count - ref_count
                if diff == 0:
                    diff_str = "[green]= azonos[/]"
                elif abs(diff) <= 2:
                    diff_str = f"[yellow]{diff:+d}[/]"
                else:
                    diff_str = f"[red]{diff:+d}[/]"

            fb_style = "green" if fb < 10 else ("yellow" if fb < 30 else "red")

            row.append(
                f"{count} chunk {diff_str}\n"
                f"átl: {avg_w:.0f} | med: {med_w:.0f}\n"
                f"min: {min_w} | max: {max_w}\n"
                f"összefoglaló: {sum_pct:.0f}%\n"
                f"[{fb_style}]fallback: {fb:.0f}%[/]"
            )
        quality_table.add_row(*row)

    console.print(quality_table)

    # --- Keresési relevancia táblázat ---
    has_search = any(
        "avg_search_similarity" in m
        for r in all_results
        for m in r["models"].values()
        if isinstance(m, dict) and "avg_search_similarity" in m
    )
    if has_search:
        search_table = Table(
            title="🔍 Keresési Relevancia (átl. top-5 cosine hasonlóság)",
            border_style="magenta",
            show_lines=True,
        )
        search_table.add_column("Fájl", style="cyan", max_width=25)
        for model in models:
            search_table.add_column(model, justify="center", min_width=16)

        for result in all_results:
            row = [result["filename"]]
            ref_sim = None
            for model in models:
                m = result["models"].get(model, {})
                if "error" in m or "avg_search_similarity" not in m:
                    row.append("[dim]—[/]")
                    continue
                avg_sim = m["avg_search_similarity"]
                if ref_sim is None:
                    ref_sim = avg_sim
                    diff_str = "[dim](ref)[/]"
                else:
                    diff = avg_sim - ref_sim
                    if abs(diff) < 0.01:
                        diff_str = "[green]≈ azonos[/]"
                    elif diff > 0:
                        diff_str = f"[green]▲ +{diff:.3f}[/]"
                    else:
                        diff_str = f"[red]▼ {diff:.3f}[/]"

                # Per-query részletek
                query_lines = []
                for sr in m.get("search_relevance", []):
                    q_short = sr["query"][:30]
                    query_lines.append(f"  {q_short}: {sr['avg_top5_sim']:.3f}")

                query_detail = "\n".join(query_lines)
                row.append(f"[bold]{avg_sim:.3f}[/] {diff_str}\n{query_detail}")
            search_table.add_row(*row)

        console.print(search_table)


def print_chunk_diff(all_results: list[dict], models: list[str]):
    """Összefoglalók összehasonlítása a modellek között."""
    for result in all_results:
        console.print(f"\n[bold]📝 Összefoglalók: {result['filename']}[/]")
        for model in models:
            m = result["models"].get(model, {})
            if "error" in m:
                continue
            raw = m.get("raw_chunks", [])
            summaries = [c["summary"] for c in raw if c.get("summary")]
            if summaries:
                console.print(f"\n  [bold cyan]{model}[/] ({len(summaries)} összefoglaló):")
                for i, s in enumerate(summaries[:8], 1):
                    console.print(f"    {i}. {s}")
                if len(summaries) > 8:
                    console.print(f"    ... +{len(summaries) - 8} további")


def main():
    parser = argparse.ArgumentParser(
        description="Chunkolási Benchmark — Modellek összehasonlítása"
    )
    parser.add_argument(
        "--file",
        type=str,
        action="append",
        help="Tesztelendő fájl(ok). Többször megadható. Alapértelmezett: data/ összes fájlja.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Tesztelendő modellek, vesszővel elválasztva (alapértelmezett: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--search-test",
        action="store_true",
        help="Keresési relevancia teszt futtatása (lassabb, de informatívabb)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="Egyedi tesztkérdések, pontosvesszővel elválasztva",
    )
    parser.add_argument(
        "--show-summaries",
        action="store_true",
        help="Összefoglalók megjelenítése modellenkénti összehasonlításban",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Eredmények mentése JSON fájlba (benchmark_results/ könyvtárba)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Részletes naplózás",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Modellek
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        console.print("[red]❌ Nincs megadott modell![/]")
        return

    available_models = get_available_models(models)
    if not available_models:
        console.print("[red]❌ Egyetlen modell sem elérhető![/]")
        return

    if len(available_models) < 2:
        console.print("[yellow]⚠ Legalább 2 modell szükséges az összehasonlításhoz.[/]")
        console.print(f"  Elérhető: {', '.join(available_models)}")
        console.print(f"  Telepíts még egyet, pl.: [bold]ollama pull qwen2.5:7b[/]")
        # Azért folytatjuk, hátha így is hasznos

    # Fájlok
    if args.file:
        files = [Path(f).resolve() for f in args.file]
        for f in files:
            if not f.is_file():
                console.print(f"[red]❌ Fájl nem található: {f}[/]")
                return
    else:
        source_dir = config.DATA_DIR
        files = sorted(
            f for f in source_dir.iterdir()
            if f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        )
        if not files:
            console.print(f"[red]❌ Nincs fájl a {source_dir} könyvtárban![/]")
            return

    # Tesztkérdések
    test_queries = DEFAULT_TEST_QUERIES
    if args.queries:
        test_queries = [q.strip() for q in args.queries.split(";") if q.strip()]

    # Header
    console.print(Panel(
        f"[bold]Chunkolási Benchmark[/]\n\n"
        f"Modellek: [cyan]{', '.join(available_models)}[/]\n"
        f"Fájlok: [cyan]{len(files)}[/] db\n"
        f"Keresési teszt: [cyan]{'Igen' if args.search_test else 'Nem'}[/]\n"
        f"Referencia modell: [bold yellow]{available_models[0]}[/]",
        title="🏁 Benchmark Indítás",
        border_style="bright_blue",
    ))

    # Futtatás
    all_results = []
    total_start = time.time()

    for file_path in files:
        result = run_benchmark_for_file(
            file_path,
            available_models,
            run_search=args.search_test,
            test_queries=test_queries,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start

    # Eredmények
    console.print(f"\n{'═' * 60}")
    print_comparison_table(all_results, available_models)

    if args.show_summaries:
        print_chunk_diff(all_results, available_models)

    # Összesített vélemény
    console.print(Panel(
        f"Összes idő: [bold]{total_elapsed:.1f}s[/]\n"
        f"Fájlok: {len(files)} | Modellek: {len(available_models)}",
        title="🏁 Benchmark Kész",
        border_style="green",
    ))

    # JSON mentés
    if args.save:
        save_dir = config.BASE_DIR / "benchmark_results"
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"benchmark_{timestamp}.json"

        # raw_chunks kiírása nélkül (túl nagy lenne)
        save_data = {
            "timestamp": timestamp,
            "models": available_models,
            "files": [],
        }
        for result in all_results:
            file_entry = {
                "filename": result["filename"],
                "text_length": result["text_length"],
                "word_count": result["word_count"],
                "models": {},
            }
            for model, m in result["models"].items():
                if "error" in m:
                    file_entry["models"][model] = {"error": m["error"]}
                else:
                    entry = {k: v for k, v in m.items() if k != "raw_chunks"}
                    file_entry["models"][model] = entry
            save_data["files"].append(file_entry)

        save_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[green]💾 Eredmények mentve: {save_path}[/]")

    # .gitignore frissítés figyelmeztetés
    console.print(
        "\n[dim]Tipp: add hozzá a 'benchmark_results/' sort a .gitignore-hoz, ha még nincs benne.[/]"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print("\n[yellow]👋 Benchmark megszakítva.[/]\n")
