"""
CLI felület a kereséshez.
Interaktív chat mód és egyszeri kérdés mód.
"""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from indexer.vectorstore import VectorStore
from search.retriever import Retriever
from search.generator import generate_answer

console = Console()


def print_answer_streaming(query: str, retriever: Retriever):
    """Streaming válasz kiírása: a tokenek azonnal megjelennek."""
    # Kontextus keresés
    with console.status("[bold cyan]🔍 Keresés a dokumentumokban...[/]"):
        context = retriever.get_context_for_query(query)

    if not context:
        console.print(
            "\n[yellow]⚠ Nem találtam releváns dokumentumot ehhez a kérdéshez.[/]\n"
        )
        return

    # Források kiírása
    results = retriever.search(query)
    if results:
        console.print("\n[dim]📚 Források:[/]")
        for i, hit in enumerate(results, 1):
            filename = hit["metadata"].get("filename", "?")
            sim = hit["similarity"]
            console.print(f"  [dim]{i}. {filename} ({sim:.0%})[/]")
        console.print()

    # Streaming válasz
    console.print("[bold green]💬 Válasz:[/]\n")
    full_response = ""
    for token in generate_answer(query, context, stream=True):
        print(token, end="", flush=True)
        full_response += token
    print("\n")


def single_query(query: str):
    """Egyszeri kérdés megválaszolása."""
    retriever = Retriever()
    print_answer_streaming(query, retriever)


def interactive_chat():
    """Interaktív chat mód."""
    console.print(
        Panel(
            "[bold]🤖 RAG Chat — Lokális Tudásbázis Keresés[/]\n\n"
            "Kérdezz bármit a dokumentumaidról magyarul!\n"
            "Kilépés: [bold]quit[/], [bold]exit[/], vagy [bold]Ctrl+C[/]",
            border_style="cyan",
        )
    )

    retriever = Retriever()

    while True:
        try:
            console.print()
            query = console.input("[bold cyan]❓ Kérdés: [/]").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "kilépés", "q"):
                console.print("\n[dim]👋 Viszlát![/]\n")
                break

            print_answer_streaming(query, retriever)

        except KeyboardInterrupt:
            console.print("\n\n[dim]👋 Viszlát![/]\n")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Hiba: {e}[/]\n")


def list_sources():
    """Indexelt források listázása."""
    store = VectorStore()
    sources = store.get_sources()
    total_chunks = store.get_count()

    if not sources:
        console.print("[yellow]Nincs indexelt dokumentum.[/]")
        console.print("Futtasd: [bold]uv run python index.py[/]")
        return

    table = Table(title="📚 Indexelt Dokumentumok")
    table.add_column("Forrás", style="cyan")

    for source in sources:
        table.add_column(source)
        table.add_row(source)

    # Egyszerűbb megoldás: soronként kiírás
    console.print(f"\n[bold]📚 Indexelt dokumentumok[/] ({total_chunks} chunk összesen)\n")
    for i, source in enumerate(sources, 1):
        console.print(f"  {i}. [cyan]{source}[/]")
    console.print()
