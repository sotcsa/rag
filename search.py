"""
RAG Keresés — Főprogram

Használat:
    uv run python search.py "Mi az a RAG?"           # Egyszeri kérdés
    uv run python search.py --chat                    # Interaktív chat
    uv run python search.py --list-sources            # Indexelt források listázása
"""

import argparse
import sys

from rich.console import Console

from search.cli import single_query, interactive_chat, list_sources

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="RAG Keresés — Kérdezd a dokumentumaidat"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Kérdés (ha nincs megadva, interaktív chat indul)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interaktív chat mód",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Indexelt források listázása",
    )
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
    elif args.chat:
        interactive_chat()
    elif args.query:
        single_query(args.query)
    else:
        # Ha nincs argumentum, interaktív chat indul
        interactive_chat()


if __name__ == "__main__":
    main()
