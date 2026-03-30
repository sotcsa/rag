"""
Vektor-alapú keresés a ChromaDB-ben.
"""

import logging

from indexer.embedder import generate_single_embedding
from indexer.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Dokumentum chunk-ok keresése szemantikus hasonlóság alapján."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Kérdés alapú keresés.

        1. A kérdésből embedding vektor generálása (bge-m3)
        2. A ChromaDB-ben a legközelebbi vektorok keresése
        3. Minimum similarity szűrés

        Args:
            query: Felhasználói kérdés (magyarul)
            top_k: Hány eredményt adjon vissza

        Returns:
            Releváns chunk-ok listája, csökkenő relevancia sorrendben
        """
        logger.info("Keresés: '%s'", query[:80])

        # Embedding generálás a kérdésből
        query_embedding = generate_single_embedding(query)

        # Vektor keresés
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        logger.info("Találatok: %d", len(results))
        return results

    def get_context_for_query(self, query: str, top_k: int = None) -> str:
        """
        Kérdésből kontextus szöveg összeállítása a válaszgeneráláshoz.
        A talált chunk-okat szöveges formában adja vissza, forrás hivatkozásokkal.
        """
        results = self.search(query, top_k)

        if not results:
            return ""

        context_parts = []
        for i, hit in enumerate(results, 1):
            source = hit["metadata"].get("filename", "ismeretlen")
            summary = hit["metadata"].get("summary", "")
            similarity_pct = f"{hit['similarity']:.0%}"

            header = f"[Forrás {i}: {source} (relevancia: {similarity_pct})]"
            if summary:
                header += f"\nÖsszefoglaló: {summary}"

            context_parts.append(f"{header}\n{hit['document']}")

        return "\n\n---\n\n".join(context_parts)
