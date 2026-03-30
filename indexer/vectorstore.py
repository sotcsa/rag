"""
ChromaDB vektor adatbázis kezelés.
Persistent storage — az adatok a chroma_db/ könyvtárban maradnak,
és bármikor átvihetők másik gépre.
"""

import logging
from pathlib import Path

import chromadb

import config

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB persistent vektor adatbázis."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or config.CHROMA_DB_DIR)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Cosine hasonlóság
        )
        logger.info(
            "ChromaDB inicializálva: %s (meglévő elemek: %d)",
            self.db_path,
            self.collection.count(),
        )

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ):
        """
        Chunkok hozzáadása a vektor adatbázishoz.

        Args:
            ids: Egyedi azonosítók (pl. "fájlnév_chunk_0")
            documents: Chunk szövegek
            embeddings: Embedding vektorok
            metadatas: Metaadatok (fájlnév, összefoglaló, stb.)
        """
        if not ids:
            return

        # ChromaDB batch limit: egyszerre max ~5000 elemet adunk hozzá
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info("%d chunk hozzáadva a vektor adatbázishoz", len(ids))

    def delete_by_source(self, source_file: str):
        """
        Adott fájlhoz tartozó összes chunk törlése.
        Fájl módosítás utáni újraindexeléshez.
        """
        try:
            # Lekérdezzük az adott fájlhoz tartozó ID-kat
            results = self.collection.get(
                where={"source": source_file},
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    "%d chunk törölve a forrásból: %s",
                    len(results["ids"]),
                    source_file,
                )
        except Exception as e:
            logger.warning("Törlés hiba (%s): %s", source_file, e)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = None,
    ) -> list[dict]:
        """
        Vektor keresés a legközelebbi chunkok lekérdezéséhez.

        Args:
            query_embedding: A kérdés embedding vektora
            top_k: Hány eredményt adjon vissza

        Returns:
            Találatok listája (document, metadata, distance)
        """
        top_k = top_k or config.SEARCH_TOP_K

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # ChromaDB cosine distance: 0 = tökéletes egyezés, 2 = teljesen különböző
                # Átalakítjuk similarity-re: 1 - (distance / 2)
                similarity = 1 - (distance / 2)

                if similarity >= config.SEARCH_MIN_SIMILARITY:
                    hits.append(
                        {
                            "id": doc_id,
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": distance,
                            "similarity": similarity,
                        }
                    )

        return hits

    def get_count(self) -> int:
        """Tárolt chunkok száma."""
        return self.collection.count()

    def get_sources(self) -> list[str]:
        """Összes indexelt forrás fájl listája."""
        try:
            results = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in results["metadatas"]:
                if "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            return []
