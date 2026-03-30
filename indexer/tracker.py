"""
Feldolgozási nyilvántartás SQLite-tel.
Nyomon követi, mely fájlok lettek már indexelve, és szükség van-e újraindexelésre.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config


def _get_file_hash(file_path: Path) -> str:
    """Fájl SHA-256 hash kiszámítása."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


class Tracker:
    """SQLite-alapú fájl feldolgozási nyilvántartás."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or config.TRACKING_DB_PATH
        self._init_db()

    def _init_db(self):
        """Adatbázis és tábla létrehozása, ha nem létezik."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER,
                    chunk_count INTEGER DEFAULT 0,
                    indexed_at TEXT,
                    status TEXT DEFAULT 'ok'
                )
            """)
            conn.commit()

    def get_unprocessed_files(self, source_dir: Path) -> list[Path]:
        """
        Visszaadja azokat a fájlokat, amelyek:
        - Még nem lettek feldolgozva, VAGY
        - Módosultak a legutóbbi feldolgozás óta (hash eltér)
        """
        source_dir = Path(source_dir)
        unprocessed = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for file_path in source_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                    cursor.execute(
                        "SELECT file_hash FROM processed_files WHERE file_path = ?",
                        (str(file_path),),
                    )
                    row = cursor.fetchone()

                    if row is None:
                        # Új fájl — még nincs az adatbázisban
                        unprocessed.append(file_path)
                    else:
                        # Létező fájl — ellenőrizzük, hogy változott-e
                        current_hash = _get_file_hash(file_path)
                        if current_hash != row[0]:
                            unprocessed.append(file_path)

        return sorted(unprocessed)

    def mark_processed(
        self, file_path: Path, chunk_count: int, status: str = "ok"
    ):
        """Fájl megjelölése feldolgozottként."""
        file_path = Path(file_path)
        file_hash = _get_file_hash(file_path)
        file_size = file_path.stat().st_size

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_files
                    (file_path, file_hash, file_size, chunk_count, indexed_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(file_path),
                    file_hash,
                    file_size,
                    chunk_count,
                    datetime.now(timezone.utc).isoformat(),
                    status,
                ),
            )
            conn.commit()

    def mark_error(self, file_path: Path, error_msg: str = ""):
        """Fájl megjelölése hibásként."""
        file_path = Path(file_path)
        try:
            file_hash = _get_file_hash(file_path)
            file_size = file_path.stat().st_size
        except OSError:
            file_hash = "error"
            file_size = 0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_files
                    (file_path, file_hash, file_size, chunk_count, indexed_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(file_path),
                    file_hash,
                    file_size,
                    0,
                    datetime.now(timezone.utc).isoformat(),
                    f"error: {error_msg}" if error_msg else "error",
                ),
            )
            conn.commit()

    def remove_file(self, file_path: Path):
        """Fájl eltávolítása a nyilvántartásból."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM processed_files WHERE file_path = ?",
                (str(file_path),),
            )
            conn.commit()

    def get_all_processed(self) -> list[dict]:
        """Összes feldolgozott fájl lekérdezése."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM processed_files ORDER BY indexed_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Statisztikák lekérdezése."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE status = 'ok'")
            ok_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE status LIKE 'error%'")
            error_count = cursor.fetchone()[0]
            cursor.execute("SELECT SUM(chunk_count) FROM processed_files WHERE status = 'ok'")
            total_chunks = cursor.fetchone()[0] or 0
            cursor.execute("SELECT SUM(file_size) FROM processed_files WHERE status = 'ok'")
            total_size = cursor.fetchone()[0] or 0

        return {
            "processed_files": ok_count,
            "error_files": error_count,
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
        }

    def clear(self):
        """Teljes nyilvántartás törlése."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM processed_files")
            conn.commit()
