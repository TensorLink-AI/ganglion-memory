"""SQLite backend for memory storage.

One table. One schema. Embedding vectors stored as binary blobs.
Cosine similarity computed in Python (no vector extensions needed).

    backend = SqliteBackend("memory.db")
    eid = await backend.store(experience)
    exp = await backend.get(eid)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any

from ganglion.memory.types import Experience

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL DEFAULT '',
    tags TEXT DEFAULT '[]',
    source TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    confirmation_count INTEGER NOT NULL DEFAULT 0,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    embedding BLOB,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_exp_source ON experiences(source);
CREATE INDEX IF NOT EXISTS idx_exp_updated ON experiences(updated_at);
"""


class SqliteBackend:
    """Single-table SQLite storage for experiences with embedding support.

    Thread-safe for single-writer via WAL mode.
    All async methods delegate to threads to avoid blocking the event loop.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # -- Encoding helpers ----------------------------------------------------

    @staticmethod
    def _encode_embedding(embedding: list[float] | None) -> bytes | None:
        """Pack a float list into a binary blob."""
        if embedding is None:
            return None
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _decode_embedding(blob: bytes | None) -> list[float] | None:
        """Unpack a binary blob into a float list."""
        if blob is None:
            return None
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))

    # -- Write operations ----------------------------------------------------

    def _store_sync(self, exp: Experience) -> int:
        cursor = self._conn.execute(
            """INSERT INTO experiences
               (content, tags, source, created_at, updated_at,
                confirmation_count, contradiction_count, embedding, metadata)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                exp.content,
                json.dumps(list(exp.tags)),
                exp.source,
                exp.created_at.isoformat(),
                exp.updated_at.isoformat(),
                exp.confirmation_count,
                exp.contradiction_count,
                self._encode_embedding(exp.embedding),
                json.dumps(exp.metadata) if exp.metadata else None,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    async def store(self, experience: Experience) -> int:
        """Persist a new experience. Sets experience.id and returns it."""
        eid = await asyncio.to_thread(self._store_sync, experience)
        experience.id = eid
        return eid

    def _update_sync(self, exp: Experience) -> None:
        if exp.id is None:
            raise ValueError("Cannot update experience without an id")
        self._conn.execute(
            """UPDATE experiences SET
                content=?, tags=?, source=?, created_at=?, updated_at=?,
                confirmation_count=?, contradiction_count=?,
                embedding=?, metadata=?
               WHERE id=?""",
            (
                exp.content,
                json.dumps(list(exp.tags)),
                exp.source,
                exp.created_at.isoformat(),
                exp.updated_at.isoformat(),
                exp.confirmation_count,
                exp.contradiction_count,
                self._encode_embedding(exp.embedding),
                json.dumps(exp.metadata) if exp.metadata else None,
                exp.id,
            ),
        )
        self._conn.commit()

    async def update(self, experience: Experience) -> None:
        """Update an existing experience in place."""
        await asyncio.to_thread(self._update_sync, experience)

    def _delete_sync(self, experience_id: int) -> None:
        self._conn.execute("DELETE FROM experiences WHERE id=?", (experience_id,))
        self._conn.commit()

    async def delete(self, experience_id: int) -> None:
        """Delete an experience by ID. No-op if not found."""
        await asyncio.to_thread(self._delete_sync, experience_id)

    # -- Read operations -----------------------------------------------------

    def _get_sync(self, experience_id: int) -> Experience | None:
        row = self._conn.execute(
            "SELECT * FROM experiences WHERE id=?", (experience_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experience(row)

    async def get(self, experience_id: int) -> Experience | None:
        """Get a single experience by ID. Returns None if not found."""
        return await asyncio.to_thread(self._get_sync, experience_id)

    def _search_sync(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        tags: tuple[str, ...],
    ) -> list[tuple[Experience, float]]:
        """Brute-force cosine search over all experiences with matching tags."""
        conditions: list[str] = []
        params: list[Any] = []
        for tag in tags:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"SELECT * FROM experiences {where}", params
        ).fetchall()

        scored: list[tuple[Experience, float]] = []
        for row in rows:
            row_embedding = self._decode_embedding(row["embedding"])
            if row_embedding is None:
                continue
            score = _cosine_similarity(embedding, row_embedding)
            if score >= threshold:
                scored.append((self._row_to_experience(row), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.3,
        tags: tuple[str, ...] = (),
    ) -> list[tuple[Experience, float]]:
        """Find experiences similar to the embedding vector."""
        return await asyncio.to_thread(
            self._search_sync, embedding, limit, threshold, tags,
        )

    def _query_sync(
        self,
        tags: tuple[str, ...],
        source: str | None,
        limit: int,
    ) -> list[Experience]:
        conditions: list[str] = []
        params: list[Any] = []

        for tag in tags:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        if source is not None:
            conditions.append("source = ?")
            params.append(source)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM experiences {where} ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_experience(row) for row in rows]

    async def query(
        self,
        tags: tuple[str, ...] = (),
        source: str | None = None,
        limit: int = 20,
    ) -> list[Experience]:
        """Retrieve experiences matching filters, ordered by updated_at desc."""
        return await asyncio.to_thread(self._query_sync, tags, source, limit)

    def _all_sync(self) -> list[Experience]:
        rows = self._conn.execute("SELECT * FROM experiences").fetchall()
        return [self._row_to_experience(row) for row in rows]

    async def all(self) -> list[Experience]:
        """Return every experience."""
        return await asyncio.to_thread(self._all_sync)

    def _count_sync(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()
        return row[0]

    async def count(self) -> int:
        """Return total number of stored experiences."""
        return await asyncio.to_thread(self._count_sync)

    # -- Row mapping ---------------------------------------------------------

    def _row_to_experience(self, row: sqlite3.Row) -> Experience:
        from datetime import UTC, datetime

        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return datetime.now(UTC)

        return Experience(
            id=row["id"],
            content=row["content"],
            tags=tuple(json.loads(row["tags"] or "[]")),
            source=row["source"] or "",
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            confirmation_count=row["confirmation_count"],
            contradiction_count=row["contradiction_count"],
            embedding=self._decode_embedding(row["embedding"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )
