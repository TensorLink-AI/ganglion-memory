"""SQLite backend for memory storage.

One table. One schema. Embedding vectors stored as binary blobs.
Cosine similarity used when embeddings available, Jaccard fallback otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import struct
import math
import re
from pathlib import Path
from typing import Any

from ganglion.memory.types import Belief, Observation, Valence

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(re.findall(r'[a-z0-9]+', a.lower()))
    tokens_b = set(re.findall(r'[a-z0-9]+', b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capability TEXT NOT NULL,
    description TEXT NOT NULL,
    valence TEXT NOT NULL DEFAULT 'neutral',
    confidence REAL NOT NULL DEFAULT 1.0,
    confirmation_count INTEGER NOT NULL DEFAULT 1,
    entities TEXT DEFAULT '[]',
    config TEXT,
    metric_name TEXT,
    metric_value REAL,
    last_metric_value REAL,
    source TEXT,
    first_seen TEXT NOT NULL,
    last_confirmed TEXT NOT NULL,
    last_retrieved TEXT,
    superseded_by TEXT,
    tags TEXT DEFAULT '[]',
    embedding BLOB,
    produced_with TEXT DEFAULT '[]',
    input_context TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_beliefs_capability ON beliefs(capability);
CREATE INDEX IF NOT EXISTS idx_beliefs_valence ON beliefs(valence);
CREATE INDEX IF NOT EXISTS idx_beliefs_source ON beliefs(source);
CREATE INDEX IF NOT EXISTS idx_beliefs_last_confirmed ON beliefs(last_confirmed);
"""

_MIGRATION_ADD_EMBEDDING = "ALTER TABLE beliefs ADD COLUMN embedding BLOB"
_MIGRATION_ADD_PRODUCED_WITH = "ALTER TABLE beliefs ADD COLUMN produced_with TEXT DEFAULT '[]'"
_MIGRATION_ADD_INPUT_CONTEXT = "ALTER TABLE beliefs ADD COLUMN input_context TEXT DEFAULT ''"


class SqliteMemoryBackend:
    """Single-table SQLite storage for beliefs with embedding support."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Migrate: add embedding column if missing
        try:
            self._conn.execute("SELECT embedding FROM beliefs LIMIT 0")
        except sqlite3.OperationalError:
            try:
                self._conn.execute(_MIGRATION_ADD_EMBEDDING)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass
        # Migrate: add produced_with and input_context columns if missing
        for migration in [_MIGRATION_ADD_PRODUCED_WITH, _MIGRATION_ADD_INPUT_CONTEXT]:
            try:
                self._conn.execute(migration)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # -- Write operations --------------------------------------------------

    @staticmethod
    def _encode_embedding(embedding: list[float] | None) -> bytes | None:
        if embedding is None:
            return None
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _decode_embedding(blob: bytes | None) -> list[float] | None:
        if blob is None:
            return None
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))

    def _store_sync(self, belief: Belief) -> int:
        cursor = self._conn.execute(
            """INSERT INTO beliefs
               (capability, description, valence, confidence, confirmation_count,
                entities, config, metric_name, metric_value, last_metric_value,
                source, first_seen, last_confirmed, last_retrieved,
                superseded_by, tags, embedding, produced_with, input_context)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                belief.capability,
                belief.description,
                belief.valence.value,
                belief.confidence,
                belief.confirmation_count,
                json.dumps(list(belief.entities)),
                json.dumps(belief.config) if belief.config else None,
                belief.metric_name,
                belief.metric_value,
                belief.last_metric_value,
                belief.source,
                belief.first_seen.isoformat(),
                belief.last_confirmed.isoformat(),
                belief.last_retrieved.isoformat() if belief.last_retrieved else None,
                belief.superseded_by,
                json.dumps(list(belief.tags)),
                self._encode_embedding(belief.embedding),
                json.dumps(list(belief.produced_with)),
                belief.input_context,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    async def store(self, belief: Belief) -> None:
        belief.id = await asyncio.to_thread(self._store_sync, belief)

    def _update_sync(self, belief: Belief) -> None:
        if belief.id is None:
            raise ValueError("Cannot update a belief without an id")
        self._conn.execute(
            """UPDATE beliefs SET
                capability=?, description=?, valence=?, confidence=?,
                confirmation_count=?, entities=?, config=?,
                metric_name=?, metric_value=?, last_metric_value=?,
                source=?, first_seen=?, last_confirmed=?, last_retrieved=?,
                superseded_by=?, tags=?, embedding=?, produced_with=?,
                input_context=?
               WHERE id=?""",
            (
                belief.capability,
                belief.description,
                belief.valence.value,
                belief.confidence,
                belief.confirmation_count,
                json.dumps(list(belief.entities)),
                json.dumps(belief.config) if belief.config else None,
                belief.metric_name,
                belief.metric_value,
                belief.last_metric_value,
                belief.source,
                belief.first_seen.isoformat(),
                belief.last_confirmed.isoformat(),
                belief.last_retrieved.isoformat() if belief.last_retrieved else None,
                belief.superseded_by,
                json.dumps(list(belief.tags)),
                self._encode_embedding(belief.embedding),
                json.dumps(list(belief.produced_with)),
                belief.input_context,
                belief.id,
            ),
        )
        self._conn.commit()

    async def update(self, belief: Belief) -> None:
        await asyncio.to_thread(self._update_sync, belief)

    def _remove_sync(self, belief: Belief) -> None:
        if belief.id is None:
            return
        self._conn.execute("DELETE FROM beliefs WHERE id=?", (belief.id,))
        self._conn.commit()

    async def remove(self, belief: Belief) -> None:
        await asyncio.to_thread(self._remove_sync, belief)

    # -- Read operations ---------------------------------------------------

    def _find_similar_sync(
        self,
        observation: Observation,
        threshold: float = 0.75,
        embedding: list[float] | None = None,
    ) -> Belief | None:
        """Find the belief most similar to this observation.

        Uses cosine similarity on embeddings when available,
        falls back to Jaccard on description text.
        """
        rows = self._conn.execute(
            "SELECT * FROM beliefs WHERE capability = ? AND superseded_by IS NULL",
            (observation.capability,),
        ).fetchall()

        best_match: Belief | None = None
        best_score = 0.0

        if embedding is not None:
            for row in rows:
                belief_embedding = self._decode_embedding(row["embedding"])
                if belief_embedding is not None:
                    score = _cosine_similarity(embedding, belief_embedding)
                    if score >= threshold and score > best_score:
                        best_score = score
                        best_match = self._row_to_belief(row)
            if best_match is not None:
                return best_match

        # Fallback to Jaccard
        jaccard_threshold = max(threshold, 0.85) if embedding is not None else threshold
        for row in rows:
            score = _jaccard_similarity(observation.description, row["description"])
            if score >= jaccard_threshold and score > best_score:
                best_score = score
                best_match = self._row_to_belief(row)

        return best_match

    async def find_similar(
        self,
        observation: Observation,
        threshold: float = 0.75,
        embedding: list[float] | None = None,
    ) -> Belief | None:
        return await asyncio.to_thread(
            self._find_similar_sync, observation, threshold, embedding,
        )

    def _query_sync(
        self,
        capability: str | None = None,
        valence: Valence | None = None,
        entities: tuple[str, ...] = (),
        exclude_source: str | None = None,
        tags: tuple[str, ...] = (),
        min_strength: float = 0.0,
        limit: int = 20,
    ) -> list[Belief]:
        conditions: list[str] = []
        params: list[Any] = []

        if capability:
            conditions.append("capability = ?")
            params.append(capability)
        if valence:
            conditions.append("valence = ?")
            params.append(valence.value)
        if exclude_source:
            conditions.append("(source IS NULL OR source != ?)")
            params.append(exclude_source)
        for entity in entities:
            conditions.append("entities LIKE ?")
            params.append(f'%"{entity}"%')
        for tag in tags:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM beliefs {where} ORDER BY last_confirmed DESC LIMIT ?"
        params.append(limit * 2)

        rows = self._conn.execute(sql, params).fetchall()
        beliefs = [self._row_to_belief(row) for row in rows]

        if min_strength > 0:
            beliefs = [b for b in beliefs if b.strength >= min_strength]

        return beliefs[:limit]

    async def query(
        self,
        capability: str | None = None,
        valence: Valence | None = None,
        entities: tuple[str, ...] = (),
        exclude_source: str | None = None,
        tags: tuple[str, ...] = (),
        min_strength: float = 0.0,
        limit: int = 20,
    ) -> list[Belief]:
        return await asyncio.to_thread(
            self._query_sync, capability, valence, entities,
            exclude_source, tags, min_strength, limit,
        )

    def _all_beliefs_sync(self) -> list[Belief]:
        rows = self._conn.execute("SELECT * FROM beliefs").fetchall()
        return [self._row_to_belief(row) for row in rows]

    async def all_beliefs(self) -> list[Belief]:
        return await asyncio.to_thread(self._all_beliefs_sync)

    # -- Row mapping -------------------------------------------------------

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        from datetime import UTC, datetime

        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return datetime.now(UTC)

        embedding = None
        try:
            embedding = self._decode_embedding(row["embedding"])
        except (KeyError, IndexError):
            pass

        return Belief(
            id=row["id"],
            capability=row["capability"],
            description=row["description"],
            valence=Valence(row["valence"]),
            confidence=row["confidence"],
            confirmation_count=row["confirmation_count"],
            entities=tuple(json.loads(row["entities"] or "[]")),
            config=json.loads(row["config"]) if row["config"] else None,
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            last_metric_value=row["last_metric_value"],
            source=row["source"],
            first_seen=_parse_dt(row["first_seen"]),
            last_confirmed=_parse_dt(row["last_confirmed"]),
            last_retrieved=_parse_dt(row["last_retrieved"]) if row["last_retrieved"] else None,
            superseded_by=row["superseded_by"],
            tags=tuple(json.loads(row["tags"] or "[]")),
            embedding=embedding,
            produced_with=tuple(json.loads(row["produced_with"] or "[]")),
            input_context=row["input_context"] or "",
        )
