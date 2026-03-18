"""SQLite backend for memory storage.

One table. One schema. Replaces the old patterns + antipatterns +
agent_designs triple with a single beliefs table where the valence
column distinguishes positive/negative/neutral.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from ganglion.memory.types import Belief, Observation, Valence

logger = logging.getLogger(__name__)

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
    tags TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_beliefs_capability ON beliefs(capability);
CREATE INDEX IF NOT EXISTS idx_beliefs_valence ON beliefs(valence);
CREATE INDEX IF NOT EXISTS idx_beliefs_source ON beliefs(source);
CREATE INDEX IF NOT EXISTS idx_beliefs_last_confirmed ON beliefs(last_confirmed);
"""


class SqliteMemoryBackend:
    """Single-table SQLite storage for beliefs."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # -- Write operations --------------------------------------------------

    async def store(self, belief: Belief) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO beliefs
                   (capability, description, valence, confidence, confirmation_count,
                    entities, config, metric_name, metric_value, last_metric_value,
                    source, first_seen, last_confirmed, last_retrieved,
                    superseded_by, tags)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                ),
            )
            belief.id = cursor.lastrowid

    async def update(self, belief: Belief) -> None:
        if belief.id is None:
            raise ValueError("Cannot update a belief without an id")
        with self._connect() as conn:
            conn.execute(
                """UPDATE beliefs SET
                    capability=?, description=?, valence=?, confidence=?,
                    confirmation_count=?, entities=?, config=?,
                    metric_name=?, metric_value=?, last_metric_value=?,
                    source=?, first_seen=?, last_confirmed=?, last_retrieved=?,
                    superseded_by=?, tags=?
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
                    belief.id,
                ),
            )

    async def remove(self, belief: Belief) -> None:
        if belief.id is None:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM beliefs WHERE id=?", (belief.id,))

    # -- Read operations ---------------------------------------------------

    async def find_similar(
        self,
        observation: Observation,
        threshold: float = 0.85,
    ) -> Belief | None:
        """Find a belief matching this observation.

        Default: exact match on (capability, description[:200]).
        For fuzzy/embedding similarity, subclass and override this.
        """
        desc_prefix = observation.description[:200]
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE capability=? AND description LIKE ? LIMIT 1",
                (observation.capability, f"{desc_prefix}%"),
            ).fetchone()
        return self._row_to_belief(row) if row else None

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
        # Entity filtering: check if any requested entity appears in the JSON array
        for entity in entities:
            conditions.append("entities LIKE ?")
            params.append(f'%"{entity}"%')
        # Tag filtering
        for tag in tags:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM beliefs {where} ORDER BY last_confirmed DESC LIMIT ?"
        params.append(limit * 2)  # over-fetch for strength filtering

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        beliefs = [self._row_to_belief(row) for row in rows]

        # Filter by strength in Python (can't compute in SQL without the formula)
        if min_strength > 0:
            beliefs = [b for b in beliefs if b.strength >= min_strength]

        return beliefs[:limit]

    async def all_beliefs(self) -> list[Belief]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM beliefs").fetchall()
        return [self._row_to_belief(row) for row in rows]

    # -- Row mapping -------------------------------------------------------

    @staticmethod
    def _row_to_belief(row: sqlite3.Row) -> Belief:
        from datetime import datetime

        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return datetime.now(UTC)

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
        )
