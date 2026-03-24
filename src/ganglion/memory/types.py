"""
Core type for Ganglion Memory v4.

One type replaces three:
    Experience — replaces Observation, Belief, and Delta.

No valence, no confidence scoring, no biological metaphors.
Just content, tags, counts, and an optional embedding.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Experience:
    """The universal memory unit.

    Everything stored in memory is an Experience. No patterns, no
    antipatterns, no beliefs, no observations, no deltas. Just
    experiences with raw confirmation/contradiction counts.

    Attributes:
        id:                   Database-assigned identifier.
        content:              The memory text — what happened or was learned.
        tags:                 Searchable labels (capability, outcome, domain, ...).
        source:               Who created this (bot_id, user, system, ...).
        created_at:           When first stored.
        updated_at:           When last modified (confirm/contradict/rewrite).
        confirmation_count:   How many times this was confirmed or re-observed.
        contradiction_count:  How many times this was contradicted.
        embedding:            Optional vector for semantic search.
        metadata:             Arbitrary key/value data (input_text, output_text,
                              metric_name, metric_value, config, ...).
    """

    id: int | None = None
    content: str = ""
    tags: tuple[str, ...] = ()
    source: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    confirmation_count: int = 0
    contradiction_count: int = 0
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None

    @property
    def net_score(self) -> int:
        """Confirmations minus contradictions. Positive = trustworthy."""
        return self.confirmation_count - self.contradiction_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict. Embedding stored as base64 float blob."""
        d: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "tags": list(self.tags),
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confirmation_count": self.confirmation_count,
            "contradiction_count": self.contradiction_count,
            "metadata": self.metadata,
        }
        if self.embedding is not None:
            d["embedding"] = base64.b64encode(
                struct.pack(f"{len(self.embedding)}f", *self.embedding)
            ).decode("ascii")
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experience:
        """Deserialize from a plain dict."""

        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            if isinstance(val, datetime):
                return val
            return datetime.now(UTC)

        embedding = None
        if data.get("embedding"):
            raw = base64.b64decode(data["embedding"])
            count = len(raw) // 4
            embedding = list(struct.unpack(f"{count}f", raw))

        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            tags=tuple(data.get("tags", ())),
            source=data.get("source", ""),
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            confirmation_count=data.get("confirmation_count", 0),
            contradiction_count=data.get("contradiction_count", 0),
            embedding=embedding,
            metadata=data.get("metadata"),
        )
