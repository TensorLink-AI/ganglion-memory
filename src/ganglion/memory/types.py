"""
Core types for Ganglion's memory system.

Three types replace seven:
    Observation — everything that enters the system
    Belief      — everything the system remembers
    Delta       — everything the system notices changing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class Valence(str, Enum):
    """Did this work or fail?"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class Observation:
    """The universal input. Everything enters memory as this.

    A mining run, a config change, a subnet shift, an agent design —
    all are observations. Replaces the old Pattern/Antipattern/
    AgentDesignPattern split.
    """

    capability: str
    description: str
    valence: Valence
    entities: tuple[str, ...] = ()
    config: dict[str, Any] | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    source: str | None = None
    run_id: str | None = None
    tags: tuple[str, ...] = ()
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "description": self.description,
            "valence": self.valence.value,
            "entities": list(self.entities),
            "config": self.config,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "source": self.source,
            "run_id": self.run_id,
            "tags": list(self.tags),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class Belief:
    """The universal storage unit.

    A pattern is a belief with positive valence.
    An antipattern is a belief with negative valence.
    An agent design is a belief tagged "agent_design".

    No separate types, no separate tables, no separate queries.
    """

    id: int | None = None
    capability: str = ""
    description: str = ""
    valence: Valence = Valence.NEUTRAL
    confidence: float = 1.0
    confirmation_count: int = 1
    entities: tuple[str, ...] = ()
    config: dict[str, Any] | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    last_metric_value: float | None = None
    source: str | None = None
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_confirmed: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_retrieved: datetime | None = None
    superseded_by: str | None = None
    tags: tuple[str, ...] = ()

    @property
    def is_pattern(self) -> bool:
        return self.valence == Valence.POSITIVE

    @property
    def is_antipattern(self) -> bool:
        return self.valence == Valence.NEGATIVE

    @property
    def strength(self) -> float:
        """Composite score: confirmation × recency × confidence.

        Used for ranking (best beliefs surface first in prompts)
        and eviction (weakest beliefs get forgotten first).
        """
        recency_hours = (datetime.now(UTC) - self.last_confirmed).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + recency_hours / 168.0)
        return self.confidence * self.confirmation_count * recency_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "capability": self.capability,
            "description": self.description,
            "valence": self.valence.value,
            "confidence": self.confidence,
            "confirmation_count": self.confirmation_count,
            "entities": list(self.entities),
            "config": self.config,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "last_metric_value": self.last_metric_value,
            "source": self.source,
            "first_seen": self.first_seen.isoformat(),
            "last_confirmed": self.last_confirmed.isoformat(),
            "last_retrieved": self.last_retrieved.isoformat() if self.last_retrieved else None,
            "superseded_by": self.superseded_by,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Belief:
        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            if isinstance(val, datetime):
                return val
            return datetime.now(UTC)

        return cls(
            id=data.get("id"),
            capability=data.get("capability", ""),
            description=data.get("description", ""),
            valence=Valence(data.get("valence", "neutral")),
            confidence=data.get("confidence", 1.0),
            confirmation_count=data.get("confirmation_count", 1),
            entities=tuple(data.get("entities", ())),
            config=data.get("config"),
            metric_name=data.get("metric_name"),
            metric_value=data.get("metric_value"),
            last_metric_value=data.get("last_metric_value"),
            source=data.get("source"),
            first_seen=_parse_dt(data.get("first_seen")),
            last_confirmed=_parse_dt(data.get("last_confirmed")),
            last_retrieved=_parse_dt(data["last_retrieved"]) if data.get("last_retrieved") else None,
            superseded_by=data.get("superseded_by"),
            tags=tuple(data.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class Delta:
    """Emitted when the system detects a meaningful change.

    The prediction-error signal. Consumers can log it, alert on it,
    or feed it into the next run's planning phase.
    """

    old_belief: Belief
    new_observation: Observation
    delta_type: str  # "metric_shift", "contradiction"
    magnitude: float | None = None

    @property
    def summary(self) -> str:
        if self.delta_type == "metric_shift" and self.magnitude is not None:
            return (
                f"{self.old_belief.capability}: "
                f"{self.old_belief.metric_name} shifted "
                f"{self.old_belief.metric_value} → {self.new_observation.metric_value} "
                f"({self.magnitude:+.0%})"
            )
        return (
            f"{self.old_belief.capability}: "
            f"'{self.old_belief.description}' contradicted by "
            f"'{self.new_observation.description}'"
        )
