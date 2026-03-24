"""ReMem-style refine tools for active memory editing.

Opt-in tools for LLM-driven memory management. These are building
blocks for an agent that actively manages its own memory store:

    merge_experiences   — combine multiple experiences into one
    split_experience    — split one experience into multiple
    rewrite_experience  — update the content of an experience
    forget_experience   — delete an experience

Inspired by the ReMem paper: instead of fixed memory policies,
let the LLM decide when to merge, split, rewrite, or forget.

Usage:
    from ganglion.memory.refine import merge_experiences, rewrite_experience

    # LLM decides these two experiences should be merged
    merged = await merge_experiences(memory, [exp1.id, exp2.id], "combined insight")

    # LLM rewrites stale memory
    updated = await rewrite_experience(memory, exp.id, "corrected understanding")
"""

from __future__ import annotations

from datetime import UTC, datetime

from ganglion.memory.core import Memory
from ganglion.memory.types import Experience


async def merge_experiences(
    memory: Memory,
    experience_ids: list[int],
    new_content: str,
) -> Experience:
    """Merge multiple experiences into one new experience.

    Sums confirmation and contradiction counts. Unions tags.
    Deletes the originals. Preserves the earliest created_at.

    Args:
        memory:         The Memory instance to operate on.
        experience_ids: IDs of experiences to merge (minimum 2).
        new_content:    Content for the merged experience.

    Returns:
        The newly created merged Experience.

    Raises:
        ValueError: If fewer than 2 IDs provided or any ID not found.
    """
    if len(experience_ids) < 2:
        raise ValueError("Need at least 2 experiences to merge")

    experiences: list[Experience] = []
    for eid in experience_ids:
        exp = await memory.get(eid)
        if exp is None:
            raise ValueError(f"Experience {eid} not found")
        experiences.append(exp)

    # Aggregate
    total_confirms = sum(e.confirmation_count for e in experiences)
    total_contradicts = sum(e.contradiction_count for e in experiences)
    all_tags = sorted({t for e in experiences for t in e.tags})
    source = experiences[0].source
    earliest = min(e.created_at for e in experiences)

    # Delete originals
    for exp in experiences:
        if exp.id is not None:
            await memory.delete(exp.id)

    # Create merged experience
    embedding = await memory._embed(new_content)
    now = datetime.now(UTC)
    merged = Experience(
        content=new_content,
        tags=tuple(all_tags),
        source=source,
        created_at=earliest,
        updated_at=now,
        confirmation_count=total_confirms,
        contradiction_count=total_contradicts,
        embedding=embedding,
    )
    await memory.backend.store(merged)
    return merged


async def split_experience(
    memory: Memory,
    experience_id: int,
    new_contents: list[str],
) -> list[Experience]:
    """Split one experience into multiple new ones.

    Each new experience inherits the original's source and tags.
    The original is deleted. New experiences start with zero counts.

    Args:
        memory:        The Memory instance to operate on.
        experience_id: ID of the experience to split.
        new_contents:  Content strings for the new experiences.

    Returns:
        List of newly created Experience objects.

    Raises:
        ValueError: If new_contents is empty or experience not found.
    """
    if not new_contents:
        raise ValueError("Need at least one content string to split into")

    original = await memory.get(experience_id)
    if original is None:
        raise ValueError(f"Experience {experience_id} not found")

    # Delete original
    await memory.delete(experience_id)

    # Create new experiences
    results: list[Experience] = []
    now = datetime.now(UTC)
    for content in new_contents:
        embedding = await memory._embed(content)
        exp = Experience(
            content=content,
            tags=original.tags,
            source=original.source,
            created_at=now,
            updated_at=now,
            confirmation_count=0,
            contradiction_count=0,
            embedding=embedding,
        )
        await memory.backend.store(exp)
        results.append(exp)

    return results


async def rewrite_experience(
    memory: Memory,
    experience_id: int,
    new_content: str,
) -> Experience:
    """Rewrite the content of an experience.

    Preserves confirmation/contradiction counts, tags, and source.
    Updates the embedding to match the new content.

    Args:
        memory:        The Memory instance to operate on.
        experience_id: ID of the experience to rewrite.
        new_content:   The new content text.

    Returns:
        The updated Experience.

    Raises:
        ValueError: If experience not found.
    """
    exp = await memory.get(experience_id)
    if exp is None:
        raise ValueError(f"Experience {experience_id} not found")

    exp.content = new_content
    exp.updated_at = datetime.now(UTC)
    exp.embedding = await memory._embed(new_content)
    await memory.backend.update(exp)
    return exp


async def forget_experience(
    memory: Memory,
    experience_id: int,
) -> None:
    """Delete an experience. No-op if it doesn't exist.

    Args:
        memory:        The Memory instance to operate on.
        experience_id: ID of the experience to forget.
    """
    exp = await memory.get(experience_id)
    if exp is not None and exp.id is not None:
        await memory.delete(exp.id)
