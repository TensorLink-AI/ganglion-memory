"""Simple token-based similarity for belief matching.

No external dependencies. For embedding-based similarity, subclass
the backend and override find_similar.
"""

import re


def tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric token extraction."""
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
