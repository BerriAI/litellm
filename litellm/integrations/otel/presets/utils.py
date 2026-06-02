"""Shared helpers for the integration presets."""

from typing import Iterable


def ensure_mappers(mapper_names: Iterable[str], *names: str) -> list[str]:
    """Return ``mapper_names`` with each of ``names`` appended if not already present.

    Order is preserved and duplicates are skipped, so composing several presets
    (or re-applying one) never double-adds a vocabulary.
    """
    result = list(mapper_names)
    for name in names:
        if name not in result:
            result.append(name)
    return result
