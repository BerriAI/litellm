from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue


def changed_section_keys(
    baseline: Mapping[str, JsonValue], new: Mapping[str, JsonValue]
) -> tuple[Mapping[str, JsonValue], frozenset[str]]:
    changed: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {key: value for key, value in new.items() if key not in baseline or baseline[key] != value}
    )
    removed: Final = frozenset(baseline).difference(new)
    return changed, removed
