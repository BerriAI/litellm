from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import JsonValue


def with_field(mapping: Mapping[str, JsonValue], path: Sequence[str], value: JsonValue) -> Mapping[str, JsonValue]:
    """Return a copy of `mapping` with the nested key `path` set to `value`, creating intermediate objects."""
    head: Final = path[0]
    if len(path) == 1:
        return {**mapping, head: value}
    inner: Final = mapping.get(head)
    return {**mapping, head: with_field(inner if isinstance(inner, dict) else {}, path[1:], value)}
