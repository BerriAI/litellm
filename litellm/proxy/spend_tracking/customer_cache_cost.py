from collections.abc import Mapping
from sys import float_info
from typing import Final

from pydantic import TypeAdapter

_FIELDS: Final = TypeAdapter(dict[str, object])


def recorded_customer_cache_cost(
    breakdown: object, cache_read_tokens: int, cache_write_tokens: int
) -> tuple[float, int]:
    if not isinstance(breakdown, Mapping):
        return 0.0, 0
    fields: Final = _FIELDS.validate_python(breakdown)
    read: Final = _recorded_amount(fields.get("cache_read_cost"))
    write: Final = _recorded_amount(fields.get("cache_creation_cost"))
    return (
        (read or 0.0) + (write or 0.0),
        (cache_read_tokens if read is not None else 0) + (cache_write_tokens if write is not None else 0),
    )


def _recorded_amount(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= float_info.max:
        return None
    return float(value)
