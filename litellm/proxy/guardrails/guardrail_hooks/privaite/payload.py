import functools
from collections.abc import Callable, Sequence
from typing import Final, TypeAlias

JsonObject: TypeAlias = dict[str, object]

JsonArray: TypeAlias = list[object]

Write: TypeAlias = Callable[[str], None]

Slot: TypeAlias = tuple[str, Write]


def as_object(value: object) -> JsonObject | None:
    """`value` as a JSON object, or None. A JSON object's keys are strings, which isinstance cannot state."""
    return value if isinstance(value, dict) else None  # pyright: ignore[reportUnknownVariableType]  # JSON keys are str


def as_array(value: object) -> JsonArray | None:
    """`value` as a JSON array, or None. See `as_object`."""
    return value if isinstance(value, list) else None  # pyright: ignore[reportUnknownVariableType]  # see as_object


def read_field(holder: object, name: str) -> object:
    """Reads one field from a dict or from an object.

    LiteLLM hands the hooks Pydantic models on some paths and plain dicts on others,
    so every walk here reads both shapes.
    """
    fields: Final = as_object(holder)
    if fields is not None:
        return fields.get(name)
    return getattr(holder, name, None)


def read_list(holder: object, name: str) -> Sequence[object]:
    """Reads a list field from a dict or an object. Anything else reads as empty."""
    return as_array(read_field(holder, name)) or ()


def write_field(holder: object, name: str, value: object) -> None:
    """Writes one field back into a dict or an object. Pairs with read_field."""
    fields: Final = as_object(holder)
    if fields is not None:
        fields[name] = value
    else:
        setattr(holder, name, value)


def field_writer(fields: JsonObject, key: str) -> Write:
    return functools.partial(fields.__setitem__, key)


def item_writer(items: JsonArray, index: int) -> Write:
    return functools.partial(items.__setitem__, index)


def index_of(holder: object) -> int:
    """A streamed choice or tool call is matched across chunks by its index, 0 when absent."""
    index: Final = read_field(holder, "index")
    return index if isinstance(index, int) else 0
