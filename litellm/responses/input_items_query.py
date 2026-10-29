from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

INPUT_ITEMS_QUERY_KEYS: Final = frozenset({"after", "before", "include", "include[]", "limit", "order"})
_MAX_LIMIT_DIGITS: Final = len(str(2**63 - 1))


@dataclass(frozen=True, slots=True)
class InputItemsQuery:
    after: str | None
    before: str | None
    include: tuple[str, ...]
    limit: int
    order: Literal["asc", "desc"]


@dataclass(frozen=True, slots=True)
class InvalidInputItemsQuery:
    param: Literal["after", "before", "include", "limit", "order"]
    message: str


def _parsed_cursor(
    param: Literal["after", "before"], value: object, default: str | None, drop_malformed: bool
) -> str | None | InvalidInputItemsQuery:
    match value:
        case None:
            return default
        case str():
            return value
        case _ if drop_malformed:
            return default
        case _:
            return InvalidInputItemsQuery(param, f"Invalid '{param}': expected a single item id, got {value!r}")


def _parsed_limit(value: object, default: int, drop_malformed: bool) -> int | InvalidInputItemsQuery:
    match value:
        case None:
            return default
        case int() if not isinstance(value, bool) and value >= 0:
            return value
        case str() if value.isascii() and value.isdigit() and len(value) <= _MAX_LIMIT_DIGITS:
            return int(value)
        case _ if drop_malformed:
            return default
        case _:
            return InvalidInputItemsQuery("limit", f"Invalid 'limit': expected an integer >= 0, got {value!r}")


def _parsed_order(
    value: object, default: Literal["asc", "desc"], drop_malformed: bool
) -> Literal["asc", "desc"] | InvalidInputItemsQuery:
    match value:
        case None:
            return default
        case "asc":
            return "asc"
        case "desc":
            return "desc"
        case _ if drop_malformed:
            return default
        case _:
            return InvalidInputItemsQuery("order", f"Invalid 'order': expected asc or desc, got {value!r}")


def _include_values(value: object) -> tuple[object, ...]:
    match value:
        case None:
            return ()
        case list() | tuple():
            return tuple(value)
        case _:
            return (value,)


def _parsed_include(
    include: object, bracketed_include: object, default: tuple[str, ...], drop_malformed: bool
) -> tuple[str, ...] | InvalidInputItemsQuery:
    values: Final = (*_include_values(include), *_include_values(bracketed_include))
    if not values:
        return default
    names: Final = tuple(value for value in values if isinstance(value, str))
    if len(names) == len(values):
        return names
    if drop_malformed:
        return default
    return InvalidInputItemsQuery("include", f"Invalid 'include': expected a list of strings, got {values!r}")


def parse_input_items_query(
    query: Mapping[str, object], defaults: InputItemsQuery, drop_malformed: bool
) -> InputItemsQuery | InvalidInputItemsQuery:
    after: Final = _parsed_cursor("after", query.get("after"), defaults.after, drop_malformed)
    if isinstance(after, InvalidInputItemsQuery):
        return after
    before: Final = _parsed_cursor("before", query.get("before"), defaults.before, drop_malformed)
    if isinstance(before, InvalidInputItemsQuery):
        return before
    include: Final = _parsed_include(query.get("include"), query.get("include[]"), defaults.include, drop_malformed)
    if isinstance(include, InvalidInputItemsQuery):
        return include
    limit: Final = _parsed_limit(query.get("limit"), defaults.limit, drop_malformed)
    if isinstance(limit, InvalidInputItemsQuery):
        return limit
    order: Final = _parsed_order(query.get("order"), defaults.order, drop_malformed)
    if isinstance(order, InvalidInputItemsQuery):
        return order
    return InputItemsQuery(after=after, before=before, include=include, limit=limit, order=order)
