from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

_GOOGLE_SEARCH_TOOL_TYPE_PREFIX: Final = "GOOGLE_SEARCH"


@dataclass(frozen=True, slots=True)
class GroundingRequests:
    web_search_requests: int | None
    google_maps_grounding_requests: int | None

    def has_billable_grounding(self) -> bool:
        return bool(self.web_search_requests or self.google_maps_grounding_requests)


def _chunk_kinds(item: Mapping[str, object]) -> frozenset[str]:
    chunks: Final = item.get("groundingChunks")
    if not isinstance(chunks, list):
        return frozenset()
    return frozenset(kind for chunk in chunks if isinstance(chunk, Mapping) for kind in chunk)


def _queries_in_order(item: Mapping[str, object]) -> tuple[str, ...]:
    return _non_empty_strings(item.get("webSearchQueries"))


def _queries(item: Mapping[str, object]) -> frozenset[str]:
    return frozenset(_queries_in_order(item))


def _is_maps_item(item: Mapping[str, object]) -> bool:
    return "maps" in _chunk_kinds(item) or bool(item.get("googleMapsWidgetContextToken"))


def _attributes_queries_to_maps(item: Mapping[str, object]) -> bool:
    return _is_maps_item(item) and "web" not in _chunk_kinds(item)


def calculate_grounding_requests(
    grounding_metadata: Sequence[Mapping[str, object]],
    tool_call_queries: Sequence[str] = (),
) -> GroundingRequests:
    """Billable grounding requests across candidates, counting each distinct query once.

    Web searches are the distinct queries of the Google Search toolCall parts, or of
    groundingMetadata.webSearchQueries (#36377) when no toolCall reported one, since Vertex repeats
    each toolCall query there with its site: operators removed. Empty strings are ignored. Maps grounding
    is floored at one request whenever a candidate carries maps chunks or a widget token,
    since per-prompt billing charges the prompt even when no query is reported.
    """
    items: Final = tuple(item for item in grounding_metadata if isinstance(item, Mapping))
    maps_queries: Final = frozenset(
        query for item in items if _attributes_queries_to_maps(item) for query in _queries(item)
    )
    has_maps: Final = any(_is_maps_item(item) for item in items)
    return GroundingRequests(
        web_search_requests=len(web_search_queries(items, tool_call_queries)) or None,
        google_maps_grounding_requests=max(len(maps_queries), 1) if has_maps else None,
    )


def _non_empty_strings(values: object) -> tuple[str, ...]:
    return tuple(value for value in as_object_list(values) or () if isinstance(value, str) and value)


def _tool_call_search_queries(part: Mapping[str, object]) -> tuple[str, ...]:
    match as_str_mapping(part.get("toolCall")):
        case {"toolType": str() as tool_type, "args": args} if tool_type.startswith(_GOOGLE_SEARCH_TOOL_TYPE_PREFIX):
            return _non_empty_strings((as_str_mapping(args) or {}).get("queries"))
        case _:
            return ()


def _search_tool_call_query_stream(parts: Iterable[Mapping[str, object]]) -> Iterator[str]:
    for part in parts:
        yield from _tool_call_search_queries(part)


def search_tool_call_queries(parts: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(_search_tool_call_query_stream(parts))


def _grounding_query_stream(grounding_metadata: Sequence[Mapping[str, object]]) -> Iterator[str]:
    for item in grounding_metadata:
        if not _attributes_queries_to_maps(item):
            yield from _queries_in_order(item)


def web_search_queries(
    grounding_metadata: Sequence[Mapping[str, object]],
    tool_call_queries: Sequence[str] = (),
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tool_call_queries or _grounding_query_stream(grounding_metadata)))
