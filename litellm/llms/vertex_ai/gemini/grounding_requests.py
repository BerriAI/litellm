from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final


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


def _queries(item: Mapping[str, object]) -> frozenset[str]:
    queries: Final = item.get("webSearchQueries")
    if not isinstance(queries, list):
        return frozenset()
    return frozenset(query for query in queries if isinstance(query, str) and query)


def _is_maps_item(item: Mapping[str, object]) -> bool:
    return "maps" in _chunk_kinds(item) or bool(item.get("googleMapsWidgetContextToken"))


def _attributes_queries_to_maps(item: Mapping[str, object]) -> bool:
    return _is_maps_item(item) and "web" not in _chunk_kinds(item)


def calculate_grounding_requests(grounding_metadata: Sequence[Mapping[str, object]]) -> GroundingRequests:
    """Billable grounding requests across candidates, counting each distinct query once.

    Duplicate queries within and across grounding metadata items collapse to the
    distinct-query count (#36377), and empty strings are ignored. Maps grounding is
    floored at one request whenever a candidate carries maps chunks or a widget token,
    since per-prompt billing charges the prompt even when no query is reported.
    """
    items: Final = tuple(item for item in grounding_metadata if isinstance(item, Mapping))
    web_queries: Final = frozenset(
        query for item in items if not _attributes_queries_to_maps(item) for query in _queries(item)
    )
    maps_queries: Final = frozenset(
        query for item in items if _attributes_queries_to_maps(item) for query in _queries(item)
    )
    has_maps: Final = any(_is_maps_item(item) for item in items)
    return GroundingRequests(
        web_search_requests=len(web_queries) or None,
        google_maps_grounding_requests=max(len(maps_queries), 1) if has_maps else None,
    )
