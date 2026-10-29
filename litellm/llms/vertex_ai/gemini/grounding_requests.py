from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

# toolType values Gemini uses for Google Search in server-side tool invocations
# (includeServerSideToolInvocations / include_server_side_tool_invocations).
_GOOGLE_SEARCH_INVOCATION_TYPES: Final = frozenset(
    {
        "GOOGLE_SEARCH_WEB",
        "GOOGLE_SEARCH",
        "GOOGLE_SEARCH_RETRIEVAL",
        "google_search",
        "googleSearch",
        "google_search_retrieval",
        "googleSearchRetrieval",
    }
)


def web_search_queries_from_server_side_invocations(
    invocations: Sequence[Mapping[str, object]] | None,
) -> frozenset[str]:
    """Distinct nonempty search queries from Gemini server-side tool invocations.

    When ``include_server_side_tool_invocations`` is on, search evidence often
    arrives only as ``toolCall`` / ``toolResponse`` parts (``args.queries`` or
    ``args.query``), not as ``groundingMetadata.webSearchQueries``. Count those
    the same way grounding metadata is counted so usage and tool cost stay
    accurate (#49012).
    """
    if not invocations:
        return frozenset()
    queries: set[str] = set()
    for invocation in invocations:
        if not isinstance(invocation, Mapping):
            continue
        tool_type = invocation.get("tool_type")
        if tool_type is None:
            tool_type = invocation.get("toolType")
        if tool_type not in _GOOGLE_SEARCH_INVOCATION_TYPES:
            if not (isinstance(tool_type, str) and "search" in tool_type.lower() and "map" not in tool_type.lower()):
                continue
        args = invocation.get("args")
        if not isinstance(args, Mapping):
            continue
        raw_queries = args.get("queries")
        if isinstance(raw_queries, list):
            queries.update(q for q in raw_queries if isinstance(q, str) and q)
        raw_query = args.get("query")
        if isinstance(raw_query, str) and raw_query:
            queries.add(raw_query)
    return frozenset(queries)



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


def calculate_grounding_requests(
    grounding_metadata: Sequence[Mapping[str, object]],
    server_side_invocations: Sequence[Mapping[str, object]] | None = None,
) -> GroundingRequests:
    """Billable grounding requests across candidates, counting each distinct query once.

    Duplicate queries within and across grounding metadata items collapse to the
    distinct-query count (#36377), and empty strings are ignored. Maps grounding is
    floored at one request whenever a candidate carries maps chunks or a widget token,
    since per-prompt billing charges the prompt even when no query is reported.

    When search evidence is only in server-side tool invocations (#49012), those
    queries are unioned with grounding-metadata queries so each distinct string
    still counts once.
    """
    items: Final = tuple(item for item in grounding_metadata if isinstance(item, Mapping))
    web_queries: Final = frozenset(
        query for item in items if not _attributes_queries_to_maps(item) for query in _queries(item)
    ) | web_search_queries_from_server_side_invocations(server_side_invocations)
    maps_queries: Final = frozenset(
        query for item in items if _attributes_queries_to_maps(item) for query in _queries(item)
    )
    has_maps: Final = any(_is_maps_item(item) for item in items)
    return GroundingRequests(
        web_search_requests=len(web_queries) or None,
        google_maps_grounding_requests=max(len(maps_queries), 1) if has_maps else None,
    )
