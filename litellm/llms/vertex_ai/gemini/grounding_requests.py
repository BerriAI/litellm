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


_SERVER_SIDE_WEB_SEARCH_TOOL_TYPES: Final = frozenset({"GOOGLE_SEARCH_WEB", "GOOGLE_SEARCH"})


def _tool_call_search_queries(part: Mapping[str, object]) -> frozenset[str]:
    tool_call: Final = part.get("toolCall")
    if not isinstance(tool_call, Mapping) or tool_call.get("toolType") not in _SERVER_SIDE_WEB_SEARCH_TOOL_TYPES:
        return frozenset()
    args: Final = tool_call.get("args")
    queries: Final = args.get("queries") if isinstance(args, Mapping) else None
    if not isinstance(queries, list):
        return frozenset()
    return frozenset(query for query in queries if isinstance(query, str) and query)


def server_side_search_queries(candidates: Sequence[Mapping[str, object]] | None) -> frozenset[str]:
    """Distinct Google Search queries carried by server-side tool invocations.

    When ``includeServerSideToolInvocations`` is enabled, Gemini reports completed
    searches as ``toolCall`` parts (``toolType`` GOOGLE_SEARCH_WEB) holding the queries
    it ran, instead of, or in addition to, ``webSearchQueries`` in groundingMetadata.
    These are billed the same way, so they have to be counted too (#49012).
    """
    if not candidates:
        return frozenset()
    queries: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        content: Final = candidate.get("content")
        parts: Final = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, Mapping):
                queries |= _tool_call_search_queries(part)
    return frozenset(queries)


def calculate_grounding_requests(
    grounding_metadata: Sequence[Mapping[str, object]],
    extra_web_queries: frozenset[str] = frozenset(),
) -> GroundingRequests:
    """Billable grounding requests across candidates, counting each distinct query once.

    Duplicate queries within and across grounding metadata items collapse to the
    distinct-query count (#36377), and empty strings are ignored. ``extra_web_queries``
    carries web-search queries reported outside groundingMetadata (server-side tool
    invocations, #49012); they join the distinct web-query set, so a query reported in
    both places is still counted once. Maps grounding is floored at one request whenever
    a candidate carries maps chunks or a widget token, since per-prompt billing charges
    the prompt even when no query is reported.
    """
    items: Final = tuple(item for item in grounding_metadata if isinstance(item, Mapping))
    web_queries: Final = extra_web_queries | frozenset(
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
