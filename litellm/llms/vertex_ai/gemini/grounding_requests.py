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


def _query_count(item: Mapping[str, object]) -> int:
    queries: Final = item.get("webSearchQueries")
    if not isinstance(queries, list):
        return 0
    return len([query for query in queries if query])


def grounding_item_requests(item: Mapping[str, object]) -> GroundingRequests:
    kinds: Final = _chunk_kinds(item)
    queries: Final = _query_count(item)
    if "maps" not in kinds and not item.get("googleMapsWidgetContextToken"):
        return GroundingRequests(web_search_requests=queries or None, google_maps_grounding_requests=None)
    if "web" in kinds:
        return GroundingRequests(web_search_requests=queries or None, google_maps_grounding_requests=1)
    return GroundingRequests(web_search_requests=None, google_maps_grounding_requests=max(queries, 1))


def _total(counts: Sequence[int | None]) -> int | None:
    present: Final = tuple(count for count in counts if count is not None)
    return sum(present) if present else None


def calculate_grounding_requests(grounding_metadata: Sequence[Mapping[str, object]]) -> GroundingRequests:
    per_item: Final = tuple(grounding_item_requests(item) for item in grounding_metadata if isinstance(item, Mapping))
    return GroundingRequests(
        web_search_requests=_total(tuple(item.web_search_requests for item in per_item)),
        google_maps_grounding_requests=_total(tuple(item.google_maps_grounding_requests for item in per_item)),
    )
