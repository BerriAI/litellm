"""
Dictionary mapping API routes to their corresponding CallTypes in LiteLLM.

This dictionary maps each API endpoint to the CallTypes that can be used for that route.
Each route can have both async (prefixed with 'a') and sync call types.

Route patterns may contain placeholders like {agent_id}, {model}, {batch_id}; these
match a single path segment when resolving call types for a concrete path.
"""

from collections.abc import Sequence
from typing import Final

from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes


def _segment_matches(route_segment: str, pattern_segment: str) -> bool:
    """
    Match one concrete path segment against one pattern segment.
    A bare placeholder ({param}) matches any segment; a placeholder with a
    literal suffix ({model}:generateContent) requires the segment to end with
    that suffix and have a non-empty value before it.
    """
    if not pattern_segment.startswith("{"):
        return route_segment == pattern_segment
    placeholder_end: Final = pattern_segment.find("}")
    if placeholder_end == -1:
        return route_segment == pattern_segment
    literal_suffix: Final = pattern_segment[placeholder_end + 1 :]
    if not literal_suffix:
        return True
    return route_segment.endswith(literal_suffix) and len(route_segment) > len(literal_suffix)


def _pattern_tail_spans_segments(pattern_tail: str) -> bool:
    """
    Whether the pattern's last segment is a suffixed placeholder
    ({model}:generateContent) that may absorb extra route segments, mirroring
    FastAPI's {model_name:path} converter for slash-containing model names.
    """
    return pattern_tail.startswith("{") and "}" in pattern_tail and not pattern_tail.endswith("}")


def _route_matches_pattern(route: str, pattern: str) -> bool:
    """
    Return True if the concrete route matches the pattern.
    Pattern segments like {param} match any single path segment, and a
    suffixed placeholder in the last segment may span multiple segments.
    """
    route_parts: Final = route.strip("/").split("/")
    pattern_parts: Final = pattern.strip("/").split("/")
    if len(route_parts) < len(pattern_parts):
        return False
    if len(route_parts) > len(pattern_parts) and not _pattern_tail_spans_segments(pattern_parts[-1]):
        return False
    head_count: Final = len(pattern_parts) - 1
    merged_parts: Final = (*route_parts[:head_count], "/".join(route_parts[head_count:]))
    return all(_segment_matches(r, p) for r, p in zip(merged_parts, pattern_parts))


def get_call_types_for_route(route: str) -> Sequence[CallTypes] | None:
    """
    Get the CallTypes for a given API route.

    Supports both exact keys and dynamic patterns (e.g. /a2a/my-agent/message/send
    matches /a2a/{agent_id}/message/send).

    Args:
        route: API route path (e.g., "/chat/completions" or "/a2a/my-pydantic-agent/message/send")

    Returns:
        CallTypes for that route, or None if route not found
    """
    exact: Final = API_ROUTE_TO_CALL_TYPES.get(route, None)
    if exact is not None:
        return exact
    for pattern, call_types in API_ROUTE_TO_CALL_TYPES.items():
        if _route_matches_pattern(route, pattern):
            return call_types
    return None


def get_routes_for_call_type(call_type: CallTypes) -> list:
    """
    Get all routes that use a specific CallType.

    Args:
        call_type: The CallType to search for

    Returns:
        List of routes that use this CallType
    """
    routes: Final = []
    for route, types in API_ROUTE_TO_CALL_TYPES.items():
        if call_type in types:
            routes.append(route)
    return routes
