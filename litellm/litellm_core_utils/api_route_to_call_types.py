"""
Dictionary mapping API routes to their corresponding CallTypes in LiteLLM.

This dictionary maps each API endpoint to the CallTypes that can be used for that route.
Each route can have both async (prefixed with 'a') and sync call types.

Route patterns may contain placeholders like {agent_id}, {model}, {batch_id}; these
match a single path segment when resolving call types for a concrete path.
"""

from typing import List, Optional

from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes


def _route_matches_pattern(route: str, pattern: str) -> bool:
    """
    Return True if the concrete route matches the pattern.
    Pattern segments like {param} match any single path segment.
    """
    route_parts = route.strip("/").split("/")
    pattern_parts = pattern.strip("/").split("/")
    if len(route_parts) != len(pattern_parts):
        return False
    for r, p in zip(route_parts, pattern_parts):
        if p.startswith("{") and p.endswith("}"):
            continue
        if r != p:
            return False
    return True


def get_call_types_for_route(route: str) -> Optional[List[CallTypes]]:
    """
    Get the list of CallTypes for a given API route.

    Supports both exact keys and dynamic patterns (e.g. /a2a/my-agent/message/send
    matches /a2a/{agent_id}/message/send).

    Args:
        route: API route path (e.g., "/chat/completions" or "/a2a/my-pydantic-agent/message/send")

    Returns:
        List of CallTypes for that route, or None if route not found
    """
    exact = API_ROUTE_TO_CALL_TYPES.get(route, None)
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
    routes = []
    for route, types in API_ROUTE_TO_CALL_TYPES.items():
        if call_type in types:
            routes.append(route)
    return routes
