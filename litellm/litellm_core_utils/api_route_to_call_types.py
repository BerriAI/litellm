"""
Dictionary mapping API routes to their corresponding CallTypes in LiteLLM.

This dictionary maps each API endpoint to the CallTypes that can be used for that route.
Each route can have both async (prefixed with 'a') and sync call types.
"""

from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes


def get_call_types_for_route(route: str) -> list:
    """
    Get the list of CallTypes for a given API route.

    Args:
        route: API route path (e.g., "/chat/completions")

    Returns:
        List of CallTypes for that route, or empty list if route not found
    """
    return API_ROUTE_TO_CALL_TYPES.get(route, [])


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
