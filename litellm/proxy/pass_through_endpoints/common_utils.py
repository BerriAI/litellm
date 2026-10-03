from typing import Final

from fastapi import APIRouter, Request
from fastapi.routing import APIRoute, APIWebSocketRoute

from litellm.types.passthrough_endpoints.pass_through_endpoints import LITELLM_PROVIDER_PASS_THROUGH_ENDPOINT_MARKER


def mark_provider_pass_through_routes(router: APIRouter) -> None:
    for route in router.routes:
        if isinstance(route, (APIRoute, APIWebSocketRoute)):
            setattr(route.endpoint, LITELLM_PROVIDER_PASS_THROUGH_ENDPOINT_MARKER, True)


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key: Final = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")
