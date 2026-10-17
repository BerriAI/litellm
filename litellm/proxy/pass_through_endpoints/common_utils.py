from collections.abc import Mapping
from typing import Final

import httpx
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import redact_credential_headers


def passthrough_logging_headers(request: Request, user: UserAPIKeyAuth) -> Mapping[str, str]:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import caller_headers_without_litellm_secrets

    return redact_credential_headers(caller_headers_without_litellm_secrets(request, user, frozenset()))


def _loaded_httpx_content(message: httpx.Request | httpx.Response) -> bytes:
    try:
        content: Final = message.content
    except (httpx.RequestNotRead, httpx.ResponseNotRead):
        return b""
    return content


def passthrough_logging_response(response: httpx.Response) -> httpx.Response:
    return httpx.Response(
        status_code=response.status_code,
        headers=response.headers,
        content=_loaded_httpx_content(response),
        request=httpx.Request(
            response.request.method,
            response.request.url,
            headers=redact_credential_headers(response.request.headers),
            content=_loaded_httpx_content(response.request),
        ),
    )


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
