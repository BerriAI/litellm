"""Auth, budget and model-access failures are raised by ``user_api_key_auth`` before the Messages
handler runs, so the central handlers pick the Anthropic envelope from the request route."""

from collections.abc import Mapping
from typing import Final

from fastapi import Request, status
from fastapi.responses import JSONResponse

from litellm.anthropic_interface.exceptions import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    AnthropicExceptionMapping,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.error_body_call_id import error_body_call_id
from litellm.proxy.common_utils.openai_error_payload import LITELLM_CALL_ID_HEADER

ANTHROPIC_MESSAGES_ROUTES: Final[frozenset[str]] = frozenset(
    {
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/claude_code_gateway/v1/messages",
        "/claude_code_gateway/v1/messages/count_tokens",
    }
)
ANTHROPIC_PASS_THROUGH_PREFIX: Final = "/anthropic/"


def uses_anthropic_error_envelope(route: str) -> bool:
    return route in ANTHROPIC_MESSAGES_ROUTES or route.startswith(ANTHROPIC_PASS_THROUGH_PREFIX)


def anthropic_error_status_code(exc: ProxyException) -> int:
    code: Final = exc.code
    if code.isdigit():
        return int(code)
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _with_provider_specific_fields(exc: ProxyException, detail: AnthropicErrorDetail) -> AnthropicErrorDetail:
    if not exc.provider_specific_fields:
        return detail
    with_fields: Final[AnthropicErrorDetail] = {**detail, "provider_specific_fields": exc.provider_specific_fields}
    return with_fields


def _anthropic_error_detail(
    exc: ProxyException, detail: AnthropicErrorDetail, call_id: str | None
) -> AnthropicErrorDetail:
    if call_id is None:
        return _with_provider_specific_fields(exc, detail)
    with_call_id: Final[AnthropicErrorDetail] = {
        **_with_provider_specific_fields(exc, detail),
        "litellm_call_id": call_id,
    }
    return with_call_id


def anthropic_error_content(
    exc: ProxyException,
    status_code: int,
    request_id: str | None,
    general_settings: Mapping[str, object],
) -> AnthropicErrorResponse:
    envelope: Final = AnthropicExceptionMapping.transform_to_anthropic_error(
        status_code=status_code,
        raw_message=exc.message,
        request_id=request_id,
    )
    body_call_id: Final = error_body_call_id(general_settings, exc.headers.get(LITELLM_CALL_ID_HEADER))
    content: Final[AnthropicErrorResponse] = {
        **envelope,
        "error": _anthropic_error_detail(exc, envelope["error"], body_call_id),
    }
    return content


def anthropic_error_json_response(exc: ProxyException, request: Request) -> JSONResponse:
    from litellm.proxy.proxy_server import (
        _close_dangling_otel_server_span,  # pyright: ignore[reportPrivateUsage]  # proxy_server keeps the span-close helper private; a JSONResponse returned instead of a raised error must stamp the OTel server span like the global ProxyException handler does
        general_settings_view,
    )

    status_code: Final = anthropic_error_status_code(exc)
    _close_dangling_otel_server_span(request, status_code, exc=exc)
    content: Final = anthropic_error_content(
        exc, status_code, request.headers.get("x-request-id"), general_settings_view()
    )
    return JSONResponse(status_code=status_code, content=content, headers=exc.headers)


def anthropic_internal_server_error_response(request: Request) -> JSONResponse:
    content: Final = AnthropicExceptionMapping.transform_to_anthropic_error(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        raw_message="Internal server error",
        request_id=request.headers.get("x-request-id"),
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)
