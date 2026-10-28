"""Auth, budget and model-access failures are raised by ``user_api_key_auth`` before the Messages
handler runs, so the central handlers pick the Anthropic envelope from the request route."""

from collections.abc import Mapping
from typing import Final

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.types import Scope

from litellm.anthropic_interface.exceptions import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    AnthropicErrorType,
    AnthropicExceptionMapping,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.common_utils.error_body_call_id import error_body_call_id
from litellm.proxy.common_utils.openai_error_payload import LITELLM_CALL_ID_HEADER, openai_error_type_for_status

ANTHROPIC_MESSAGES_ROUTES: Final[frozenset[str]] = frozenset(
    {
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/claude_code_gateway/v1/messages",
        "/claude_code_gateway/v1/messages/count_tokens",
    }
)
ANTHROPIC_PASS_THROUGH_PREFIX: Final = "/anthropic/"
ANTHROPIC_SHAPED_MODEL_LIST_ROUTES: Final[frozenset[str]] = frozenset(
    {"/v1/models", "/models", "/claude_code_gateway/v1/models"}
)


def uses_anthropic_error_envelope(request: Request) -> bool:
    route: Final = get_request_route(request)
    if route in ANTHROPIC_MESSAGES_ROUTES or route.startswith(ANTHROPIC_PASS_THROUGH_PREFIX):
        return True
    return route in ANTHROPIC_SHAPED_MODEL_LIST_ROUTES and "anthropic-version" in request.headers


def route_error_content(
    scope: Scope, anthropic_type: AnthropicErrorType, message: str, openai_content: Mapping[str, object]
) -> Mapping[str, object]:
    request: Final = Request(scope)
    if not uses_anthropic_error_envelope(request):
        return openai_content
    content: Final[AnthropicErrorResponse] = {"type": "error", "error": {"type": anthropic_type, "message": message}}
    request_id: Final = request.headers.get("x-request-id")
    return content if request_id is None else {**content, "request_id": request_id}


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


def _with_litellm_error_type(
    exc: ProxyException, status_code: int, detail: AnthropicErrorDetail
) -> AnthropicErrorDetail:
    if not exc.type or exc.type in (detail["type"], openai_error_type_for_status(status_code)):
        return detail
    with_type: Final[AnthropicErrorDetail] = {**detail, "litellm_error_type": exc.type}
    return with_type


def _anthropic_error_detail(
    exc: ProxyException, status_code: int, detail: AnthropicErrorDetail, call_id: str | None
) -> AnthropicErrorDetail:
    with_proxy_fields: Final = _with_litellm_error_type(exc, status_code, _with_provider_specific_fields(exc, detail))
    if call_id is None:
        return with_proxy_fields
    with_call_id: Final[AnthropicErrorDetail] = {**with_proxy_fields, "litellm_call_id": call_id}
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
        "error": _anthropic_error_detail(exc, status_code, envelope["error"], body_call_id),
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
