"""Async-first composition and the public translation entry points.

``translate_chat_request`` is the pure request transform (parse -> serialize)
the differential gate runs. ``execute_chat_request`` is the full async
pipeline: translate, send through the injected HTTP port (popping the
``json_mode`` transform-seam marker exactly like v1's HTTP handler), parse
the provider response, and serialize the outbound body. Everything returns
one ``Result`` and never raises; sync callers get the one wrapper the seam
provides (v1's completion() already runs on an executor thread).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from expression import Error, Ok, Result

from ..deps import TranslationDeps
from ..dispatch import Provider
from ..errors import TranslateResult, TranslationError
from ..inbound.openai_chat import parse_request
from ..inbound.openai_chat.response import serialize_response
from ..ir import Body, ChatRequest, ChatResponse, PlainJson
from ..providers.anthropic import serialize_request
from ..providers.anthropic.response import parse_response
from .http import Endpoint, ExecuteError, HttpPort, ProviderHttpError

_Serializer = Callable[[ChatRequest, TranslationDeps], Result[Body, TranslationError]]
_ResponseParser = Callable[
    [PlainJson, ChatRequest], Result[ChatResponse, TranslationError]
]

_SERIALIZERS: Mapping[Provider, _Serializer] = MappingProxyType(
    {
        "anthropic": serialize_request,
    }
)

_RESPONSE_PARSERS: Mapping[Provider, _ResponseParser] = MappingProxyType(
    {
        "anthropic": parse_response,
    }
)


def translate_chat_request(
    raw: Mapping[str, object], provider: Provider, deps: TranslationDeps
) -> TranslateResult:
    serializer = _SERIALIZERS.get(provider)
    if serializer is None:
        return Error(
            TranslationError.of_unsupported(
                f"provider {provider!r} has no v2 chat serializer yet"
            )
        )
    return parse_request(raw).bind(lambda request: serializer(request, deps))


def translate_chat_response(
    raw_response: PlainJson,
    request: ChatRequest,
    provider: Provider,
    deps: TranslationDeps,
) -> TranslateResult:
    parser = _RESPONSE_PARSERS.get(provider)
    if parser is None:
        return Error(
            TranslationError.of_unsupported(
                f"provider {provider!r} has no v2 response parser yet"
            )
        )
    return parser(raw_response, request).map(
        lambda response: serialize_response(response, deps)
    )


@dataclass(frozen=True)
class PreparedRequest:
    """A request that passed the fail-closed translation; from here on the
    seam is committed to v2 (an HTTP or response failure surfaces as the
    provider error contract, never a silent re-send through v1)."""

    request: ChatRequest
    body: Body


def prepare_chat_request(
    raw: Mapping[str, object], provider: Provider, deps: TranslationDeps
) -> Result[PreparedRequest, TranslationError]:
    serializer = _SERIALIZERS.get(provider)
    if serializer is None or provider not in _RESPONSE_PARSERS:
        return Error(
            TranslationError.of_unsupported(
                f"provider {provider!r} is not fully ported to v2 yet"
            )
        )
    match parse_request(raw):
        case Result(tag="ok", ok=request):
            pass
        case Result(error=parse_err):
            return Error(parse_err)
    return serializer(request, deps).map(
        lambda body: PreparedRequest(request=request, body=body)
    )


def wire_body(prepared: PreparedRequest) -> Body:
    """v1's HTTP handler pops json_mode from optional_params before the wire;
    the marker exists only at the transform seam."""
    return {key: value for key, value in prepared.body.items() if key != "json_mode"}


async def send_prepared(
    prepared: PreparedRequest,
    provider: Provider,
    deps: TranslationDeps,
    http: HttpPort,
    endpoint: Endpoint,
) -> Result[Body, ExecuteError]:
    parser = _RESPONSE_PARSERS.get(provider)
    if parser is None:
        return Error(
            ExecuteError.of_translation(
                TranslationError.of_unsupported(
                    f"provider {provider!r} has no v2 response parser yet"
                )
            )
        )
    response = await http.post_json(endpoint, wire_body(prepared))
    if response.status_code < 200 or response.status_code >= 300:
        return Error(
            ExecuteError.of_provider_http(
                ProviderHttpError(
                    status_code=response.status_code,
                    text=response.text,
                    headers=response.headers,
                )
            )
        )
    match parser(response.body, prepared.request):
        case Result(tag="ok", ok=chat_response):
            return Ok(serialize_response(chat_response, deps))
        case Result(error=response_err):
            return Error(ExecuteError.of_translation(response_err))


async def execute_chat_request(
    raw: Mapping[str, object],
    provider: Provider,
    deps: TranslationDeps,
    http: HttpPort,
    endpoint: Endpoint,
) -> Result[Body, ExecuteError]:
    match prepare_chat_request(raw, provider, deps):
        case Result(tag="ok", ok=prepared):
            return await send_prepared(prepared, provider, deps, http, endpoint)
        case Result(error=err):
            return Error(ExecuteError.of_translation(err))
