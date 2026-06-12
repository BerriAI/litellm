"""Async-first composition and the public translation entry points.

``translate_chat_request`` is the pure request transform (parse -> serialize)
the differential gate runs. ``execute_chat_request`` is the full async
pipeline: translate, send through the injected HTTP port (popping the
transform-seam markers exactly like v1's HTTP handlers), parse the provider
response, and serialize the outbound body. Everything returns one ``Result``
and never raises; sync callers get the one wrapper the seam provides (v1's
completion() already runs on an executor thread).

Per-provider registrations live in the three tables below: the serializer,
the response parser, and the outbound response dialect (which v1 transform
the chunk/message/usage shapes mirror). To add a provider, fill all three.
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
from ..inbound.openai_chat.response import ResponseDialect, serialize_response
from ..ir import Body, ChatRequest, ChatResponse, PlainJson
from ..providers.anthropic import serialize_request
from ..providers.anthropic.response import parse_response
from ..providers.azure import parse_response as azure_parse_response
from ..providers.azure import serialize_request as azure_serialize_request
from ..providers.azure import (
    unsupported_request_shapes as azure_unsupported_request_shapes,
)
from ..providers.azure_ai import claude_parse_response as azure_ai_claude_parse_response
from ..providers.azure_ai import (
    claude_serialize_request as azure_ai_claude_serialize_request,
)
from ..providers.azure_ai import parse_response as azure_ai_parse_response
from ..providers.azure_ai import serialize_request as azure_ai_serialize_request
from ..providers.azure_ai import (
    unsupported_request_shapes as azure_ai_unsupported_request_shapes,
)
from ..providers.bedrock_converse import (
    parse_response as bedrock_converse_parse_response,
)
from ..providers.bedrock_converse import (
    serialize_request as bedrock_converse_serialize_request,
)
from ..providers.bedrock_invoke import parse_response as bedrock_invoke_parse_response
from ..providers.bedrock_invoke import (
    serialize_request as bedrock_invoke_serialize_request,
)
from ..providers.google_genai import parse_response as google_parse_response
from ..providers.google_genai import (
    serialize_request_studio as google_serialize_request_studio,
)
from ..providers.google_genai import (
    serialize_request_vertex as google_serialize_request_vertex,
)
from ..providers.openai_compat import parse_response as openai_compat_parse_response
from ..providers.openai_compat import (
    serialize_request as openai_compat_serialize_request,
)
from ..providers.openai_compat import (
    unsupported_request_shapes as openai_compat_unsupported_request_shapes,
)
from ..providers.vertex_anthropic import (
    parse_response as vertex_anthropic_parse_response,
)
from ..providers.vertex_anthropic import (
    serialize_request as vertex_anthropic_serialize_request,
)
from .http import Endpoint, ExecuteError, HttpPort, ProviderHttpError

_Serializer = Callable[[ChatRequest, TranslationDeps], Result[Body, TranslationError]]
_ResponseParser = Callable[
    [PlainJson, ChatRequest], Result[ChatResponse, TranslationError]
]

_SERIALIZERS: Mapping[Provider, _Serializer] = MappingProxyType(
    {
        "anthropic": serialize_request,
        "bedrock_converse": bedrock_converse_serialize_request,
        "bedrock_invoke": bedrock_invoke_serialize_request,
        "openai_compat": openai_compat_serialize_request,
        "vertex_ai": google_serialize_request_vertex,
        "gemini": google_serialize_request_studio,
        "vertex_anthropic": vertex_anthropic_serialize_request,
        "azure": azure_serialize_request,
        "azure_ai": azure_ai_serialize_request,
        "azure_ai_anthropic": azure_ai_claude_serialize_request,
    }
)

_RESPONSE_PARSERS: Mapping[Provider, _ResponseParser] = MappingProxyType(
    {
        "anthropic": parse_response,
        "bedrock_converse": bedrock_converse_parse_response,
        "bedrock_invoke": bedrock_invoke_parse_response,
        "openai_compat": openai_compat_parse_response,
        "vertex_ai": google_parse_response,
        "gemini": google_parse_response,
        "vertex_anthropic": vertex_anthropic_parse_response,
        "azure": azure_parse_response,
        "azure_ai": azure_ai_parse_response,
        "azure_ai_anthropic": azure_ai_claude_parse_response,
    }
)

_RESPONSE_DIALECTS: Mapping[Provider, ResponseDialect] = MappingProxyType(
    {
        "anthropic": "anthropic",
        "bedrock_converse": "bedrock_converse",
        "bedrock_invoke": "anthropic",  # invoke delegates to the anthropic transform
        "openai_compat": "openai",  # same-family: the wire-derived body
        "vertex_ai": "gemini",
        "gemini": "gemini",
        "vertex_anthropic": "anthropic",  # vertex claude delegates to the anthropic transform
        "azure": "openai",  # same normalizer (convert_to_model_response_object)
        "azure_ai": "openai",
        "azure_ai_anthropic": "anthropic",  # genuine anthropic wire format
    }
)

_RawGuard = Callable[[Mapping[str, object]], TranslationError | None]

_RAW_GUARDS: Mapping[Provider, _RawGuard] = MappingProxyType(
    # Same-family providers run a raw-shape fidelity guard BEFORE parse: the
    # inbound parse normalizes wire forms v1 forwards verbatim, so shapes it
    # cannot round-trip losslessly fall back to v1 as typed errors.
    {
        "openai_compat": openai_compat_unsupported_request_shapes,
        "azure": azure_unsupported_request_shapes,
        "azure_ai": azure_ai_unsupported_request_shapes,
    }
)


def _raw_guard_error(
    raw: Mapping[str, object], provider: Provider
) -> TranslationError | None:
    guard = _RAW_GUARDS.get(provider)
    return guard(raw) if guard is not None else None


def response_dialect(provider: Provider) -> ResponseDialect:
    return _RESPONSE_DIALECTS.get(provider, "anthropic")


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
    guard_error = _raw_guard_error(raw, provider)
    if guard_error is not None:
        return Error(guard_error)
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
    dialect = response_dialect(provider)
    return parser(raw_response, request).map(
        lambda response: serialize_response(response, deps, dialect)
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
    guard_error = _raw_guard_error(raw, provider)
    if guard_error is not None:
        return Error(guard_error)
    match parse_request(raw):
        case Result(tag="ok", ok=request):
            pass
        case Result(error=parse_err):
            return Error(parse_err)
    return serializer(request, deps).map(
        lambda body: PreparedRequest(request=request, body=body)
    )


def wire_body(prepared: PreparedRequest, provider: Provider = "anthropic") -> Body:
    """Strip the transform-seam markers v1's HTTP layer pops before the wire:
    ``json_mode`` (anthropic family) and converse's ``stream`` rider inside
    ``additionalModelRequestFields`` (production pops ``stream`` from
    optional_params before transform, so the wire never carries it)."""
    body = {key: value for key, value in prepared.body.items() if key != "json_mode"}
    if provider != "bedrock_converse":
        return body
    additional = body.get("additionalModelRequestFields")
    if not isinstance(additional, dict):
        return body
    stripped = {key: value for key, value in additional.items() if key != "stream"}
    if stripped:
        return {**body, "additionalModelRequestFields": stripped}
    return {
        key: value
        for key, value in body.items()
        if key != "additionalModelRequestFields"
    }


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
    response = await http.post_json(endpoint, wire_body(prepared, provider))
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
            return Ok(
                serialize_response(chat_response, deps, response_dialect(provider))
            )
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
