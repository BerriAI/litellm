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
from ..providers.cohere import parse_response as cohere_parse_response
from ..providers.cohere import serialize_request as cohere_serialize_request
from ..providers.cohere import (
    unsupported_request_shapes as cohere_unsupported_request_shapes,
)
from ..providers.compat_httpx import GUARDS as compat_httpx_guards
from ..providers.compat_httpx import PARSERS as compat_httpx_parsers
from ..providers.compat_httpx import SERIALIZERS as compat_httpx_serializers
from ..providers.compat_httpx.response import ResponseStyle
from ..providers.compat_sdk import GUARDS as compat_sdk_guards
from ..providers.compat_sdk import SERIALIZERS as compat_sdk_serializers
from ..providers.deepseek import parse_response as deepseek_parse_response
from ..providers.deepseek import serialize_request as deepseek_serialize_request
from ..providers.deepseek import (
    unsupported_request_shapes as deepseek_unsupported_request_shapes,
)
from ..providers.fireworks_ai import parse_response as fireworks_parse_response
from ..providers.fireworks_ai import serialize_request as fireworks_serialize_request
from ..providers.fireworks_ai import (
    unsupported_request_shapes as fireworks_unsupported_request_shapes,
)
from ..providers.google_genai import parse_response as google_parse_response
from ..providers.google_genai import (
    serialize_request_studio as google_serialize_request_studio,
)
from ..providers.google_genai import (
    serialize_request_vertex as google_serialize_request_vertex,
)
from ..providers.google_genai import (
    unsupported_request_shapes as google_unsupported_request_shapes,
)
from ..providers.groq import parse_response as groq_parse_response
from ..providers.groq import serialize_request as groq_serialize_request
from ..providers.groq import (
    unsupported_request_shapes as groq_unsupported_request_shapes,
)
from ..providers.hosted_vllm import parse_response as hosted_vllm_parse_response
from ..providers.hosted_vllm import serialize_request as hosted_vllm_serialize_request
from ..providers.hosted_vllm import (
    unsupported_request_shapes as hosted_vllm_unsupported_request_shapes,
)
from ..providers.huggingface import parse_response as huggingface_parse_response
from ..providers.huggingface import serialize_request as huggingface_serialize_request
from ..providers.huggingface import (
    unsupported_request_shapes as huggingface_unsupported_request_shapes,
)
from ..providers.mistral import parse_response as mistral_parse_response
from ..providers.mistral import serialize_request as mistral_serialize_request
from ..providers.mistral import (
    unsupported_request_shapes as mistral_unsupported_request_shapes,
)
from ..providers.openai_compat import parse_response as openai_compat_parse_response
from ..providers.openai_compat import (
    serialize_request as openai_compat_serialize_request,
)
from ..providers.openai_compat import (
    unsupported_request_shapes as openai_compat_unsupported_request_shapes,
)
from ..providers.openrouter import parse_response as openrouter_parse_response
from ..providers.openrouter import serialize_request as openrouter_serialize_request
from ..providers.openrouter import (
    unsupported_request_shapes as openrouter_unsupported_request_shapes,
)
from ..providers.sagemaker_chat import (
    parse_response as sagemaker_chat_parse_response,
)
from ..providers.sagemaker_chat import (
    serialize_request as sagemaker_chat_serialize_request,
)
from ..providers.sagemaker_chat import (
    unsupported_request_shapes as sagemaker_chat_unsupported_request_shapes,
)
from ..providers.snowflake import parse_response as snowflake_parse_response
from ..providers.snowflake import serialize_request as snowflake_serialize_request
from ..providers.snowflake import (
    unsupported_request_shapes as snowflake_unsupported_request_shapes,
)
from ..providers.vertex_anthropic import (
    parse_response as vertex_anthropic_parse_response,
)
from ..providers.vertex_anthropic import (
    serialize_request as vertex_anthropic_serialize_request,
)
from ..providers.watsonx import parse_response as watsonx_parse_response
from ..providers.watsonx import serialize_request as watsonx_serialize_request
from ..providers.watsonx import (
    unsupported_request_shapes as watsonx_unsupported_request_shapes,
)
from ..providers.xai import parse_response as xai_parse_response
from ..providers.xai import serialize_request as xai_serialize_request
from ..providers.xai import (
    unsupported_request_shapes as xai_unsupported_request_shapes,
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
        "xai": xai_serialize_request,
        # compat_sdk family: the registry is the family's own frozen
        # PROFILES-derived table, spliced in whole. This splice (one line per
        # TABLE, never per provider) is the registration convention every
        # wave-1b/2 family follows.
        **compat_sdk_serializers,
        # compat_httpx family (wave-1b): the dedicated-elif shims; same
        # splice convention, second family variant (no seam model preset).
        **compat_httpx_serializers,
        # wave-2b-alpha own-module providers (per-provider rows: these are
        # NOT family members; each package owns its serializer).
        "deepseek": deepseek_serialize_request,
        "openrouter": openrouter_serialize_request,
        "hosted_vllm": hosted_vllm_serialize_request,
        "fireworks_ai": fireworks_serialize_request,
        "snowflake": snowflake_serialize_request,
        "huggingface": huggingface_serialize_request,
        # wave-2b-beta own modules: cohere/cohere_chat are ONE module (the
        # main.py elif serves both provider names; v2 is the default route).
        "cohere": cohere_serialize_request,
        "cohere_chat": cohere_serialize_request,
        "mistral": mistral_serialize_request,
        "watsonx": watsonx_serialize_request,
        "sagemaker_chat": sagemaker_chat_serialize_request,
        "groq": groq_serialize_request,
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
        "xai": xai_parse_response,
        # compat_sdk family: the live v1 normalizer is the same
        # convert_to_model_response_object the openai parser mirrors; the
        # {provider}/{wire_model} re-prefix is the seam's preset arm
        # (_to_model_response_openai), not parser scope. All members are
        # SDK-path (cometapi moved to compat_httpx at the sibling merge).
        **{
            provider: openai_compat_parse_response
            for provider in compat_sdk_serializers
        },
        # compat_httpx family: the openai parser with NO seam preset (bare
        # wire model, the xai R4 rule) except the compactifai/amazon-nova/
        # lemonade request-model prefixes, which are parser scope (the
        # family's PARSERS table carries the per-provider truth).
        **compat_httpx_parsers,
        # wave-2b-alpha own-module providers. deepseek/openrouter: the base
        # GPT transform_response is live on their httpx elifs -> the shared
        # openai parser with NO seam preset (bare wire model, the xai R4
        # rule; openrouter's usage.cost hidden-params header is a fork
        # obligation pinned in its response gate).
        "deepseek": deepseek_parse_response,
        "openrouter": openrouter_parse_response,
        "hosted_vllm": hosted_vllm_parse_response,
        # fireworks_ai: its OWN transform_response (the OpenAILike DIRECT
        # construction + the fireworks_ai/{wire model} prefix) -> the shared
        # direct-parser factory with the fireworks policy; the seam fork must
        # use the "openai_like" construction arm (RESPONSE_STYLE, pinned).
        "fireworks_ai": fireworks_parse_response,
        # snowflake: content_list pre-rewrite + the direct parser with the
        # snowflake/{wire model} prefix policy ("openai_like" seam arm).
        "snowflake": snowflake_parse_response,
        "huggingface": huggingface_parse_response,
        # wave-2b-beta: the cohere parser builds the normalized body itself
        # (cohere-native wire) and rides it on ChatResponse.wire; the seam's
        # "openai" construction arm reproduces v1's fresh-ModelResponse
        # mutation byte-for-byte (probed; finish is ALWAYS "stop" in v1).
        "cohere": cohere_parse_response,
        "cohere_chat": cohere_parse_response,
        # mistral: two raw-body pre-steps (empty-content -> None, magistral
        # content-list collapse) then the shared openai parser (bare wire
        # model — the cdr fresh-ModelResponse arm).
        "mistral": mistral_parse_response,
        # watsonx: openai parse for validation, then the verbatim body with
        # the LIVE "watsonx/{wire_model}" prefix (the openai_like arm's one
        # wave-2b consumer; the seam must construct with usage_style
        # "openai_like", NOT "openai").
        "watsonx": watsonx_parse_response,
        # sagemaker_chat: the shared openai parser verbatim (base
        # transform_response is LIVE; bare wire model, no seam preset).
        "sagemaker_chat": sagemaker_chat_parse_response,
        # groq: openai parse for validation, verbatim wire + the
        # service_tier clamp (bare wire model; construction arm
        # "openai_like" — the direct ModelResponse(**json) style).
        "groq": groq_parse_response,
    }
)

OWN_MODULE_RESPONSE_STYLES: Mapping[Provider, ResponseStyle] = MappingProxyType(
    {
        # The v1 response-CONSTRUCTION style per own-module provider — the
        # machine-readable truth any future completion() fork must select
        # ``usage_style`` from, exactly like compat_httpx.RESPONSE_STYLES
        # (critic-wave2b-alpha MAJOR-4: this obligation lived in prose).
        # "openai" = cdr via the base GPT transform_response; "openai_like" =
        # ModelResponse(**json) DIRECT construction (no stop->tool_calls
        # rewrite, different pydantic dump, NO seam preset either way).
        # Wrong-arm divergence is pinned per MEMBER in its response gate:
        # openai_like members ride a verbatim wire index 5 the cdr arm
        # enumerate-rewrites (the fireworks_ai/snowflake template); openai
        # members use the INVERTED template — a float wire ``created`` the
        # cdr arm coerces-and-serves like v1 while the direct construction
        # raises ValidationError (cohere pins the ambient envelope id its
        # parser-built body cannot carry instead) — verifier-wave2b-final
        # F1 closed the eight value-unpinned "openai" rows. The
        # registration gate asserts every own-module provider has a row.
        # The seam's AST gate
        # (test_seam_forks_never_select_usage_style_via_response_dialect)
        # already rejects any fork that routes through response_dialect().
        "deepseek": "openai",
        "openrouter": "openai",
        "hosted_vllm": "openai",
        "huggingface": "openai",
        "fireworks_ai": "openai_like",
        "snowflake": "openai_like",
        # wave-2b-beta own modules, registered at the sibling merge
        # (integrator consistency sweep — the gate's coverage now spans ALL
        # eleven wave-2b own modules). cohere/mistral/sagemaker_chat ride
        # the cdr arm ("openai"); watsonx and groq are DIRECT
        # ModelResponse(**json) construction ("openai_like") with wrong-arm
        # divergence pins in their response gates (the fireworks/snowflake
        # template: wire index 5 rides verbatim on the correct arm, the cdr
        # arm enumerate-rewrites it to 0).
        "cohere": "openai",
        "cohere_chat": "openai",
        "mistral": "openai",
        "sagemaker_chat": "openai",
        "watsonx": "openai_like",
        "groq": "openai_like",
    }
)

_OPENAI_DIALECT: ResponseDialect = "openai"

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
        "xai": "openai",  # httpx path, same normalized wire-body ride
        # compat_sdk family: SDK path, default openai wrapper arm (the
        # per-provider stream replays pin that no dedicated branch fires).
        **{provider: _OPENAI_DIALECT for provider in compat_sdk_serializers},
        # compat_httpx family: httpx path, same normalized wire-body ride
        # (the chunk-fold dialect is "xai" — the generic httpx dict path —
        # and the per-provider line parser is the family's LINE_PARSERS
        # table: cometapi's strict-envelope policy row, the shared factory
        # policy for the rest — selected by the stream gates/future
        # streaming seam, not this outbound-body table)
        **{provider: _OPENAI_DIALECT for provider in compat_httpx_serializers},
        # wave-2b-alpha own-module providers: httpx path, wire-derived
        # outbound bodies (per-provider stream truth lives in each package's
        # stream.py, composed from the shared httpx_chunk factory).
        "deepseek": _OPENAI_DIALECT,
        "openrouter": _OPENAI_DIALECT,
        "hosted_vllm": _OPENAI_DIALECT,
        "fireworks_ai": _OPENAI_DIALECT,
        "snowflake": _OPENAI_DIALECT,
        "huggingface": _OPENAI_DIALECT,
        # wave-2b-beta: the cohere parser rides the normalized
        # chat-completion body on ChatResponse.wire (the openai outbound
        # dialect); the chunk-fold dialect is "generic" (the wrapper's
        # GenericStreamingChunk arm — selected by the stream gates/future
        # streaming seam, not this outbound-body table).
        "cohere": _OPENAI_DIALECT,
        "cohere_chat": _OPENAI_DIALECT,
        # mistral: openai outbound body; the chunk-fold dialect is "xai"
        # (the generic httpx dict path) over the mistral line parser.
        "mistral": _OPENAI_DIALECT,
        # watsonx: openai outbound body (the parser rides it on wire); the
        # chunk-fold dialect is "generic" and the seam CONSTRUCTION arm is
        # "openai_like" — this table is the outbound-body dialect only (the
        # construction-arm gate guards the seam read).
        "watsonx": _OPENAI_DIALECT,
        # sagemaker_chat: openai everywhere (chunk-fold dialect "openai" at
        # the AWS event-stream parsed-event seam).
        "sagemaker_chat": _OPENAI_DIALECT,
        # groq: openai outbound body; chunk-fold dialect "xai" (the httpx
        # dict path) over the groq line parser; seam construction arm
        # "openai_like".
        "groq": _OPENAI_DIALECT,
    }
)

_RawGuard = Callable[[Mapping[str, object]], TranslationError | None]

_RAW_GUARDS: Mapping[Provider, _RawGuard] = MappingProxyType(
    # Raw-shape guards run BEFORE parse. Same-family providers use them for
    # fidelity (the inbound parse normalizes wire forms v1 forwards
    # verbatim); the google routes use one because the parse DROPS the
    # message ``name`` field, whose bytes the cache-marker token bound must
    # otherwise account for (verifier-integration blocker).
    {
        "openai_compat": openai_compat_unsupported_request_shapes,
        "azure": azure_unsupported_request_shapes,
        "azure_ai": azure_ai_unsupported_request_shapes,
        "vertex_ai": google_unsupported_request_shapes,
        "gemini": google_unsupported_request_shapes,
        "xai": xai_unsupported_request_shapes,
        # the family GUARDS tables are complete (per-provider overrides for
        # the cache_control-preserving and content-list-flattening configs,
        # the shared guard for everyone else)
        **compat_sdk_guards,
        **compat_httpx_guards,
        # wave-2b-alpha own-module providers (same-family wire formats; each
        # guard composes the shared openai guard with its provider arms).
        "deepseek": deepseek_unsupported_request_shapes,
        "openrouter": openrouter_unsupported_request_shapes,
        "hosted_vllm": hosted_vllm_unsupported_request_shapes,
        "fireworks_ai": fireworks_unsupported_request_shapes,
        "snowflake": snowflake_unsupported_request_shapes,
        "huggingface": huggingface_unsupported_request_shapes,
        # wave-2b-beta: the cohere guard carries the v1-route / explicit-v2-
        # prefix predicates plus the shared openai guard (full message-name
        # fallback — cohere's transform is the inherited GPT one).
        "cohere": cohere_unsupported_request_shapes,
        "cohere_chat": cohere_unsupported_request_shapes,
        # mistral: own name-matrix arm (tool-role names kept by v1; image
        # branch forwards every name) + the shared openai guard with
        # skip_name_fallback; deliberately NO explicit stream:false arm
        # (v1's map only copies stream=True).
        "mistral": mistral_unsupported_request_shapes,
        # watsonx: the shared openai guard (full name fallback); no
        # stream:false arm — the wire ALWAYS carries the stream key.
        "watsonx": watsonx_unsupported_request_shapes,
        # sagemaker_chat: explicit stream:false + the shared openai guard
        # (full name fallback).
        "sagemaker_chat": sagemaker_chat_unsupported_request_shapes,
        # groq: explicit stream:false + the shared openai guard (full name
        # fallback).
        "groq": groq_unsupported_request_shapes,
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
    return parser(raw_response, request).bind(
        lambda response: _serialized_body(response, deps, dialect)
    )


def _serialized_body(
    response: ChatResponse, deps: TranslationDeps, dialect: ResponseDialect
) -> TranslateResult:
    body = serialize_response(response, deps, dialect)
    if isinstance(body, TranslationError):
        return Error(body)
    return Ok(body)


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
            body = serialize_response(chat_response, deps, response_dialect(provider))
            if isinstance(body, TranslationError):
                return Error(ExecuteError.of_translation(body))
            return Ok(body)
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
