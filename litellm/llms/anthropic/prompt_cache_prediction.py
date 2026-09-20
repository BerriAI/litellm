from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate, groupby
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, TypeAdapter, ValidationError

import litellm
from litellm.llms.anthropic.common_utils import AnthropicModelInfo, is_anthropic_oauth_key
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import COUNT_TOKEN_OPTION_NAMES
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    DEFAULT_ANTHROPIC_API_VERSION,
    AnthropicMessagesConfig,
)
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import ModelResponse
from litellm.utils import supports_thinking_cache_preservation

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_HEADERS: Final = TypeAdapter(dict[str, str])
_counter: Final = AnthropicCountTokensHandler()


_NATIVE_HEADERS: Final = frozenset(
    (
        "host",
        "accept",
        "accept-encoding",
        "connection",
        "user-agent",
        "content-length",
        "content-type",
        "x-api-key",
        "anthropic-version",
    )
)

_DEPLOYMENT_OPTIONS: Final = frozenset(
    {
        "model",
        "api_key",
        "api_base",
        "custom_llm_provider",
        "rpm",
        "tpm",
        "timeout",
        "stream_timeout",
        "max_retries",
        "num_retries",
        "max_parallel_requests",
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
    }
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _CacheControl(_StrictModel):
    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] = "5m"


class _Text(_StrictModel):
    type: Literal["text"]
    text: str = Field(min_length=1, pattern=r"\S")
    cache_control: _CacheControl | None = None


class _ToolUse(_StrictModel):
    type: Literal["tool_use"]
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: Mapping[str, JsonValue]
    cache_control: _CacheControl | None = None


class _ResultText(_StrictModel):
    type: Literal["text"]
    text: str


class _ToolResult(_StrictModel):
    type: Literal["tool_result"]
    tool_use_id: str = Field(min_length=1)
    content: str | Annotated[tuple[_ResultText, ...], Field(strict=False)]
    is_error: bool | None = None
    cache_control: _CacheControl | None = None


_Block: TypeAlias = Annotated[_Text | _ToolUse | _ToolResult, Field(discriminator="type")]


class _Message(_StrictModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(min_length=1, pattern=r"\S")] | Annotated[tuple[_Block, ...], Field(strict=False)]


class _Tool(_StrictModel):
    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: Mapping[str, JsonValue]
    type: Literal["custom"] | None = None


class _RequestOptions(_StrictModel):
    model: str | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    temperature: float | int | None = None
    top_p: float | int | None = None
    top_k: int | None = None
    stop_sequences: Annotated[tuple[str, ...], Field(strict=False)] | None = None
    metadata: Mapping[str, JsonValue] | None = None


class _Request(_RequestOptions):
    messages: tuple[_Message, ...] = Field(min_length=1, strict=False)
    system: str | Annotated[tuple[_ResultText, ...], Field(strict=False)] | None = None
    tools: Annotated[tuple[_Tool, ...], Field(strict=False)] | None = None


class _Thinking(_StrictModel):
    type: Literal["thinking"]
    thinking: str
    signature: str = Field(min_length=1)


_PlanBlock: TypeAlias = Annotated[_Text | _ToolUse | _ToolResult | _Thinking, Field(discriminator="type")]


class _PlanMessage(_StrictModel):
    role: Literal["user", "assistant", "system"]
    content: str | Annotated[tuple[_PlanBlock, ...], Field(strict=False)]


class _PlanTool(_Tool):
    cache_control: _CacheControl | None = None


class _PlanRequest(_RequestOptions):
    messages: tuple[_PlanMessage, ...] = Field(min_length=1, strict=False)
    system: str | Annotated[tuple[_Text, ...], Field(strict=False)] | None = None
    tools: Annotated[tuple[_PlanTool, ...], Field(strict=False)] | None = None
    cache_control: _CacheControl | None = None
    thinking: Mapping[str, JsonValue] | None = None
    tool_choice: Mapping[str, JsonValue] | None = None
    output_config: Mapping[str, JsonValue] | None = None
    speed: Literal["fast", "standard"] | None = None
    service_tier: Literal["auto", "standard_only"] | None = None


@dataclass(frozen=True, slots=True)
class CacheBoundary:
    fingerprint: str
    prefix_body: Mapping[str, JsonValue] = field(repr=False)
    ttl_seconds: int
    lookback_fingerprints: tuple[str, ...]
    content_fingerprint: str = ""
    lookback_content_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptCachePlan:
    full_body: Mapping[str, JsonValue] = field(repr=False)
    breakpoints: tuple[CacheBoundary, ...]


@dataclass(frozen=True, slots=True)
class UnsupportedCachePlan:
    reason: Literal[
        "unsupported_prompt_shape",
        "conflicting_cache_ttl",
        "too_many_cache_breakpoints",
        "invalid_cache_ttl_order",
        "unsupported_thinking_cache_semantics",
        "token_count_unavailable",
        "inconsistent_prefix_token_count",
    ]


@dataclass(frozen=True, slots=True)
class CountedBreakpoint:
    fingerprint: str
    ttl_seconds: int
    prefix_tokens: int
    lookback_fingerprints: tuple[str, ...]
    content_fingerprint: str = ""
    lookback_content_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CountedPromptCachePlan:
    total_tokens: int
    breakpoints: tuple[CountedBreakpoint, ...]


@dataclass(frozen=True, slots=True)
class _Position:
    section: Literal["tools", "system", "messages"]
    message_index: int
    role: str
    block: Mapping[str, JsonValue]
    marker: _CacheControl | None


def _content_blocks(content: JsonValue) -> tuple[Mapping[str, JsonValue], ...]:
    if isinstance(content, str):
        return (MappingProxyType({"type": "text", "text": content}),)
    return tuple(_JSON_OBJECT.validate_python(block) for block in content) if isinstance(content, list) else ()


def _position(
    section: Literal["tools", "system", "messages"],
    message_index: int,
    role: str,
    block: Mapping[str, JsonValue],
) -> _Position:
    control: Final = block.get("cache_control")
    return _Position(
        section,
        message_index,
        role,
        MappingProxyType({key: value for key, value in block.items() if key != "cache_control"}),
        _CacheControl.model_validate(control) if control is not None else None,
    )


def _positions(body: Mapping[str, JsonValue]) -> tuple[_Position, ...]:
    tools: Final = body.get("tools")
    messages: Final = body.get("messages")
    return (
        *tuple(
            _position("tools", -1, "", _JSON_OBJECT.validate_python(tool))
            for tool in (tools if isinstance(tools, list) else ())
        ),
        *tuple(_position("system", -1, "", block) for block in _content_blocks(body.get("system"))),
        *tuple(
            _position("messages", message_index, str(message.get("role")), block)
            for message_index, raw_message in enumerate(messages if isinstance(messages, list) else ())
            for message in (_JSON_OBJECT.validate_python(raw_message),)
            for block in _content_blocks(message.get("content"))
        ),
    )


def _prefix_body(
    body: Mapping[str, JsonValue],
    positions: tuple[_Position, ...],
    last_index: int,
) -> Mapping[str, JsonValue]:
    prefix: Final = positions[: last_index + 1]
    sections: Final = MappingProxyType(
        {
            section: _count_objects(tuple(position.block for position in prefix if position.section == section))
            for section in ("tools", "system")
            if any(position.section == section for position in prefix)
        }
    )
    messages: Final = tuple(
        MappingProxyType(
            _JSON_OBJECT.validate_python(
                MappingProxyType(
                    {"role": group[0].role, "content": _count_objects(tuple(position.block for position in group))}
                )
            )
        )
        for _, values in groupby(
            (position for position in prefix if position.section == "messages"),
            key=lambda position: position.message_index,
        )
        for group in (tuple(values),)
    )
    return MappingProxyType(
        _JSON_OBJECT.validate_python(
            MappingProxyType(
                {
                    **MappingProxyType({key: body[key] for key in COUNT_TOKEN_OPTION_NAMES if key in body}),
                    **sections,
                    "messages": _count_objects(messages),
                }
            )
        )
    )


def _position_group(position: _Position, index: int) -> tuple[str, int, str | int]:
    block_type: Final = position.block.get("type")
    return (
        position.section,
        position.message_index,
        block_type if isinstance(block_type, str) and block_type in ("tool_use", "tool_result") else index,
    )


def _chain_digest(previous: str, current: str) -> str:
    return _digest((previous, current))


def _cacheable_position(position: _Position) -> bool:
    block_type: Final = position.block.get("type")
    if block_type == "thinking":
        return False
    text: Final = position.block.get("text")
    return block_type != "text" or (isinstance(text, str) and bool(text.strip()))


def _entry_fingerprint(fingerprint: str, ttl_seconds: int) -> str:
    return _digest(("native-cache-prefix-v2", fingerprint, ttl_seconds))


def parse_cache_plan(body: Mapping[str, JsonValue]) -> PromptCachePlan | UnsupportedCachePlan:
    try:
        request: Final = _PlanRequest.model_validate(body)
        positions: Final = _positions(body)
    except ValidationError:
        return UnsupportedCachePlan("unsupported_prompt_shape")
    explicit: Final = tuple(
        (index, position.marker) for index, position in enumerate(positions) if position.marker is not None
    )
    automatic_index: Final = next(
        (index for index in reversed(range(len(positions))) if _cacheable_position(positions[index])), None
    )
    automatic_existing: Final = next((marker for index, marker in explicit if index == automatic_index), None)
    if (
        request.cache_control is not None
        and automatic_existing is not None
        and automatic_existing != request.cache_control
    ):
        return UnsupportedCachePlan("conflicting_cache_ttl")
    automatic: Final = (
        ((automatic_index, request.cache_control),)
        if (request.cache_control is not None and automatic_index is not None and automatic_existing is None)
        else ()
    )
    markers: Final = tuple(sorted((*explicit, *automatic), key=lambda value: value[0]))
    if len(markers) > 4:
        return UnsupportedCachePlan("too_many_cache_breakpoints")
    ttls: Final = tuple(3600 if marker.ttl == "1h" else 300 for _, marker in markers)
    if any(first < second for first, second in zip(ttls, ttls[1:])):
        return UnsupportedCachePlan("invalid_cache_ttl_order")
    settings: Final = MappingProxyType(
        {
            key: body[key]
            for key in ("thinking", "output_config", "speed")
            if key in body and not (key == "speed" and body[key] == "standard")
        }
    )
    hashes: Final = tuple(
        accumulate(
            (
                _digest(
                    (
                        position.section,
                        position.message_index,
                        position.role,
                        position.block,
                        body.get("tool_choice") if position.section == "messages" else None,
                    )
                )
                for position in positions
            ),
            _chain_digest,
            initial=_digest(settings),
        )
    )[1:]
    groups: Final = tuple(
        tuple(index for index, _ in values)
        for _, values in groupby(
            enumerate(positions),
            key=lambda item: _position_group(item[1], item[0]),
        )
    )
    return PromptCachePlan(
        full_body=MappingProxyType(dict(body)),
        breakpoints=tuple(
            CacheBoundary(
                fingerprint=_entry_fingerprint(hashes[index], ttl),
                prefix_body=_prefix_body(body, positions, index),
                ttl_seconds=ttl,
                lookback_fingerprints=tuple(
                    _entry_fingerprint(hashes[earlier], ttl)
                    for group in reversed(tuple(group for group in groups if group[0] <= index)[-20:])
                    for earlier in reversed(group)
                    if earlier <= index
                ),
                content_fingerprint=hashes[index],
                lookback_content_fingerprints=tuple(
                    hashes[earlier]
                    for group in reversed(tuple(group for group in groups if group[0] <= index)[-20:])
                    for earlier in reversed(group)
                    if earlier <= index
                ),
            )
            for (index, _), ttl in zip(markers, ttls)
        ),
    )


@dataclass(frozen=True, slots=True)
class PromptPrefix:
    prefix_body: Mapping[str, JsonValue]
    fingerprint: str
    fingerprints: tuple[str, ...]
    ttl_seconds: int


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, default=_json_object, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _json_object(value: object) -> dict[str, JsonValue]:  # mutable-ok: JSON serialization requires a dictionary
    return _JSON_OBJECT.validate_python(value)


def parse_prompt(body: Mapping[str, JsonValue]) -> PromptPrefix | None:
    try:
        _Request.model_validate(body)
    except ValidationError:
        return None
    plan: Final = parse_cache_plan(body)
    if isinstance(plan, UnsupportedCachePlan) or len(plan.breakpoints) != 1:
        return None
    prefix: Final = plan.breakpoints[0]
    return PromptPrefix(
        prefix_body=prefix.prefix_body,
        fingerprint=prefix.fingerprint,
        fingerprints=prefix.lookback_fingerprints,
        ttl_seconds=prefix.ttl_seconds,
    )


def cache_scope(
    caller_key_hash: str,
    deployment_id: str,
    provider_key: str,
    model: str,
    anthropic_version: str = DEFAULT_ANTHROPIC_API_VERSION,
) -> str:
    return _digest((caller_key_hash, deployment_id, provider_key, model, anthropic_version))


class _TTLUsage(BaseModel):
    model_config = ConfigDict(strict=True)
    ephemeral_5m_input_tokens: int = Field(default=0, ge=0)
    ephemeral_1h_input_tokens: int = Field(default=0, ge=0)


class _CacheUsage(BaseModel):
    model_config = ConfigDict(strict=True)
    cached_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    cache_creation_token_details: _TTLUsage | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt_tokens: int = Field(ge=0)
    prompt_tokens_details: _CacheUsage


class _Choice(BaseModel):
    finish_reason: str = Field(min_length=1)


class _Response(BaseModel):
    model_config = ConfigDict(strict=True)
    model: str
    usage: _Usage
    choices: tuple[_Choice, ...] = Field(min_length=1, strict=False)


class _CountBody(BaseModel):
    messages: Sequence[Mapping[str, JsonValue]]
    tools: Sequence[Mapping[str, JsonValue]] | None = None
    system: str | Sequence[Mapping[str, JsonValue]] | None = None
    thinking: Mapping[str, JsonValue] | None = None
    tool_choice: Mapping[str, JsonValue] | None = None
    output_config: Mapping[str, JsonValue] | None = None


class _CountResult(BaseModel):
    input_tokens: Annotated[StrictInt, Field(ge=0)]


class TokenCounter(Protocol):
    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None: ...


def _count_objects(
    values: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:  # mutable-ok: the existing provider count API requires JSON lists/dicts
    return [dict(value) for value in values]  # mutable-ok: serialize read-only inputs at the provider API boundary


def _messages_url(model: str, api_key: str, api_base: str | None) -> str:
    return AnthropicMessagesConfig().get_complete_url(  # pyright: ignore[reportUnknownMemberType]  # canonical native URL owner takes legacy JSON arguments
        api_base=api_base,
        api_key=api_key,
        model=model,
        optional_params=_JSON_OBJECT.validate_python(MappingProxyType({})),
        litellm_params=_JSON_OBJECT.validate_python(MappingProxyType({})),
    )


async def count_prompt_tokens(
    model: str,
    api_key: str,
    body: Mapping[str, JsonValue],
    api_base: str | None = None,
) -> int | None:
    try:
        native: Final = _CountBody.model_validate(body)
        count_url: Final = _messages_url(model, api_key, api_base) + "/count_tokens"
        result: Final = _CountResult.model_validate(
            await _counter.handle_count_tokens_request(
                model=model,
                messages=_count_objects(native.messages),
                tools=_count_objects(native.tools) if native.tools is not None else None,
                system=_JSON_OBJECT.validate_python(MappingProxyType({"system": native.system}))["system"],
                api_key=api_key,
                api_base=count_url,
                optional_params=_JSON_OBJECT.validate_python(
                    MappingProxyType({key: body[key] for key in COUNT_TOKEN_OPTION_NAMES if key in body})
                ),
                timeout=15.0,
            )
        )
    except Exception:  # noqa: BLE001  # provider/count validation failures are unavailable estimates, not zero tokens
        return None
    return result.input_tokens


async def count_cache_plan(
    model: str,
    api_key: str,
    plan: PromptCachePlan,
    token_counter: TokenCounter = count_prompt_tokens,
) -> CountedPromptCachePlan | UnsupportedCachePlan:
    if any(position.block.get("type") == "thinking" for position in _positions(plan.full_body)):
        if not supports_thinking_cache_preservation(model, "anthropic"):
            return UnsupportedCachePlan("unsupported_thinking_cache_semantics")
    total: Final = await token_counter(model, api_key, plan.full_body)
    if total is None:
        return UnsupportedCachePlan("token_count_unavailable")
    counts: Final = tuple(
        await asyncio.gather(*(token_counter(model, api_key, marker.prefix_body) for marker in plan.breakpoints))
    )
    if any(value is None for value in counts):
        return UnsupportedCachePlan("token_count_unavailable")
    known: Final = tuple(value for value in counts if value is not None)
    if any(value < 0 for value in (total, *known)) or any(
        first > second for first, second in zip(known, (*known[1:], total))
    ):
        return UnsupportedCachePlan("inconsistent_prefix_token_count")
    return CountedPromptCachePlan(
        total,
        tuple(
            CountedBreakpoint(
                marker.fingerprint,
                marker.ttl_seconds,
                count,
                marker.lookback_fingerprints,
                marker.content_fingerprint,
                marker.lookback_content_fingerprints,
            )
            for marker, count in zip(plan.breakpoints, known)
        ),
    )


@dataclass(frozen=True, slots=True)
class NativePredictionTarget:
    model: str
    api_key: str = field(repr=False)
    api_base: str | None = None


def supported_baseline_recipient(target: NativePredictionTarget, wire: httpx.Request) -> bool:
    return wire.headers.get("x-api-key") == target.api_key and wire.url == httpx.URL(
        _messages_url(target.model, target.api_key, target.api_base)
    )


@dataclass(frozen=True, slots=True)
class UnsupportedPredictionTarget:
    reason: Literal[
        "unsupported_deployment_configuration",
        "unsupported_provider_endpoint",
        "unsupported_provider",
        "unsupported_provider_credentials",
    ]


def resolve_prediction_target(params: LiteLLM_Params) -> NativePredictionTarget | UnsupportedPredictionTarget:
    return _resolve_prediction_target(params, allow_configured_endpoint=False)


def resolve_baseline_prediction_target(params: LiteLLM_Params) -> NativePredictionTarget | UnsupportedPredictionTarget:
    return _resolve_prediction_target(params, allow_configured_endpoint=True)


def _resolve_prediction_target(
    params: LiteLLM_Params,
    *,
    allow_configured_endpoint: bool,
) -> NativePredictionTarget | UnsupportedPredictionTarget:
    configured_options: Final = frozenset(params.model_dump(exclude_defaults=True, exclude_none=True))
    if configured_options - _DEPLOYMENT_OPTIONS:
        return UnsupportedPredictionTarget("unsupported_deployment_configuration")
    api_base: Final = AnthropicModelInfo.get_api_base(params.api_base)
    if not allow_configured_endpoint and api_base not in (
        "https://api.anthropic.com",
        "https://api.anthropic.com/v1/messages",
    ):
        return UnsupportedPredictionTarget("unsupported_provider_endpoint")
    try:
        model, provider, _, _ = litellm.get_llm_provider(
            model=params.model, custom_llm_provider=params.custom_llm_provider
        )
    except Exception:  # noqa: BLE001  # the shared provider resolver raises for unknown deployments
        return UnsupportedPredictionTarget("unsupported_provider")
    if provider != "anthropic":
        return UnsupportedPredictionTarget("unsupported_provider")
    api_key: Final = AnthropicModelInfo.get_api_key(params.api_key)
    if api_key is None or not _supported_provider_key(api_key):
        return UnsupportedPredictionTarget("unsupported_provider_credentials")
    return NativePredictionTarget(model=model, api_key=api_key, api_base=api_base)


def _supported_provider_key(api_key: str) -> bool:
    return bool(api_key) and not is_anthropic_oauth_key(api_key)


def supported_prediction_headers(headers: Mapping[str, str]) -> bool:
    return all(
        name.lower() != "anthropic-beta"
        and (name.lower() != "anthropic-version" or value == DEFAULT_ANTHROPIC_API_VERSION)
        for name, value in headers.items()
    )


@dataclass(frozen=True, slots=True)
class ObservedCachePrefix:
    prefix: PromptPrefix
    scope: str
    cached_tokens: int
    cache_creation_tokens: int


def parse_observed_cache(
    wire: httpx.Request, response_obj: ModelResponse, caller_key_hash: str, deployment_id: str
) -> ObservedCachePrefix | None:
    try:
        response: Final = _Response.model_validate(response_obj, from_attributes=True)
        body: Final = _JSON_OBJECT.validate_json(wire.content)
        headers: Final = _HEADERS.validate_python(wire.headers)
    except (ValidationError, RuntimeError, httpx.RequestNotRead):
        return None
    if (
        wire.url.scheme != "https"
        or wire.url.host != "api.anthropic.com"
        or wire.url.path != "/v1/messages"
        or wire.url.query
        or wire.url.port not in (None, 443)
    ):
        return None
    if (
        frozenset(headers) - _NATIVE_HEADERS
        or not supported_prediction_headers(headers)
        or headers.get("anthropic-version") != DEFAULT_ANTHROPIC_API_VERSION
    ):
        return None
    provider_key: Final = headers.get("x-api-key", "")
    model: Final = body.get("model")
    if not _supported_provider_key(provider_key) or not isinstance(model, str) or model != response.model:
        return None
    prefix: Final = parse_prompt(body)
    if prefix is None:
        return None
    usage: Final = response.usage.prompt_tokens_details
    cache_tokens: Final = usage.cached_tokens + usage.cache_creation_tokens
    if cache_tokens <= 0 or cache_tokens > response.usage.prompt_tokens:
        return None
    split: Final = usage.cache_creation_token_details
    if usage.cache_creation_tokens and split is None:
        return None
    if split is not None and (
        split.ephemeral_5m_input_tokens + split.ephemeral_1h_input_tokens != usage.cache_creation_tokens
        or (prefix.ttl_seconds == 300 and split.ephemeral_1h_input_tokens > 0)
        or (prefix.ttl_seconds == 3600 and split.ephemeral_5m_input_tokens > 0)
    ):
        return None
    return ObservedCachePrefix(
        prefix=prefix,
        scope=cache_scope(caller_key_hash, deployment_id, provider_key, model),
        cached_tokens=cache_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
    )
