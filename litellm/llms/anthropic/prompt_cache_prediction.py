from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, TypeAdapter, ValidationError

import litellm
from litellm.llms.anthropic.common_utils import AnthropicModelInfo, is_anthropic_oauth_key
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import DEFAULT_ANTHROPIC_API_VERSION
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import ModelResponse

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
    content: str | Annotated[tuple[_Block, ...], Field(strict=False)]

    def blocks(self) -> tuple[_Text | _ToolUse | _ToolResult, ...]:
        return (_Text(type="text", text=self.content),) if isinstance(self.content, str) else tuple(self.content)


class _Tool(_StrictModel):
    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: Mapping[str, JsonValue]
    type: Literal["custom"] | None = None


class _Request(_StrictModel):
    messages: tuple[_Message, ...] = Field(min_length=1, strict=False)
    system: str | Annotated[tuple[_ResultText, ...], Field(strict=False)] | None = None
    tools: Annotated[tuple[_Tool, ...], Field(strict=False)] | None = None
    model: str | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    temperature: float | int | None = None
    top_p: float | int | None = None
    top_k: int | None = None
    stop_sequences: Annotated[tuple[str, ...], Field(strict=False)] | None = None
    metadata: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class PromptPrefix:
    prefix_body: Mapping[str, JsonValue]
    fingerprint: str
    fingerprints: tuple[str, ...]
    ttl_seconds: int


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _next_digest(previous: str, boundary: tuple[int, str, Mapping[str, JsonValue]]) -> str:
    return _digest((previous, boundary))


def parse_prompt(body: Mapping[str, JsonValue]) -> PromptPrefix | None:
    try:
        request: Final = _Request.model_validate(body)
        blocks: Final = tuple(message.blocks() for message in request.messages)
    except ValidationError:
        return None
    markers: Final = tuple(
        (message_index, block_index, block.cache_control)
        for message_index, message_blocks in enumerate(blocks)
        for block_index, block in enumerate(message_blocks)
        if block.cache_control is not None
    )
    if len(markers) != 1:
        return None
    message_end, block_end, marker = markers[0]
    normalized: Final = _JSON_OBJECT.validate_python(request.model_dump(mode="json", exclude_none=True))
    context: Final = MappingProxyType({key: normalized[key] for key in ("system", "tools") if key in normalized})
    boundaries: Final = tuple(
        (
            message_index,
            request.messages[message_index].role,
            _JSON_OBJECT.validate_python(
                block.model_dump(mode="json", exclude=MappingProxyType({"cache_control": True}), exclude_none=True)
            ),
        )
        for message_index, message_blocks in enumerate(blocks[: message_end + 1])
        for block_index, block in enumerate(message_blocks)
        if message_index < message_end or block_index <= block_end
    )
    hashes: Final = tuple(
        accumulate(boundaries, _next_digest, initial=_digest((_JSON_OBJECT.validate_python(context), marker.ttl)))
    )[1:]
    prefix_messages: Final = tuple(
        _Message(
            role=request.messages[message_index].role,
            content=tuple(
                block
                for block_index, block in enumerate(message_blocks)
                if message_index < message_end or block_index <= block_end
            ),
        )
        for message_index, message_blocks in enumerate(blocks[: message_end + 1])
    )
    return PromptPrefix(
        prefix_body=MappingProxyType(
            _JSON_OBJECT.validate_python(
                _Request(messages=prefix_messages, system=request.system, tools=request.tools).model_dump(
                    mode="json", exclude_none=True
                )
            )
        ),
        fingerprint=hashes[-1],
        fingerprints=tuple(reversed(hashes[-20:])),
        ttl_seconds=3600 if marker.ttl == "1h" else 300,
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


class _CountResult(BaseModel):
    input_tokens: Annotated[StrictInt, Field(ge=0)]


class TokenCounter(Protocol):
    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None: ...


def _count_objects(
    values: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:  # mutable-ok: the existing provider count API requires JSON lists/dicts
    return [dict(value) for value in values]  # mutable-ok: serialize read-only inputs at the provider API boundary


async def count_prompt_tokens(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
    native: Final = _CountBody.model_validate(body)
    try:
        result: Final = _CountResult.model_validate(
            await _counter.handle_count_tokens_request(
                model=model,
                messages=_count_objects(native.messages),
                tools=_count_objects(native.tools) if native.tools is not None else None,
                system=native.system,
                api_key=api_key,
                timeout=15.0,
            )
        )
    except Exception:  # noqa: BLE001  # provider/count validation failures are unavailable estimates, not zero tokens
        return None
    return result.input_tokens


@dataclass(frozen=True, slots=True)
class NativePredictionTarget:
    model: str
    api_key: str


@dataclass(frozen=True, slots=True)
class UnsupportedPredictionTarget:
    reason: Literal[
        "unsupported_deployment_configuration",
        "unsupported_provider_endpoint",
        "unsupported_provider",
        "unsupported_provider_credentials",
    ]


def resolve_prediction_target(params: LiteLLM_Params) -> NativePredictionTarget | UnsupportedPredictionTarget:
    configured_options: Final = frozenset(params.model_dump(exclude_defaults=True, exclude_none=True))
    if configured_options - _DEPLOYMENT_OPTIONS:
        return UnsupportedPredictionTarget("unsupported_deployment_configuration")
    api_base: Final = AnthropicModelInfo.get_api_base(params.api_base)
    if api_base not in ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"):
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
    return NativePredictionTarget(model=model, api_key=api_key)


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
