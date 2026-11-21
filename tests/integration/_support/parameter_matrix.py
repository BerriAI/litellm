import itertools
import re
import typing
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import resolve_model_and_provider_for_metadata
from litellm.types.llms.anthropic import AnthropicMessagesRequestOptionalParams
from litellm.types.llms.openai import ResponsesAPIRequestParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

CLIENT_SETTINGS: Final = frozenset(
    {
        "api_base",
        "api_key",
        "api_version",
        "base_url",
        "default_headers",
        "deployment_id",
        "max_retries",
        "organization",
        "request_timeout",
        "timeout",
    }
)


class Endpoint(Enum):
    CHAT = "/v1/chat/completions"
    RESPONSES = "/v1/responses"
    MESSAGES = "/v1/messages"


ATTRIBUTES: Final[Mapping[Endpoint, tuple[str, ...]]] = MappingProxyType(
    {
        Endpoint.CHAT: tuple(sorted(set(litellm.OPENAI_CHAT_COMPLETION_PARAMS) - CLIENT_SETTINGS)),
        Endpoint.RESPONSES: tuple(
            sorted(name for name in typing.get_type_hints(ResponsesAPIRequestParams) if name not in ("input", "model"))
        ),
        Endpoint.MESSAGES: tuple(sorted(typing.get_type_hints(AnthropicMessagesRequestOptionalParams))),
    }
)

_PROVIDERS: Final = frozenset(provider.value for provider in LlmProviders)
_MODES: Final = frozenset({"chat", "responses"})

_RESPONSES_BRIDGE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tool_choice": "tool_choice",
        "tools": "tools",
        "top_p": "top_p",
        "user": "user",
        "temperature": "temperature",
        "parallel_tool_calls": "parallel_tool_calls",
        "max_output_tokens": "max_tokens",
        "stream": "stream",
        "service_tier": "service_tier",
        "safety_identifier": "safety_identifier",
        "text": "response_format",
        "reasoning": "reasoning_effort",
        "context_management": "context_management",
    }
)
_RESPONSES_BRIDGE_ALWAYS: Final = frozenset({"instructions"})
BRIDGE_SESSION_STATE: Final = frozenset({"previous_response_id", "store"})
_MESSAGES_ADAPTER: Final[Mapping[str, str]] = MappingProxyType(
    {
        "metadata": "user",
        "stop_sequences": "stop",
        "output_format": "response_format",
        "output_config": "reasoning_effort",
    }
)
_MESSAGES_ADAPTER_ALWAYS: Final = frozenset({"system"})
_STAND_INS: Final[Mapping[str, str]] = MappingProxyType(
    {"max_tokens": "max_completion_tokens", "thinking": "reasoning_effort"}
)

_SPECIALISED: Final = re.compile(
    r"ft:|guard|embed|whisper|tts|audio|realtime|image|search|moderation|transcribe|computer-use|codex|deep-research|instruct",
    re.IGNORECASE,
)
_CROSS_REGION: Final = re.compile(r"(bedrock/)?(us|global)\.")
_BARE_BEDROCK: Final = re.compile(r"(bedrock/)?(anthropic|openai|meta|amazon|mistral|deepseek|cohere)\.")


class _Listing(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    mode: str | None = None
    litellm_provider: str | None = None


@dataclass(frozen=True, slots=True)
class Model:
    key: str
    name: str
    provider: str


@dataclass(frozen=True, slots=True)
class _Member:
    model: Model
    supported: frozenset[str]
    native: bool


@dataclass(frozen=True, slots=True, eq=False)
class Group:
    provider: str
    endpoint: Endpoint
    supported: frozenset[str]
    native: bool
    models: tuple[Model, ...]

    @property
    def id(self) -> str:
        bits: Final = "".join("1" if name in self.supported else "0" for name in ATTRIBUTES[self.endpoint])
        pattern: Final = f"{int(bits, 2):0{(len(bits) + 3) // 4}x}"
        return f"{self.provider}-{pattern}" if self.native else f"{self.provider}-{pattern}-bridged"


@dataclass(frozen=True, slots=True)
class Cell:
    group: Group
    attribute: str
    drop_params: bool

    @property
    def id(self) -> str:
        setting: Final = "on" if self.drop_params else "off"
        return f"{self.group.endpoint.name.lower()}-{self.group.id}-{self.attribute}-drop_params_{setting}"


@dataclass(frozen=True, slots=True)
class Sample:
    value: JsonValue
    companions: Mapping[str, JsonValue] = field(default_factory=dict)
    alternatives: tuple[JsonValue, ...] = ()
    otherwise: tuple["Sample", ...] = ()


def _listed_route(key: str, listing: _Listing) -> Model | None:
    provider: Final = listing.litellm_provider
    if provider is None or provider not in _PROVIDERS:
        return None
    return Model(key=key, name=key.removeprefix(f"{provider}/"), provider=provider)


def _route(key: str, listing: _Listing) -> Model | None:
    try:
        name, provider = resolve_model_and_provider_for_metadata(key)
    except litellm.BadRequestError:
        return _listed_route(key, listing)
    return Model(key=key, name=name, provider=provider)


@lru_cache(maxsize=1)
def _routes() -> tuple[tuple[str, Model | None], ...]:
    listings: Final = TypeAdapter(dict[str, _Listing]).validate_python(litellm.model_cost)
    return tuple((key, _route(key, listing)) for key, listing in sorted(listings.items()) if listing.mode in _MODES)


def models() -> tuple[Model, ...]:
    return tuple(model for _, model in _routes() if model is not None)


def unroutable() -> tuple[str, ...]:
    return tuple(key for key, model in _routes() if model is None)


def _has(chat: frozenset[str], name: str) -> bool:
    return name in chat or _STAND_INS.get(name) in chat


def chat_name(group: "Group", attribute: str) -> str | None:
    if group.native:
        return attribute if group.endpoint is Endpoint.CHAT else None
    if group.endpoint is Endpoint.RESPONSES:
        return _RESPONSES_BRIDGE.get(attribute, attribute)
    return _MESSAGES_ADAPTER.get(attribute, attribute)


def _chat(model: Model) -> frozenset[str]:
    return frozenset(litellm.get_supported_openai_params(model=model.name, custom_llm_provider=model.provider) or ())


def _responses(model: Model) -> tuple[frozenset[str], bool]:
    config: Final = (
        ProviderConfigManager.get_provider_responses_api_config(provider=LlmProviders(model.provider), model=model.name)
        if model.provider in _PROVIDERS
        else None
    )
    if config is not None:
        return frozenset(config.get_supported_openai_params(model.name)), True
    chat: Final = _chat(model)
    bridged: Final = frozenset(name for name, target in _RESPONSES_BRIDGE.items() if _has(chat, target))
    return bridged | _RESPONSES_BRIDGE_ALWAYS, False


def _messages(model: Model) -> tuple[frozenset[str], bool]:
    config: Final = (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model.name, provider=LlmProviders(model.provider)
        )
        if model.provider in _PROVIDERS
        else None
    )
    if config is not None:
        return frozenset(config.get_supported_anthropic_messages_params(model.name)), True
    chat: Final = _chat(model)
    adapted: Final = frozenset(
        name for name in ATTRIBUTES[Endpoint.MESSAGES] if _has(chat, _MESSAGES_ADAPTER.get(name, name))
    )
    return adapted | _MESSAGES_ADAPTER_ALWAYS, False


def _support(model: Model, endpoint: Endpoint) -> tuple[frozenset[str], bool]:
    match endpoint:
        case Endpoint.CHAT:
            return _chat(model), True
        case Endpoint.RESPONSES:
            return _responses(model)
        case Endpoint.MESSAGES:
            return _messages(model)
        case _:
            assert_never(endpoint)


def _member(model: Model, endpoint: Endpoint) -> _Member:
    names, native = _support(model, endpoint)
    return _Member(model=model, supported=names & frozenset(ATTRIBUTES[endpoint]), native=native)


def _pattern(member: _Member) -> tuple[str, tuple[str, ...], bool]:
    return (member.model.provider, tuple(sorted(member.supported)), member.native)


def _rank(model: Model) -> tuple[bool, int, int, str]:
    region: Final = 0 if _CROSS_REGION.match(model.key) else 2 if _BARE_BEDROCK.match(model.key) else 1
    return (bool(_SPECIALISED.search(model.key)), region, len(model.key), model.key)


def _group(endpoint: Endpoint, members: tuple[_Member, ...]) -> Group:
    first: Final = members[0]
    return Group(
        provider=first.model.provider,
        endpoint=endpoint,
        supported=first.supported,
        native=first.native,
        models=tuple(sorted((member.model for member in members), key=_rank)),
    )


@lru_cache(maxsize=None)
def groups(endpoint: Endpoint) -> tuple[Group, ...]:
    members: Final = sorted((_member(model, endpoint) for model in models()), key=_pattern)
    return tuple(_group(endpoint, tuple(batch)) for _, batch in itertools.groupby(members, key=_pattern))


def _cells(endpoint: Endpoint) -> Iterator[Cell]:
    for group, attribute, drop_params in itertools.product(groups(endpoint), ATTRIBUTES[endpoint], (True, False)):
        yield Cell(group=group, attribute=attribute, drop_params=drop_params)


def cells() -> tuple[Cell, ...]:
    return tuple(itertools.chain.from_iterable(_cells(endpoint) for endpoint in Endpoint))


_FUNCTION: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
_CHAT_TOOL: Final[dict[str, JsonValue]] = {"type": "function", "function": _FUNCTION}
_RESPONSES_TOOL: Final[dict[str, JsonValue]] = {"type": "function", **_FUNCTION}
_MESSAGES_TOOL: Final[dict[str, JsonValue]] = {
    "name": _FUNCTION["name"],
    "description": _FUNCTION["description"],
    "input_schema": _FUNCTION["parameters"],
}
_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"ok": {"type": "string"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_THINKING: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024}
_CHAT_REASONING_OFF: Final[dict[str, JsonValue]] = {"reasoning_effort": "none"}
_RESPONSES_REASONING_OFF: Final[dict[str, JsonValue]] = {"reasoning": {"effort": "none"}}
_MESSAGES_REASONING_OFF: Final[dict[str, JsonValue]] = {"thinking": {"type": "disabled"}}

SAMPLES: Final[Mapping[Endpoint, Mapping[str, Sample]]] = MappingProxyType(
    {
        Endpoint.CHAT: MappingProxyType(
            {
                "audio": Sample(
                    {"voice": "alloy", "format": "wav"},
                    {"modalities": ["text", "audio"]},
                    ({"voice": "alloy", "format": "pcm16"},),
                ),
                "extra_headers": Sample({"x-matrix": "1"}),
                "frequency_penalty": Sample(0.1, alternatives=(0.5,), otherwise=(Sample(0.1, _CHAT_REASONING_OFF),)),
                "function_call": Sample("auto", {"functions": [_FUNCTION]}),
                "functions": Sample([_FUNCTION]),
                "include_server_side_tool_invocations": Sample(True),
                "logit_bias": Sample({"1234": 1}),
                "logprobs": Sample(True, otherwise=(Sample(True, _CHAT_REASONING_OFF),)),
                "max_completion_tokens": Sample(64),
                "max_tokens": Sample(64),
                "modalities": Sample(["text"]),
                "n": Sample(2, alternatives=(3,)),
                "parallel_tool_calls": Sample(False, {"tools": [_CHAT_TOOL]}, (True,)),
                "prediction": Sample({"type": "content", "content": "ok"}),
                "presence_penalty": Sample(0.1, alternatives=(0.5,), otherwise=(Sample(0.1, _CHAT_REASONING_OFF),)),
                "prompt_cache_key": Sample("matrix"),
                "prompt_cache_retention": Sample("24h"),
                "reasoning_effort": Sample("low", alternatives=("high",)),
                "response_format": Sample(
                    {"type": "json_schema", "json_schema": {"name": "matrix", "schema": _SCHEMA, "strict": True}},
                    alternatives=(
                        {"type": "json_schema", "json_schema": {"name": "matrix", "schema": _SCHEMA}},
                        {"type": "json_object"},
                    ),
                ),
                "safety_identifier": Sample("matrix-user"),
                "seed": Sample(7, alternatives=(11,)),
                "service_tier": Sample("auto", alternatives=("default",)),
                "stop": Sample(["zzz"], alternatives=(["yyy"],)),
                "store": Sample(False),
                "stream": Sample(True),
                "stream_options": Sample({"include_usage": True}, {"stream": True}, ({"include_usage": False},)),
                "temperature": Sample(0.5, alternatives=(0.9, 1), otherwise=(Sample(0.5, _CHAT_REASONING_OFF),)),
                "thinking": Sample(_THINKING, {"max_tokens": 2048}),
                "tool_choice": Sample("auto", {"tools": [_CHAT_TOOL]}, ("required", "none")),
                "tools": Sample([_CHAT_TOOL]),
                "top_logprobs": Sample(
                    2, {"logprobs": True}, otherwise=(Sample(2, {"logprobs": True, **_CHAT_REASONING_OFF}),)
                ),
                "top_p": Sample(0.9, otherwise=(Sample(0.9, _CHAT_REASONING_OFF),)),
                "user": Sample("matrix-user"),
                "verbosity": Sample("low"),
                "web_search_options": Sample({"search_context_size": "low"}),
            }
        ),
        Endpoint.RESPONSES: MappingProxyType(
            {
                "background": Sample(True),
                "context_management": Sample([{"type": "compaction", "compact_threshold": 100000}]),
                "include": Sample(["message.output_text.logprobs"]),
                "instructions": Sample("Be brief."),
                "max_output_tokens": Sample(64),
                "max_tool_calls": Sample(1),
                "metadata": Sample({"matrix": "1"}),
                "moderation": Sample({"model": "omni-moderation-latest"}),
                "parallel_tool_calls": Sample(False, {"tools": [_RESPONSES_TOOL]}, (True,)),
                "partial_images": Sample(1),
                "previous_response_id": Sample("resp_matrix123"),
                "prompt": Sample({"id": "pmpt_matrix"}),
                "prompt_cache_key": Sample("matrix"),
                "prompt_cache_options": Sample({"mode": "explicit"}),
                "prompt_cache_retention": Sample("24h"),
                "reasoning": Sample({"effort": "low"}),
                "safety_identifier": Sample("matrix-user"),
                "service_tier": Sample("auto", alternatives=("default",)),
                "store": Sample(False),
                "stream": Sample(True),
                "stream_options": Sample({"include_obfuscation": False}, {"stream": True}),
                "temperature": Sample(0.5, alternatives=(0.9, 1), otherwise=(Sample(0.5, _RESPONSES_REASONING_OFF),)),
                "text": Sample(
                    {"format": {"type": "json_schema", "name": "matrix", "schema": _SCHEMA, "strict": True}},
                    alternatives=(
                        {"format": {"type": "json_schema", "name": "matrix", "schema": _SCHEMA}},
                        {"format": {"type": "json_object"}},
                    ),
                ),
                "tool_choice": Sample("auto", {"tools": [_RESPONSES_TOOL]}, ("required", "none")),
                "tools": Sample([_RESPONSES_TOOL]),
                "top_logprobs": Sample(2, otherwise=(Sample(2, _RESPONSES_REASONING_OFF),)),
                "top_p": Sample(0.9, otherwise=(Sample(0.9, _RESPONSES_REASONING_OFF),)),
                "truncation": Sample("auto"),
                "user": Sample("matrix-user"),
            }
        ),
        Endpoint.MESSAGES: MappingProxyType(
            {
                "cache_control": Sample({"type": "ephemeral"}),
                "compaction": Sample({"type": "summarize"}),
                "container": Sample({"id": "container_matrix"}),
                "context_management": Sample({"edits": [{"type": "clear_tool_uses_20250919"}]}),
                "diagnostics": Sample({"previous_message_id": "msg_matrix"}),
                "inference_geo": Sample("us", alternatives=("global",)),
                "max_tokens": Sample(32),
                "mcp_servers": Sample([{"type": "url", "url": "https://example.com/mcp", "name": "matrix"}]),
                "metadata": Sample({"user_id": "matrix"}),
                "output_config": Sample({"effort": "low"}, alternatives=({"effort": "high"},)),
                "output_format": Sample({"type": "json_schema", "schema": _SCHEMA}),
                "reasoning_effort": Sample("low", alternatives=("high",)),
                "safeguards": Sample([{"type": "dangerous_tool_use"}]),
                "service_tier": Sample("auto", alternatives=("standard_only",)),
                "speed": Sample("fast", alternatives=("standard",)),
                "stop_sequences": Sample(["zzz"]),
                "stream": Sample(True),
                "system": Sample("Be brief."),
                "temperature": Sample(0.5, alternatives=(0.9, 1), otherwise=(Sample(0.5, _MESSAGES_REASONING_OFF),)),
                "thinking": Sample(_THINKING, {"max_tokens": 2048}),
                "thread": Sample({"type": "create"}),
                "tool_choice": Sample(
                    {"type": "auto"}, {"tools": [_MESSAGES_TOOL]}, ({"type": "any"}, {"type": "none"})
                ),
                "tools": Sample([_MESSAGES_TOOL]),
                "top_k": Sample(5),
                "top_p": Sample(0.9, otherwise=(Sample(0.9, _MESSAGES_REASONING_OFF),)),
            }
        ),
    }
)
