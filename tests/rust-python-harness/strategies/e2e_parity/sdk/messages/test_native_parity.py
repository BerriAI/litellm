"""Native Anthropic Messages calls must be indistinguishable from the Python implementation.

Each case sends one `/v1/messages` SDK call through the Python handler and through the native route against the same
scripted upstream, then compares one aspect of what happened:

- dispatch: the native leg was served natively instead of falling back to Python
- request: every request that reached the upstream (method, path, body, headers)
- outcome: what the caller got back (response body, stream bytes, or the raised exception)
- hidden_params: the `_hidden_params` keys the proxy builds its response headers from
- logged: which terminal callbacks fired and the stable standard logging payload fields
- logged_metadata, logged_model_info, logged_response_headers, logged_api_key, logged_call: single callback kwargs
  fields that proxy integrations read

Both legs run once per case and every aspect of that case reads the same pair

The native leg prepends an unconstrained Rust rule, so feature detectors that keep a call on Python today do not
apply: a case for a detector-covered feature shows what Rust still lacks before that detector can go. Every known
difference is a strict xfail naming the gap, with an id from the Rust vs Python Messages inventory (A routing,
R request, S response, C callbacks, F covered by a fallback detector today)
"""

import asyncio
import copy
import datetime
import json
import os
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, Final, cast

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.anthropic_interface.exceptions.exception_mapping_utils import AnthropicExceptionMapping
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # a primed entry must land before the measured call reads it
)
from litellm.caching.dual_cache import DualCache
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.litellm_core_utils import logging_utils
from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.messages import dispatch
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.hooks.message_thread_ownership import MessageThreadOwnership
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.cooldown_handlers import is_message_thread_not_found
from litellm.rust_bridge import catalog
from litellm.rust_bridge.configuration import Rollout
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

SONNET: Final = "anthropic/claude-sonnet-4-5"
OPUS_ADAPTIVE: Final = "anthropic/claude-opus-4-7"
OPUS_FAST: Final = "anthropic/claude-opus-5-5"
API_KEY: Final = "sk-ant-parity-test"
USER: Final[Mapping[str, JsonValue]] = MappingProxyType({"role": "user", "content": "What is the capital of France?"})

_LOGGED: Final = TypeAdapter(dict[str, object])
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TRANSPORT_HEADERS: Final = frozenset(
    {"accept", "accept-encoding", "connection", "content-length", "host", "user-agent"}
)
_COMPARED_LOG_FIELDS: Final = (
    "call_type",
    "custom_llm_provider",
    "model",
    "model_group",
    "model_id",
    "api_base",
    "stream",
    "cache_hit",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "response_cost",
    "status",
)
_TERMINAL_EVENTS: Final = (
    "log_success_event",
    "async_log_success_event",
    "log_failure_event",
    "async_log_failure_event",
)
_VOLATILE: Final = (
    (re.compile(r"time taken=[0-9.]+"), "time taken=<t>"),
    (re.compile(r"Timeout passed=[0-9.]+"), "Timeout passed=<t>"),
    (re.compile(r"127\.0\.0\.1:[0-9]+"), "127.0.0.1:<port>"),
)
_SERVER_TOOL_ID: Final = re.compile(r"srvtoolu_[0-9a-f]{32}")

TEXT_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "id": "msg_01ParityText",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": "Paris"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 14, "output_tokens": 4},
    }
)

TOOL_USE_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "id": "msg_01ParityTool",
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_01Parity", "name": "get_weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use",
    }
)

THINKING_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "id": "msg_01ParityThinking",
        "content": [
            {"type": "thinking", "thinking": "The capital of France is Paris.", "signature": "EqQBCkYIARgCKkB"},
            {"type": "text", "text": "Paris"},
        ],
    }
)

CACHED_USAGE_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "usage": {
            "input_tokens": 14,
            "output_tokens": 4,
            "cache_creation_input_tokens": 2048,
            "cache_read_input_tokens": 1024,
            "service_tier": "standard",
        },
    }
)

WEATHER_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "name": "get_weather",
        "description": "Current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
)

TOOL_TURNS: Final[tuple[JsonValue, ...]] = (
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_01Parity", "name": "get_weather", "input": {"city": "Paris"}}],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01Parity", "content": "18C, sunny"}]},
)

THINKING_REPLAY: Final[tuple[JsonValue, ...]] = (
    {"role": "user", "content": "What is the capital of France?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Paris.", "signature": "EqQBCkYIARgCKkB"},
            {"type": "text", "text": "Paris"},
        ],
    },
    {"role": "user", "content": "And Germany?"},
)

_MODIFY_PARAMS: Final[Mapping[str, object]] = MappingProxyType({"modify_params": True})
NO_INPUT: Final[Mapping[str, JsonValue]] = MappingProxyType({})
PNG_PIXEL: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def _events(*events: tuple[str, JsonValue]) -> tuple[tuple[str, object], ...]:
    return tuple((name, data) for name, data in events)


def _message_start(message_id: str) -> tuple[str, JsonValue]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                **TEXT_RESPONSE,
                "id": message_id,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 14, "output_tokens": 1},
            },
        },
    )


def _text_block(index: int, *pieces: str) -> tuple[tuple[str, JsonValue], ...]:
    return (
        (
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        ),
        *(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": piece}},
            )
            for piece in pieces
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": index}),
    )


def _message_end(stop_reason: str, output_tokens: int) -> tuple[tuple[str, JsonValue], ...]:
    return (
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


TEXT_STREAM: Final = _events(
    _message_start("msg_01ParityStream"), *_text_block(0, "Pa", "ris"), *_message_end("end_turn", 4)
)

TOOL_STREAM: Final = _events(
    _message_start("msg_01ParityToolStream"),
    *_text_block(0, "Checking."),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_01Parity", "name": "get_weather", "input": {}},
        },
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    *_message_end("tool_use", 22),
)

THINKING_STREAM: Final = _events(
    _message_start("msg_01ParityThinkingStream"),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Paris."}},
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "EqQBCkYIARgCKkB"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    *_text_block(1, "Paris"),
    *_message_end("end_turn", 12),
)

ERROR_MID_STREAM: Final = _events(
    _message_start("msg_01ParityOverloaded"),
    *_text_block(0, "Pa"),
    ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
)

SLOW_STREAM: Final = _events(
    _message_start("msg_01ParitySlow"),
    *_text_block(0, "a", "b", "c", "d", "e", "f"),
    *_message_end("end_turn", 6),
)


def _upstream_error(kind: str, message: str) -> Mapping[str, JsonValue]:
    return MappingProxyType({"type": "error", "error": {"type": kind, "message": message}})


def _ok(body: Mapping[str, JsonValue] = TEXT_RESPONSE, **headers: str) -> ResponseSpec:
    return ResponseSpec(body=dict(body), headers=dict(headers))


def _stream(events: tuple[tuple[str, object], ...], chunk_delay: float = 0) -> ResponseSpec:
    return ResponseSpec(body={}, events=events, chunk_delay=chunk_delay)


def _failure(status: int, kind: str, message: str, **headers: str) -> ResponseSpec:
    return ResponseSpec(body=dict(_upstream_error(kind, message)), status=status, headers=dict(headers))


CALLBACK_ASPECTS: Final = (
    "logged_metadata",
    "logged_model_info",
    "logged_response_headers",
    "logged_api_key",
    "logged_call",
    "logged_request",
    "logged_sent_request",
)
ASPECTS: Final = ("dispatch", "request", "outcome", "hidden_params", "logged", *CALLBACK_ASPECTS)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    kwargs: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    upstream: tuple[ResponseSpec, ...] = (ResponseSpec(body=dict(TEXT_RESPONSE)),)
    model: str = SONNET
    sync: bool = False
    env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    settings: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    private_keys: bool = False
    cache: Callable[[], Cache] | None = None
    callbacks: tuple[Callable[[], CustomLogger], ...] = ()
    classifier: str | None = None
    gaps: Mapping[str, str | None] = field(default_factory=lambda: MappingProxyType({}))
    loggers: Callable[[], Sequence[CustomLogger]] | None = None

    def aspects(self) -> tuple[str, ...]:
        return (
            *ASPECTS,
            *(("cached",) if self.cache is not None else ()),
            *(("hook_inputs",) if self.loggers else ()),
            *(("classified",) if self.classifier is not None else ()),
        )

    def params(self) -> Iterator[object]:
        for aspect in self.aspects():
            gap = {**KNOWN_GAPS.get(self.name, {}), **self.gaps}.get(aspect)
            marks = () if not gap else (pytest.mark.xfail(reason=gap, strict=True),)
            yield pytest.param(self, aspect, id=f"{self.name}-{aspect}", marks=marks)


def _gaps(**gaps: str | None) -> Mapping[str, str | None]:
    return MappingProxyType(gaps)


def _kw(**kwargs: object) -> Mapping[str, object]:
    return MappingProxyType(kwargs)


A10_NO_MAX_TOKENS: Final = "A10: native admission hands a call without an int max_tokens to Python, whose signature raises the TypeError; only dispatch differs"
S6_PRIVATE_KEYS: Final = "S6: native writes _hidden_params into the response dict"
S15_COMPLETE_DEADLINE: Final = (
    "S15: by design, native bounds a complete response by one overall deadline where Python bounds each read, "
    "so a body trickling in slower than the timeout succeeds in Python and times out natively"
)

C_UNSENT_CALL_FLAG: Final = (
    "C: a call that fails natively before its provider request logs no aanthropic_messages flag, "
    "which Python's handler sets before it shapes the request"
)

KNOWN_GAPS: Final[Mapping[str, Mapping[str, str | None]]] = MappingProxyType(
    {
        "response_private_keys": _gaps(outcome=S6_PRIVATE_KEYS),
        "missing_max_tokens": _gaps(dispatch=A10_NO_MAX_TOKENS),
        **{
            name: _gaps(logged_call=C_UNSENT_CALL_FLAG)
            for name in (
                "speed_on_model_without_speed",
                "temperature_on_adaptive_model",
                "top_k_on_adaptive_model",
                "legacy_thinking_non_numeric_budget_on_adaptive_model",
                "legacy_thinking_null_budget_on_adaptive_model",
                "missing_api_key",
                "tls_security_level_python_cannot_apply",
            )
        },
    }
)


REQUEST_CASES: Final = (
    Case("plain_text"),
    Case("system_string", _kw(system="Answer in one word.")),
    Case(
        "system_blocks_with_cache_control",
        _kw(system=[{"type": "text", "text": "Answer in one word.", "cache_control": {"type": "ephemeral"}}]),
    ),
    Case("multi_turn_tool_result", _kw(messages=list(TOOL_TURNS), tools=[dict(WEATHER_TOOL)])),
    Case(
        "tools_with_tool_choice",
        _kw(tools=[dict(WEATHER_TOOL)], tool_choice={"type": "tool", "name": "get_weather"}),
        upstream=(_ok(TOOL_USE_RESPONSE),),
    ),
    Case("tool_choice_none", _kw(tools=[dict(WEATHER_TOOL)], tool_choice={"type": "none"})),
    Case("tool_input_examples", _kw(tools=[{**WEATHER_TOOL, "input_examples": [{"city": "Paris"}]}])),
    Case("sampling_params", _kw(temperature=0.3, top_p=0.9, top_k=40)),
    Case("stop_sequences", _kw(stop_sequences=["\n\nHuman:"])),
    Case("metadata_user_id", _kw(metadata={"user_id": "user-123"})),
    Case("metadata_extra_keys_are_filtered", _kw(metadata={"user_id": "user-123", "session": "s-1"})),
    Case("metadata_user_id_not_a_string", _kw(metadata={"user_id": 123})),
    Case("service_tier", _kw(service_tier="auto")),
    Case("thinking_enabled", _kw(thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096)),
    Case("thinking_disabled", _kw(thinking={"type": "disabled"})),
    Case(
        "legacy_thinking_on_adaptive_model_keeps_display",
        _kw(thinking={"type": "enabled", "budget_tokens": 2048, "display": "summarized"}, max_tokens=4096),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_display_updates_on_adaptive_model",
        _kw(thinking={"type": "enabled", "budget_tokens": 2048, "display": "updates"}, max_tokens=4096),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_string_budget_on_adaptive_model",
        _kw(thinking={"type": "enabled", "budget_tokens": " 4_096 "}, max_tokens=8192),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_non_numeric_budget_on_adaptive_model",
        _kw(thinking={"type": "enabled", "budget_tokens": "lots"}, max_tokens=8192),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_null_budget_on_adaptive_model",
        _kw(thinking={"type": "enabled", "budget_tokens": None}, max_tokens=8192),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_non_numeric_budget_dropped",
        _kw(thinking={"type": "enabled", "budget_tokens": [1]}, max_tokens=8192, drop_params=True),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_carries_extra_keys",
        _kw(
            thinking={"type": "enabled", "budget_tokens": 2048, "block_binding": "strict", "note": "kept"},
            max_tokens=4096,
        ),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_drops_extra_keys_with_drop_params",
        _kw(
            thinking={"type": "enabled", "budget_tokens": 2048, "block_binding": "strict", "note": "dropped"},
            max_tokens=4096,
            drop_params=True,
        ),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "legacy_thinking_float_budget_on_adaptive_model",
        _kw(thinking={"type": "enabled", "budget_tokens": 8192.0}, max_tokens=16000),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "adaptive_thinking_with_block_binding",
        _kw(thinking={"type": "adaptive", "block_binding": "strict"}, max_tokens=4096),
        model=OPUS_ADAPTIVE,
    ),
    Case("reasoning_effort", _kw(reasoning_effort="high", max_tokens=8192)),
    Case("reasoning_effort_on_adaptive_model", _kw(reasoning_effort="low", max_tokens=8192), model=OPUS_ADAPTIVE),
    Case("output_config_effort", _kw(output_config={"effort": "medium"}), model=OPUS_ADAPTIVE),
    Case(
        "output_format_json_schema",
        _kw(
            output_format={
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ),
    ),
    Case("speed_on_model_without_speed", _kw(speed="fast")),
    Case("speed_on_model_without_speed_drop_params", _kw(speed="fast", drop_params=True)),
    Case("speed_on_fast_model", _kw(speed="fast"), model=OPUS_FAST),
    Case("temperature_on_adaptive_model", _kw(temperature=0.5), model=OPUS_ADAPTIVE),
    Case("top_k_on_adaptive_model", _kw(top_k=5), model=OPUS_ADAPTIVE),
    Case("top_k_on_adaptive_model_drop_params", _kw(top_k=5, drop_params=True), model=OPUS_ADAPTIVE),
    Case(
        "additional_drop_params_hides_rejected_temperature",
        _kw(temperature=0.5, additional_drop_params=["temperature"]),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "additional_drop_params_nested_path",
        _kw(
            tools=[{**WEATHER_TOOL, "input_examples": [{"city": "Paris"}]}],
            additional_drop_params=["tools[*].input_examples", "metadata"],
            metadata={"user_id": "user-123"},
        ),
    ),
    Case(
        "context_management_edits",
        _kw(context_management={"edits": [{"type": "clear_tool_uses_20250919"}]}),
    ),
    Case(
        "container_skills", _kw(container={"skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]})
    ),
    Case(
        "remote_mcp_servers",
        _kw(mcp_servers=[{"type": "url", "url": "https://mcp.example.com/sse", "name": "example"}]),
    ),
    Case("inference_geo", _kw(inference_geo="us")),
    Case("top_level_cache_control", _kw(cache_control={"type": "ephemeral"})),
    Case("null_safeguards_are_stripped", _kw(safeguards=None)),
    Case(
        "empty_text_blocks_are_stripped",
        _kw(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "hi"}]},
            ]
        ),
    ),
    Case(
        "tool_use_ids_are_sanitized",
        _kw(
            messages=[
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call:abc/1", "name": "get_weather", "input": dict(NO_INPUT)}
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call:abc/1", "content": "sunny"}]},
            ],
            tools=[dict(WEATHER_TOOL)],
        ),
    ),
    Case(
        "provider_specific_fields_are_stripped",
        _kw(messages=[{"role": "user", "content": "hi", "provider_specific_fields": {"x": 1}}]),
    ),
    Case(
        "image_block",
        _kw(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_PIXEL}},
                        {"type": "text", "text": "Describe it"},
                    ],
                }
            ]
        ),
    ),
    Case(
        "document_block",
        _kw(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Paris."}},
                        {"type": "text", "text": "Summarize"},
                    ],
                }
            ]
        ),
    ),
    Case("server_web_search_tool", _kw(tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}])),
    Case("extra_headers", _kw(extra_headers={"x-trace-id": "trace-1", "anthropic-beta": "files-api-2025-04-14"})),
    Case(
        "anthropic_beta_from_headers_and_extra_headers",
        _kw(headers={"anthropic-beta": "client-beta-1"}, extra_headers={"anthropic-beta": "deploy-beta-2"}),
    ),
    Case("caller_anthropic_version_capitalised", _kw(extra_headers={"Anthropic-Version": "2023-06-01"})),
    Case(
        "provider_specific_header",
        _kw(provider_specific_header={"custom_llm_provider": "anthropic", "extra_headers": {"x-scoped": "yes"}}),
    ),
    Case("explicit_custom_llm_provider", _kw(custom_llm_provider="anthropic"), model="claude-sonnet-4-5"),
    Case("bare_model_name", model="claude-sonnet-4-5"),
    Case("api_base_with_messages_path", _kw(api_base_suffix="/v1/messages")),
    Case(
        "api_base_with_messages_path_and_query",
        _kw(api_base_suffix="/v1/messages?beta=true"),
    ),
    Case("api_key_from_environment", _kw(api_key=None), env=MappingProxyType({"ANTHROPIC_API_KEY": API_KEY})),
    Case(
        "auth_token_from_environment",
        _kw(api_key=None),
        env=MappingProxyType({"ANTHROPIC_AUTH_TOKEN": "oauth-token"}),
    ),
    Case(
        "reasoning_auto_summary",
        _kw(thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096),
        env=MappingProxyType({"LITELLM_REASONING_AUTO_SUMMARY": "true"}),
    ),
    Case("global_drop_params", _kw(top_k=5), model=OPUS_ADAPTIVE, settings=MappingProxyType({"drop_params": True})),
    Case("request_timeout_setting", settings=MappingProxyType({"request_timeout": 30})),
    Case(
        "litellm_metadata_and_metadata",
        _kw(
            metadata={"user_id": "user-123"},
            litellm_metadata={"user_api_key": "hashed-key", "user_api_key_team_id": "t1"},
        ),
    ),
    Case("model_info_with_id", _kw(model_info={"id": "deployment-1", "db_model": False})),
    Case(
        "deployment_effort_levels",
        _kw(
            output_config={"effort": "max"},
            model_info={"id": "deployment-1", "reasoning_effort_levels": ["low", "high"]},
        ),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "deployment_effort_levels_from_legacy_thinking",
        _kw(
            thinking={"type": "enabled", "budget_tokens": 30000},
            max_tokens=40000,
            model_info={"id": "deployment-1", "reasoning_effort_levels": ["low", "medium"]},
        ),
        model=OPUS_ADAPTIVE,
    ),
    Case(
        "deployment_effort_levels_below_request_with_drop_params",
        _kw(
            output_config={"effort": "high"},
            drop_params=True,
            model_info={"id": "deployment-1", "reasoning_effort_levels": ["low"]},
        ),
        model=OPUS_ADAPTIVE,
    ),
)

RESPONSE_CASES: Final = (
    Case("response_with_tool_use", _kw(tools=[dict(WEATHER_TOOL)]), upstream=(_ok(TOOL_USE_RESPONSE),)),
    Case("response_with_thinking", upstream=(_ok(THINKING_RESPONSE),)),
    Case("response_with_cache_usage", upstream=(_ok(CACHED_USAGE_RESPONSE),)),
    Case(
        "response_with_null_usage_and_container",
        upstream=(_ok({**TEXT_RESPONSE, "usage": None, "container": None}),),
    ),
    Case(
        "response_without_stop_sequence",
        upstream=(_ok({k: v for k, v in TEXT_RESPONSE.items() if k != "stop_sequence"}),),
    ),
    Case("response_with_unknown_field", upstream=(_ok({**TEXT_RESPONSE, "future_field": {"a": 1}}),)),
    Case("response_key_order", upstream=(_ok({"usage": TEXT_RESPONSE["usage"], **TEXT_RESPONSE}),)),
    Case(
        "response_upstream_headers",
        upstream=(
            _ok(TEXT_RESPONSE, **{"request-id": "req_01Parity", "anthropic-ratelimit-requests-remaining": "99"}),
        ),
    ),
    Case("response_private_keys", private_keys=True),
)

FAILURE_CASES: Final = (
    Case("upstream_400", upstream=(_failure(400, "invalid_request_error", "max_tokens: too large"),)),
    Case("upstream_401", upstream=(_failure(401, "authentication_error", "invalid x-api-key"),)),
    Case("upstream_403", upstream=(_failure(403, "permission_error", "not allowed"),)),
    Case("upstream_404", upstream=(_failure(404, "not_found_error", "model not found"),)),
    Case("upstream_413", upstream=(_failure(413, "request_too_large", "too large"),)),
    Case(
        "upstream_429_with_retry_after",
        upstream=(
            _failure(429, "rate_limit_error", "slow down", **{"retry-after": "7", "request-id": "req_01Parity"}),
        ),
    ),
    Case("upstream_500", upstream=(_failure(500, "api_error", "internal"),)),
    Case("upstream_529_overloaded", upstream=(_failure(529, "overloaded_error", "Overloaded"),)),
    Case(
        "upstream_400_context_window",
        upstream=(_failure(400, "invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum"),),
    ),
    Case("upstream_non_json_error", upstream=(ResponseSpec(body=b"<html>bad gateway</html>", status=502),)),
    Case("upstream_invalid_json_success", upstream=(ResponseSpec(body=b"not json", status=200),)),
    Case("upstream_timeout", _kw(timeout=0.5), upstream=(ResponseSpec(body=dict(TEXT_RESPONSE), delay=1.5),)),
    Case(
        "response_trickles_slower_than_timeout",
        _kw(timeout=1),
        upstream=(ResponseSpec(body=dict(TEXT_RESPONSE), chunk_delay=0.5, body_chunks=4),),
        gaps=_gaps(
            logged_request=S15_COMPLETE_DEADLINE,
            outcome=S15_COMPLETE_DEADLINE,
            hidden_params=S15_COMPLETE_DEADLINE,
            logged=S15_COMPLETE_DEADLINE,
            logged_metadata=S15_COMPLETE_DEADLINE,
            logged_response_headers=S15_COMPLETE_DEADLINE,
            logged_sent_request=S15_COMPLETE_DEADLINE,
        ),
    ),
    Case("missing_max_tokens", _kw(max_tokens=None), upstream=()),
    Case("missing_api_key", _kw(api_key=None), upstream=()),
)

STREAM_CASES: Final = (
    Case("stream_text", _kw(stream=True), upstream=(_stream(TEXT_STREAM),)),
    Case("stream_tool_use", _kw(stream=True, tools=[dict(WEATHER_TOOL)]), upstream=(_stream(TOOL_STREAM),)),
    Case(
        "stream_thinking",
        _kw(stream=True, thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096),
        upstream=(_stream(THINKING_STREAM),),
    ),
    Case("stream_error_event_mid_stream", _kw(stream=True), upstream=(_stream(ERROR_MID_STREAM),)),
    Case(
        "stream_longer_than_timeout",
        _kw(stream=True, timeout=1),
        upstream=(_stream(SLOW_STREAM, chunk_delay=0.25),),
    ),
    Case(
        "stream_read_stalls_past_timeout",
        _kw(stream=True, timeout=0.5),
        upstream=(_stream(SLOW_STREAM, chunk_delay=1.0),),
    ),
    Case(
        "stream_upstream_429",
        _kw(stream=True),
        upstream=(_failure(429, "rate_limit_error", "slow down", **{"retry-after": "3"}),),
    ),
    Case("stream_false_explicit", _kw(stream=False)),
)

SYNC_CASES: Final = (
    Case("sync_text", sync=True),
    Case("sync_stream", _kw(stream=True), upstream=(_stream(TEXT_STREAM),), sync=True),
    Case("sync_upstream_429", upstream=(_failure(429, "rate_limit_error", "slow down"),), sync=True),
)

SYSTEM_POINT: Final[Mapping[str, JsonValue]] = MappingProxyType({"location": "message", "role": "system"})
LAST_MESSAGE_POINT: Final[Mapping[str, JsonValue]] = MappingProxyType({"location": "message", "index": -1})
SYSTEM_BLOCKS: Final[tuple[JsonValue, ...]] = (
    {"type": "text", "text": "Answer in one word."},
    {"type": "text", "text": "Be terse."},
)
MARKED_TURNS: Final[tuple[JsonValue, ...]] = (
    {"role": "user", "content": [{"type": "text", "text": "One.", "cache_control": {"type": "ephemeral"}}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Two.", "cache_control": {"type": "ephemeral"}}]},
    {"role": "user", "content": [{"type": "text", "text": "Three.", "cache_control": {"type": "ephemeral"}}]},
)

PROMPT_CACHING_CASES: Final = (
    Case(
        "cache_control_injection_points",
        _kw(system="Answer in one word.", cache_control_injection_points=[dict(SYSTEM_POINT)]),
    ),
    Case("enable_prompt_caching", _kw(system="Answer in one word.", enable_prompt_caching=True)),
    Case(
        "enable_prompt_caching_without_system",
        _kw(messages=list(TOOL_TURNS), tools=[dict(WEATHER_TOOL)], enable_prompt_caching=True),
    ),
    Case(
        "global_prompt_caching_with_ttl",
        _kw(system=list(SYSTEM_BLOCKS)),
        settings=MappingProxyType({"enable_anthropic_prompt_caching": True, "anthropic_prompt_caching_ttl": "1h"}),
    ),
    Case(
        "global_prompt_caching_stands_down_on_client_marks",
        _kw(system=[{"type": "text", "text": "Answer in one word.", "cache_control": {"type": "ephemeral"}}]),
        settings=MappingProxyType({"enable_anthropic_prompt_caching": True}),
    ),
    Case(
        "global_prompt_caching_stands_down_on_marked_tools",
        _kw(tools=[{**WEATHER_TOOL, "cache_control": {"type": "ephemeral"}}]),
        settings=MappingProxyType({"enable_anthropic_prompt_caching": True}),
    ),
    Case(
        "enable_prompt_caching_false_overrides_nothing",
        _kw(system="Answer in one word.", enable_prompt_caching=False),
    ),
    Case(
        "injection_point_negative_index",
        _kw(
            messages=list(TOOL_TURNS),
            cache_control_injection_points=[
                {"location": "message", "index": -2, "control": {"type": "ephemeral", "ttl": "1h"}}
            ],
        ),
    ),
    Case(
        "injection_point_string_index",
        _kw(messages=list(TOOL_TURNS), cache_control_injection_points=[{"location": "message", "index": "1"}]),
    ),
    Case(
        "injection_point_out_of_range_index",
        _kw(messages=list(TOOL_TURNS), cache_control_injection_points=[{"location": "message", "index": 7}]),
    ),
    Case(
        "injection_point_by_role",
        _kw(messages=list(TOOL_TURNS), cache_control_injection_points=[{"location": "message", "role": "user"}]),
    ),
    Case(
        "injection_points_capped_by_client_marks",
        _kw(
            messages=list(MARKED_TURNS),
            system=list(SYSTEM_BLOCKS),
            tools=[{**WEATHER_TOOL, "cache_control": {"type": "ephemeral"}}],
            cache_control_injection_points=[dict(SYSTEM_POINT), dict(LAST_MESSAGE_POINT)],
        ),
    ),
    Case(
        "injection_points_beside_client_marks",
        _kw(
            messages=[*MARKED_TURNS[:2], {"role": "user", "content": "Three."}],
            cache_control_injection_points=[dict(SYSTEM_POINT), dict(LAST_MESSAGE_POINT)],
            system="Answer in one word.",
        ),
    ),
    Case(
        "tool_config_point_marks_the_last_tool",
        _kw(
            tools=[dict(WEATHER_TOOL), {**WEATHER_TOOL, "name": "get_time"}],
            cache_control_injection_points=[{"location": "tool_config"}, dict(LAST_MESSAGE_POINT)],
        ),
    ),
    Case(
        "tool_config_point_without_tools",
        _kw(cache_control_injection_points=[{"location": "tool_config"}]),
    ),
    Case(
        "injection_marks_the_gateway_deployment",
        _kw(
            system="Answer in one word.",
            enable_prompt_caching=True,
            litellm_metadata={"user_api_key": "hashed"},
            model_info={"id": "deployment-1"},
        ),
    ),
    Case(
        "injection_into_whitespace_only_turn",
        _kw(
            messages=[dict(USER), {"role": "assistant", "content": "Paris"}, {"role": "user", "content": "  "}],
            cache_control_injection_points=[dict(LAST_MESSAGE_POINT)],
        ),
        upstream=(_failure(400, "invalid_request_error", "messages: text content blocks must be non-empty"),),
    ),
    Case(
        "injection_after_empty_blocks_are_stripped",
        _kw(
            messages=[
                dict(USER),
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                {"role": "user", "content": "And Germany?"},
            ],
            cache_control_injection_points=[{"location": "message", "index": 1}],
        ),
    ),
    Case(
        "stream_with_prompt_caching",
        _kw(stream=True, system="Answer in one word.", enable_prompt_caching=True),
        upstream=(_stream(TEXT_STREAM),),
    ),
    Case(
        "global_prompt_caching_beside_an_unmatched_point",
        _kw(
            system="Answer in one word.", cache_control_injection_points=[{"location": "message", "role": "assistant"}]
        ),
        settings=MappingProxyType({"enable_anthropic_prompt_caching": True}),
    ),
)

SYNC_PROMPT_CACHING_CASES: Final = tuple(
    replace(case, name=f"sync_{case.name}", sync=True) for case in PROMPT_CACHING_CASES
)

LATEST_ALIAS: Final = "anthropic/claude-opus-latest"
CUSTOM_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType({**WEATHER_TOOL, "custom": {"defer_loading": True}})
COST_HEADER: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"headers": {"x-litellm-include-cost-in-streaming-usage": "true"}}
)

SERVED_NATIVELY_CASES: Final = (
    Case("mock_response", _kw(mock_response="Hi from the mock"), upstream=()),
    Case("mock_response_rate_limit", _kw(mock_response="litellm.RateLimitError"), upstream=()),
    Case("mock_response_context_window", _kw(mock_response="litellm.ContextWindowExceededError"), upstream=()),
    Case("mock_response_empty_calls_the_provider", _kw(mock_response="")),
    Case("mock_response_bad_metadata", _kw(mock_response="Hi", metadata={"user_id": 123}), upstream=()),
    Case("mock_response_stream", _kw(mock_response="Hi from the mock", stream=True), upstream=()),
    Case("mock_response_stream_rate_limit", _kw(mock_response="litellm.RateLimitError", stream=True), upstream=()),
    Case(
        "mock_response_stream_empty_calls_the_provider",
        _kw(mock_response="", stream=True),
        upstream=(_stream(TEXT_STREAM),),
    ),
    Case("mock_response_cached", _kw(mock_response="Hi from the mock"), upstream=(), cache=Cache),
    Case("mock_response_stream_cached", _kw(mock_response="Hi from the mock", stream=True), upstream=(), cache=Cache),
    Case("mock_response_sync", _kw(mock_response="Hi from the mock"), upstream=(), sync=True),
    Case("mock_response_sync_stream", _kw(mock_response="Hi from the mock", stream=True), upstream=(), sync=True),
    Case("mock_response_sync_rate_limit", _kw(mock_response="litellm.RateLimitError"), upstream=(), sync=True),
    Case("mock_response_sync_internal_error", _kw(mock_response="litellm.InternalServerError"), upstream=(), sync=True),
    Case("mock_response_sync_bad_metadata", _kw(mock_response="Hi", metadata={"user_id": 123}), upstream=(), sync=True),
    Case("mock_response_sync_empty_calls_the_provider", _kw(mock_response=""), sync=True),
    Case("latest_alias", model=LATEST_ALIAS),
    Case("latest_alias_bare_with_provider", _kw(custom_llm_provider="anthropic"), model="claude-opus-latest"),
    Case("latest_alias_stream", _kw(stream=True), upstream=(_stream(TEXT_STREAM),), model=LATEST_ALIAS),
    Case(
        "latest_alias_upstream_400",
        upstream=(_failure(400, "invalid_request_error", "max_tokens: too large"),),
        model=LATEST_ALIAS,
    ),
    Case("tool_custom_field", _kw(tools=[dict(CUSTOM_TOOL)])),
    Case(
        "tool_custom_field_keeps_other_custom_keys",
        _kw(tools=[{**WEATHER_TOOL, "custom": {"defer_loading": False, "note": "kept"}}]),
    ),
    Case(
        "tool_custom_field_dropped_with_drop_params",
        _kw(tools=[{**WEATHER_TOOL, "custom": {"defer_loading": True, "note": "dropped"}}], drop_params=True),
    ),
    Case("tool_custom_field_not_an_object", _kw(tools=[{**WEATHER_TOOL, "custom": "opaque"}])),
    Case(
        "tool_custom_field_under_own_defer_loading",
        _kw(tools=[{**WEATHER_TOOL, "defer_loading": False, "custom": {"defer_loading": True}}]),
    ),
    Case(
        "stream_cost_injection",
        _kw(stream=True),
        upstream=(_stream(TEXT_STREAM),),
        settings=MappingProxyType({"include_cost_in_streaming_usage": True}),
    ),
    Case(
        "stream_cost_injection_from_proxy_header",
        _kw(stream=True, proxy_server_request=dict(COST_HEADER)),
        upstream=(_stream(TOOL_STREAM, chunk_delay=0.01),),
    ),
    Case(
        "cache_off_by_default",
        settings=MappingProxyType({"cache": Cache(mode=CacheMode.default_off)}),
    ),
    Case(
        "cache_skips_messages_call_type",
        settings=MappingProxyType({"cache": Cache(supported_call_types=["acompletion"])}),
    ),
)

FALLBACK_CASES: Final = (
    Case("modify_params", _kw(max_tokens=1), settings=_MODIFY_PARAMS),
    Case("modify_params_clamps_max_tokens", _kw(max_tokens=200_000), settings=_MODIFY_PARAMS),
    Case(
        "modify_params_clamps_streamed_max_tokens",
        _kw(stream=True, max_tokens=200_000),
        upstream=(_stream(TEXT_STREAM),),
        settings=_MODIFY_PARAMS,
    ),
    Case(
        "modify_params_clamps_tool_turns",
        _kw(messages=list(TOOL_TURNS), tools=[dict(WEATHER_TOOL)], max_tokens=200_000),
        upstream=(_ok(TOOL_USE_RESPONSE),),
        settings=_MODIFY_PARAMS,
    ),
    Case(
        "modify_params_clamps_thinking",
        _kw(thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=200_000),
        upstream=(_ok(THINKING_RESPONSE),),
        settings=_MODIFY_PARAMS,
    ),
    Case(
        "modify_params_image_content",
        _kw(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_PIXEL}},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
            max_tokens=200_000,
        ),
        settings=_MODIFY_PARAMS,
    ),
    Case(
        "modify_params_unknown_model",
        _kw(max_tokens=200_000),
        model="anthropic/claude-parity-unknown",
        settings=_MODIFY_PARAMS,
    ),
    Case("modify_params_hf_model_name", _kw(max_tokens=200_000, hf_model_name="gpt2"), settings=_MODIFY_PARAMS),
    Case(
        "modify_params_cache_write_without_read",
        _kw(max_tokens=200_000, cache={"no-cache": True}),
        upstream=(_ok(), _ok()),
        settings=_MODIFY_PARAMS,
        cache=Cache,
    ),
    Case(
        "modify_params_clamped_call_never_hits_the_cache",
        _kw(max_tokens=200_000),
        upstream=(_ok(), _ok()),
        settings=_MODIFY_PARAMS,
        cache=Cache,
    ),
    Case(
        "modify_params_unclamped_call_hits_the_cache",
        _kw(max_tokens=10),
        upstream=(_ok(), _ok()),
        settings=_MODIFY_PARAMS,
        cache=Cache,
    ),
    Case("cache_write_without_read", _kw(cache={"no-cache": True}), upstream=(_ok(), _ok()), cache=Cache),
    Case("modify_params_sync", _kw(max_tokens=200_000), sync=True, settings=_MODIFY_PARAMS),
    Case("request_timeout_kwarg", _kw(request_timeout=30)),
    Case("stream_timeout_kwarg", _kw(stream=True, stream_timeout=30), upstream=(_stream(TEXT_STREAM),)),
)

INVALID_SIGNATURE: Final = "messages.1.content.0: Invalid `signature` in `thinking` block"
BOUND_ELSEWHERE: Final = f"{INVALID_SIGNATURE}. The block is bound to a different conversation."
THINKING: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": "enabled", "budget_tokens": 2048})
BINDING_ERRORS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}
)
THINKING_ONLY_REPLAY: Final[tuple[JsonValue, ...]] = (
    {"role": "user", "content": "What is the capital of France?"},
    {
        "role": "assistant",
        "content": [
            {"type": "redacted_thinking", "data": "EmwKAhgBEgy"},
            {"type": "thinking", "thinking": "Paris.", "signature": "EqQBCkYIARgCKkB"},
        ],
    },
    {"role": "assistant", "content": []},
    {"role": "user", "content": "And Germany?"},
)


def _replay(**kwargs: object) -> Mapping[str, object]:
    return _kw(messages=list(THINKING_REPLAY), thinking=dict(THINKING), max_tokens=4096, **kwargs)


def _rejected(message: str = INVALID_SIGNATURE, status: int = 400) -> ResponseSpec:
    return _failure(status, "invalid_request_error", message)


THINKING_RETRY_CASES: Final = (
    Case("thinking_replay", _replay()),
    Case("thinking_signature_rejected_then_retried", _replay(), upstream=(_rejected(), _ok())),
    Case(
        "thinking_signature_rejected_then_retried_stream",
        _replay(stream=True),
        upstream=(_rejected(), _stream(TEXT_STREAM)),
    ),
    Case(
        "thinking_signature_missing_then_retried",
        _replay(),
        upstream=(_rejected("messages.1.content.0.thinking.signature.str: Input should be a valid string"), _ok()),
    ),
    Case(
        "empty_thinking_rejected_then_retried_without_thinking_only_turns",
        _kw(messages=list(THINKING_ONLY_REPLAY), thinking=dict(THINKING), max_tokens=4096),
        upstream=(_rejected("messages.1.content.1.thinking: each thinking block must contain thinking"), _ok()),
    ),
    Case("thinking_signature_rejected_twice", _replay(), upstream=(_rejected(), _rejected())),
    Case(
        "thinking_signature_rejected_twice_stream",
        _replay(stream=True),
        upstream=(_rejected(), _rejected()),
    ),
    Case("thinking_signature_text_on_a_429_is_not_retried", _replay(), upstream=(_rejected(status=429),)),
    Case(
        "unrelated_400_is_not_retried",
        _replay(),
        upstream=(_rejected("max_tokens: 4096 > 64000, which is the maximum allowed"),),
    ),
    Case(
        "binding_mismatch_retried_when_the_caller_did_not_pin_it",
        _kw(messages=list(THINKING_REPLAY), thinking={"type": "adaptive"}, max_tokens=4096),
        model=OPUS_ADAPTIVE,
        upstream=(_rejected(BOUND_ELSEWHERE), _ok()),
    ),
    Case(
        "binding_mismatch_raised_when_the_caller_pinned_it",
        _kw(messages=list(THINKING_REPLAY), thinking=dict(BINDING_ERRORS), max_tokens=4096),
        model=OPUS_ADAPTIVE,
        upstream=(_rejected(BOUND_ELSEWHERE),),
    ),
    Case(
        "invalid_signature_retried_even_when_the_caller_pinned_binding",
        _kw(messages=list(THINKING_REPLAY), thinking=dict(BINDING_ERRORS), max_tokens=4096),
        model=OPUS_ADAPTIVE,
        upstream=(_rejected(), _ok()),
    ),
)


def _default_off_cache() -> Cache:
    return Cache(mode=CacheMode.default_off)


def _completion_only_cache() -> Cache:
    return Cache(supported_call_types=["acompletion"])


CACHE_CASES: Final = (
    Case("cache_hit", upstream=(_ok(), _ok()), cache=Cache),
    Case("cache_hit_stream", _kw(stream=True), upstream=(_stream(TEXT_STREAM), _stream(TEXT_STREAM)), cache=Cache),
    Case("cache_no_cache", _kw(cache={"no-cache": True}), upstream=(_ok(), _ok()), cache=Cache),
    Case("cache_no_store", _kw(cache={"no-store": True}), upstream=(_ok(), _ok()), cache=Cache),
    Case("cache_caching_false", _kw(caching=False), upstream=(_ok(), _ok()), cache=Cache),
    Case(
        "cache_ttl_and_namespace", _kw(cache={"ttl": 60, "namespace": "parity"}), upstream=(_ok(), _ok()), cache=Cache
    ),
    Case("cache_default_off", upstream=(_ok(), _ok()), cache=_default_off_cache),
    Case(
        "cache_default_off_use_cache",
        _kw(cache={"use-cache": True}),
        upstream=(_ok(), _ok()),
        cache=_default_off_cache,
    ),
    Case("cache_unsupported_call_type", upstream=(_ok(), _ok()), cache=_completion_only_cache),
    Case("cache_hit_with_metadata", _kw(metadata={"user_id": "user-123"}), upstream=(_ok(), _ok()), cache=Cache),
    Case("cache_sync", sync=True, upstream=(_ok(), _ok()), cache=Cache),
)

STRICT_REJECTION: Final = "Too many strict tools (21). The maximum number of strict tools is 20."
STRICT_TOOLS: Final[tuple[JsonValue, ...]] = (
    {**WEATHER_TOOL, "strict": True},
    {**WEATHER_TOOL, "name": "get_time", "strict": False},
    dict(WEATHER_TOOL, name="get_tide"),
)


def _strict(**kwargs: object) -> Mapping[str, object]:
    return _kw(tools=list(STRICT_TOOLS), **kwargs)


STRICT_TOOLS_RETRY_CASES: Final = (
    Case(
        "strict_tools_rejected_then_retried", _strict(drop_params=True), upstream=(_rejected(STRICT_REJECTION), _ok())
    ),
    Case(
        "strict_tools_rejected_then_retried_stream",
        _strict(drop_params=True, stream=True),
        upstream=(_rejected(STRICT_REJECTION), _stream(TEXT_STREAM)),
    ),
    Case(
        "strict_tools_rejected_then_retried_under_global_drop_params",
        _strict(),
        upstream=(_rejected("The compiled grammar is too large"), _ok()),
        settings=MappingProxyType({"drop_params": True}),
    ),
    Case(
        "strict_tools_custom_field_rejected_then_retried",
        _strict(drop_params="true"),
        upstream=(_rejected("tools.0.custom.strict: Extra inputs are not permitted"), _ok()),
    ),
    Case(
        "strict_tools_rejected_without_drop_params",
        _strict(),
        upstream=(_rejected(STRICT_REJECTION),),
        settings=MappingProxyType({"drop_params": False}),
    ),
    Case(
        "strict_tools_rejected_under_a_truthy_global_drop_params_that_is_not_true",
        _strict(),
        upstream=(_rejected(STRICT_REJECTION),),
        settings=MappingProxyType({"drop_params": 1}),
    ),
    Case(
        "strict_rejection_without_a_strict_tool",
        _kw(tools=[{**WEATHER_TOOL, "strict": False}], drop_params=True),
        upstream=(_rejected(STRICT_REJECTION),),
    ),
    Case(
        "strict_rejection_text_on_a_429_is_not_retried",
        _strict(drop_params=True),
        upstream=(_rejected(STRICT_REJECTION, status=429),),
    ),
    Case(
        "strict_tools_rejected_twice",
        _strict(drop_params=True),
        upstream=(_rejected(STRICT_REJECTION), _rejected("Grammar compilation timed out")),
    ),
    Case(
        "strict_tools_rejected_twice_stream",
        _strict(drop_params=True, stream=True),
        upstream=(_rejected(STRICT_REJECTION), _rejected(STRICT_REJECTION)),
    ),
    Case(
        "strict_tools_and_thinking_strict_rejected_then_retried",
        _replay(tools=list(STRICT_TOOLS), drop_params=True),
        upstream=(_rejected(STRICT_REJECTION), _ok()),
    ),
    Case(
        "strict_tools_and_thinking_signature_rejected_then_retried",
        _replay(tools=list(STRICT_TOOLS), drop_params=True),
        upstream=(_rejected(), _ok()),
    ),
    Case(
        "strict_tools_and_thinking_each_rejected_spends_the_one_retry",
        _replay(tools=list(STRICT_TOOLS), drop_params=True),
        upstream=(_rejected(STRICT_REJECTION), _rejected()),
    ),
    Case(
        "strict_tools_and_thinking_signature_then_strict_rejected",
        _replay(tools=list(STRICT_TOOLS), drop_params=True),
        upstream=(_rejected(), _rejected(STRICT_REJECTION), _ok()),
    ),
    Case(
        "one_error_naming_both_drops_thinking_and_strict_without_drop_params",
        _replay(tools=list(STRICT_TOOLS)),
        upstream=(_rejected(f"{INVALID_SIGNATURE}. {STRICT_REJECTION}"), _ok()),
    ),
)

THREADS_BETA: Final = "message-threads-2026-08-12"
ADVISOR_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-8"}
)
THREAD_CREATE: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": "create"})
THREAD_CONTINUE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "continue", "previous_message_id": "msg_01ParityText"}
)
ADVISOR_TURNS: Final[tuple[JsonValue, ...]] = (
    {"role": "user", "content": "Build a worker pool."},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me consult the advisor."},
            {"type": "server_tool_use", "id": "srvtoolu_01Parity", "name": "advisor", "input": {}},
            {
                "type": "advisor_tool_result",
                "tool_use_id": "srvtoolu_01Parity",
                "content": {"type": "advisor_result", "text": "Use channels."},
            },
            {"type": "text", "text": "Here is the implementation."},
        ],
    },
    {"role": "user", "content": "Now add a timeout.", "output_config": {"effort": "low"}},
    {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
    {"role": "user", "content": "Go on."},
)
THREAD_NOT_FOUND: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "error",
        "error": {
            "type": "not_found_error",
            "message": "No thread state was found for the requested `previous_message_id`.",
            "details": {"error_code": "thread_not_found"},
        },
        "request_id": "req_01Parity",
    }
)


def _thread_not_found() -> ResponseSpec:
    return ResponseSpec(body=dict(THREAD_NOT_FOUND), status=404)


THREAD_CASES: Final = (
    Case("thread_create", _kw(thread=dict(THREAD_CREATE), system="Answer in one word.")),
    Case("thread_create_with_advisor_tool", _kw(thread=dict(THREAD_CREATE), tools=[dict(ADVISOR_TOOL)])),
    Case("thread_create_scans_history", _kw(thread=dict(THREAD_CREATE), messages=list(ADVISOR_TURNS))),
    Case("thread_continue", _kw(thread=dict(THREAD_CONTINUE))),
    Case(
        "thread_continue_keeps_history_and_skips_its_betas",
        _kw(thread=dict(THREAD_CONTINUE), messages=list(ADVISOR_TURNS)),
    ),
    Case(
        "thread_continue_keeps_param_betas",
        _kw(
            thread=dict(THREAD_CONTINUE),
            messages=list(ADVISOR_TURNS),
            compaction={},
            context_management={"edits": [{"type": "compact_20260112"}]},
        ),
    ),
    Case("thread_continue_without_previous_message_id", _kw(thread={"type": "continue"}, messages=list(ADVISOR_TURNS))),
    Case(
        "thread_continue_with_a_malformed_previous_message_id",
        _kw(thread={"type": "continue", "previous_message_id": 7}, messages=list(ADVISOR_TURNS)),
    ),
    Case("thread_of_unknown_shape", _kw(thread="opaque", messages=list(ADVISOR_TURNS))),
    Case("thread_none_is_left_out", _kw(thread=None)),
    Case(
        "thread_beside_a_client_beta",
        _kw(thread=dict(THREAD_CONTINUE), extra_headers={"anthropic-beta": f"advisor-tool-2026-03-01,{THREADS_BETA}"}),
    ),
    Case(
        "thread_dropped_by_additional_drop_params", _kw(thread=dict(THREAD_CREATE), additional_drop_params=["thread"])
    ),
    Case("thread_create_stream", _kw(thread=dict(THREAD_CREATE), stream=True), upstream=(_stream(TEXT_STREAM),)),
    Case(
        "thread_continue_stream",
        _kw(thread=dict(THREAD_CONTINUE), stream=True, messages=list(ADVISOR_TURNS)),
        upstream=(_stream(TEXT_STREAM),),
    ),
    Case("thread_create_sync", _kw(thread=dict(THREAD_CREATE)), sync=True),
    Case(
        "thread_continue_sync_stream",
        _kw(thread=dict(THREAD_CONTINUE), stream=True),
        upstream=(_stream(TEXT_STREAM),),
        sync=True,
    ),
    Case("thread_not_found", _kw(thread=dict(THREAD_CONTINUE)), upstream=(_thread_not_found(),)),
    Case("thread_not_found_stream", _kw(thread=dict(THREAD_CONTINUE), stream=True), upstream=(_thread_not_found(),)),
    Case("thread_not_found_sync", _kw(thread=dict(THREAD_CONTINUE)), upstream=(_thread_not_found(),), sync=True),
    Case(
        "thread_continue_retried_after_a_rejected_signature",
        _replay(thread=dict(THREAD_CONTINUE)),
        upstream=(_rejected(), _ok()),
    ),
)


class _CallerHandler(AsyncHTTPHandler):
    def __deepcopy__(self, memo: dict[int, object]) -> "_CallerHandler":
        return _CallerHandler()


class _UncheckedHandler(_CallerHandler):
    def __deepcopy__(self, memo: dict[int, object]) -> "_UncheckedHandler":
        return _UncheckedHandler()

    async def post(self, url: str, **kwargs: object) -> httpx.Response:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: the test double ignores the rest of the post
        return httpx.Response(503, text="unavailable")


class _UnreadHandler(_CallerHandler):
    def __deepcopy__(self, memo: dict[int, object]) -> "_UnreadHandler":
        return _UnreadHandler()

    async def post(self, url: str, **kwargs: object) -> httpx.Response:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: the test double ignores the rest of the post
        return httpx.Response(500, stream=httpx.ByteStream(b"unread"), request=httpx.Request("POST", url))


class _CallerHttpxClient(httpx.AsyncClient):
    def __deepcopy__(self, memo: dict[int, object]) -> "_CallerHttpxClient":
        return _CallerHttpxClient()


class _CallerSyncHandler(HTTPHandler):
    def __deepcopy__(self, memo: dict[int, object]) -> "_CallerSyncHandler":
        return _CallerSyncHandler()


def _caller(**kwargs: object) -> Mapping[str, object]:
    return _kw(client=_CallerHandler(), **kwargs)


def _tls(**settings: object) -> Mapping[str, object]:
    return MappingProxyType({**settings, "in_memory_llm_clients_cache": LLMClientCache()})


_TLS_LEVEL: Final = _tls(ssl_security_level="DEFAULT@SECLEVEL=1")
_TLS_CURVE: Final = _tls(ssl_ecdh_curve="secp521r1")

TRANSPORT_CASES: Final = (
    Case("caller_client", _caller()),
    Case("caller_client_stream", _caller(stream=True), upstream=(_stream(TEXT_STREAM),)),
    Case(
        "caller_client_upstream_429",
        _caller(),
        upstream=(
            _failure(429, "rate_limit_error", "slow down", **{"retry-after": "7", "request-id": "req_01Parity"}),
        ),
    ),
    Case(
        "caller_client_stream_upstream_429",
        _caller(stream=True),
        upstream=(_failure(429, "rate_limit_error", "slow down", **{"retry-after": "3"}),),
    ),
    Case(
        "caller_client_non_json_error",
        _caller(),
        upstream=(ResponseSpec(body=b"<html>bad gateway</html>", status=502),),
    ),
    Case("caller_client_invalid_json_success", _caller(), upstream=(ResponseSpec(body=b"not json", status=200),)),
    Case("caller_client_timeout", _caller(timeout=0.5), upstream=(ResponseSpec(body=dict(TEXT_RESPONSE), delay=1.5),)),
    Case(
        "caller_client_stream_read_stalls_past_timeout",
        _caller(stream=True, timeout=0.5),
        upstream=(_stream(SLOW_STREAM, chunk_delay=1.0),),
    ),
    Case(
        "caller_client_thinking_signature_rejected_then_retried",
        _kw(**_replay(), client=_CallerHandler()),
        upstream=(_rejected(), _ok()),
    ),
    Case(
        "caller_client_strict_tools_rejected_then_retried_stream",
        _strict(drop_params=True, stream=True, client=_CallerHandler()),
        upstream=(_rejected(STRICT_REJECTION), _stream(TEXT_STREAM)),
    ),
    Case("caller_client_sync", _caller(), sync=True),
    Case("caller_client_post_returns_unchecked_failure", _kw(client=_UncheckedHandler()), upstream=()),
    Case("caller_client_post_returns_unread_failure", _kw(client=_UnreadHandler(), stream=True), upstream=()),
    Case("caller_httpx_client_is_ignored", _kw(client=_CallerHttpxClient())),
    Case("caller_sync_handler_is_ignored", _kw(client=_CallerSyncHandler())),
    Case("tls_security_level", settings=_TLS_LEVEL),
    Case("tls_security_level_stream", _kw(stream=True), upstream=(_stream(TEXT_STREAM),), settings=_TLS_LEVEL),
    Case("tls_ecdh_curve", settings=_TLS_CURVE),
    Case("tls_ecdh_curve_upstream_500", upstream=(_failure(500, "api_error", "internal"),), settings=_TLS_CURVE),
    Case(
        "tls_security_level_from_environment",
        env=MappingProxyType({"SSL_SECURITY_LEVEL": "DEFAULT@SECLEVEL=1"}),
        settings=_tls(),
    ),
    Case("tls_security_level_python_cannot_apply", upstream=(), settings=_tls(ssl_security_level="NOT-A-CIPHER")),
)


class _RewritingHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        kwargs["tools"] = [dict(WEATHER_TOOL)]
        kwargs["tool_choice"] = {"type": "tool", "name": "get_weather"}
        kwargs["temperature"] = 0.25
        kwargs["top_k"] = 7
        kwargs["system"] = "Answer as a forecaster."
        kwargs["stop_sequences"] = ["STOP"]
        kwargs["metadata"] = {"user_id": "hooked-user"}
        kwargs["extra_headers"] = {"x-hooked": "1"}
        return kwargs


class _InPlaceHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
        messages.append({"role": "user", "content": "And Germany?"})
        return None


class _UnstreamingHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        return {**kwargs, "stream": not kwargs["stream"]}


class _MockingHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        return {**kwargs, "mock_response": "Hi from the hook"}


class _RaisingHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        raise ValueError("the hook refused the request")


class _ModelHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        return {**kwargs, "model": OPUS_ADAPTIVE}


class _CredentialHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        return {**kwargs, "api_key": "sk-ant-hooked"}


def _instance_hook() -> CustomLogger:
    async def hook(model: str, messages: list[dict[str, object]], kwargs: dict[str, object]) -> dict[str, object]:
        return {**kwargs, "temperature": 0.25, "metadata": {"user_id": "instance-hook"}}

    logger: Final = CustomLogger()
    logger.async_pre_request_hook = hook  # pyright: ignore[reportAttributeAccessIssue]  # rebind-ok: an instance-level override, as a caller may install one
    return logger


PRE_REQUEST_HOOK_CASES: Final = (
    Case("pre_request_hook_rewrites_the_request", callbacks=(_RewritingHook,), upstream=(_ok(TOOL_USE_RESPONSE),)),
    Case("pre_request_hook_edits_messages_in_place", callbacks=(_InPlaceHook,)),
    Case(
        "pre_request_hook_after_prompt_caching",
        _kw(cache_control_injection_points=[{"location": "message", "role": "user"}]),
        callbacks=(_InPlaceHook,),
    ),
    Case(
        "pre_request_hook_unstreams",
        _kw(stream=True),
        callbacks=(_UnstreamingHook,),
    ),
    Case(
        "pre_request_hook_unstreams_behind_a_cache",
        _kw(stream=True),
        callbacks=(_UnstreamingHook,),
        upstream=(_ok(), _ok()),
        cache=Cache,
    ),
    Case("pre_request_hook_streams", callbacks=(_UnstreamingHook,), upstream=(_stream(TEXT_STREAM),)),
    Case("pre_request_hook_sets_a_mock", callbacks=(_MockingHook,), upstream=()),
    Case("pre_request_hook_raises", callbacks=(_RaisingHook,), upstream=()),
    Case("pre_request_hook_set_on_the_instance", callbacks=(_instance_hook,)),
    Case("pre_request_hook_names_the_model", callbacks=(_ModelHook,), upstream=()),
    Case("pre_request_hook_names_a_credential", callbacks=(_CredentialHook,), upstream=()),
    Case(
        "pre_request_hook_two_hooks_chain", callbacks=(_RewritingHook, _InPlaceHook), upstream=(_ok(TOOL_USE_RESPONSE),)
    ),
    Case(
        "pre_request_hook_behind_a_cache_hit",
        callbacks=(_RewritingHook,),
        upstream=(_ok(TOOL_USE_RESPONSE), _ok(TOOL_USE_RESPONSE)),
        cache=Cache,
    ),
)

_OBJECTS: Final = TypeAdapter(list[dict[str, object]])
_VOLATILE_HOOK_KWARGS: Final = frozenset({"litellm_call_id", "litellm_logging_obj"})


def _content(response: object) -> list[dict[str, object]]:
    if not isinstance(response, dict):
        return []
    return _OBJECTS.validate_python(_OBJECT.validate_python(response).get("content") or [])


def _tool_calls(response: object) -> list[dict[str, object]]:
    return [block for block in _content(response) if block.get("type") == "tool_use"]


class _FollowUpOnToolUse(CustomLogger):
    """Answers each tool call with a canned result in one follow-up turn, recording what the hooks were handed."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes untyped kwargs
        self.seen: list[Mapping[str, object]] = []

    async def async_should_run_agentic_loop(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self,
        response: object,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict[str, object],
    ) -> tuple[bool, dict[str, object]]:
        calls: Final = _tool_calls(response)
        self.seen.append(
            {
                "hook": "should_run",
                "response": _OBJECT.validate_python({**response, "id": "<id>"}) if isinstance(response, dict) else None,
                "model": model,
                "messages": _OBJECTS.validate_python(messages),
                "tools": tools,
                "stream": stream,
                "custom_llm_provider": custom_llm_provider,
                "kwargs": sorted(name for name in kwargs if name not in _VOLATILE_HOOK_KWARGS),
                "api_key": kwargs.get("api_key"),
            }
        )
        return bool(calls), {"tool_calls": calls}

    async def async_build_agentic_loop_plan(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self,
        tools: dict[str, object],
        model: str,
        messages: list[dict[str, object]],
        response: object,
        anthropic_messages_provider_config: object,
        anthropic_messages_optional_request_params: dict[str, object],
        logging_obj: Logging,
        stream: bool,
        kwargs: dict[str, object],
    ) -> AgenticLoopPlan:
        self.seen.append(
            {
                "hook": "build_plan",
                "optional_params": anthropic_messages_optional_request_params,
                "config": type(anthropic_messages_provider_config).__name__,
                "agentic_loop_params": logging_obj.model_call_details.get("agentic_loop_params"),
                "kwargs": sorted(name for name in kwargs if name not in _VOLATILE_HOOK_KWARGS),
            }
        )
        return self.plan(messages, response)

    def plan(self, messages: list[dict[str, object]], response: object) -> AgenticLoopPlan:
        answers: Final = [
            {"type": "tool_result", "tool_use_id": call["id"], "content": "18C, sunny"}
            for call in _tool_calls(response)
        ]
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                messages=[
                    *messages,
                    {"role": "assistant", "content": _content(response)},
                    {"role": "user", "content": answers},
                ]
            ),
        )


class _HeldBack(_FollowUpOnToolUse):
    server_fulfilled_tool_names: ClassVar[frozenset[str]] = frozenset({"get_weather"})


class _Terminates(_FollowUpOnToolUse):
    def plan(self, messages: list[dict[str, object]], response: object) -> AgenticLoopPlan:
        return AgenticLoopPlan(terminate=True, stop_reason="answered")


class _Overrides(_FollowUpOnToolUse):
    def plan(self, messages: list[dict[str, object]], response: object) -> AgenticLoopPlan:
        return AgenticLoopPlan(response_override={**TEXT_RESPONSE, "id": "msg_01Override"})


class _Fails(_FollowUpOnToolUse):
    async def async_build_agentic_loop_plan(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self,
        tools: dict[str, object],
        model: str,
        messages: list[dict[str, object]],
        response: object,
        anthropic_messages_provider_config: object,
        anthropic_messages_optional_request_params: dict[str, object],
        logging_obj: Logging,
        stream: bool,
        kwargs: dict[str, object],
    ) -> AgenticLoopPlan:
        raise ValueError("the agentic plan failed")


class _WebSearch(WebSearchInterceptionLogger):
    """The stock web search interception, recording what its pre-request hook leaves for its agentic hook."""

    def __init__(self, enabled_providers: list[str] | None = None, max_agentic_loops: int | None = None) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # the logger annotates bare containers
            enabled_providers=list(enabled_providers) if enabled_providers is not None else None,
            max_agentic_loops=max_agentic_loops,
        )
        self.seen: list[Mapping[str, object]] = []

    async def async_should_run_agentic_loop(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self,
        response: object,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict[str, object],
    ) -> tuple[bool, dict[str, object]]:
        self.seen.append(
            {
                "model": model,
                "messages": _OBJECTS.validate_python(messages),
                "tools": tools,
                "stream": stream,
                "custom_llm_provider": custom_llm_provider,
                "kwargs": {
                    name: kwargs[name]
                    for name in sorted(kwargs)
                    if name == "max_agentic_loops" or name.startswith("_websearch_interception")
                },
                "kwarg_names": sorted(name for name in kwargs if name not in _VOLATILE_HOOK_KWARGS),
            }
        )
        return await super().async_should_run_agentic_loop(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the logger annotates bare containers
            response=response,
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            kwargs=kwargs,
        )


def _loggers(*kinds: Callable[[], CustomLogger]) -> Callable[[], Sequence[CustomLogger]]:
    return lambda: tuple(kind() for kind in kinds)


def _web_search() -> CustomLogger:
    return _WebSearch(enabled_providers=["anthropic"], max_agentic_loops=2)


async def _search(*args: object, **kwargs: object) -> SearchResponse:  # kwargs-ok: litellm.asearch's call shape
    return SearchResponse.model_validate(
        {"results": [{"title": "Paris", "url": "https://example.com/paris", "snippet": "Paris is the capital."}]}
    )


WEB_SEARCH_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
)
WEB_SEARCH_CALL_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "id": "msg_01ParitySearch",
        "content": [
            {"type": "tool_use", "id": "toolu_01Search", "name": "litellm_web_search", "input": {"query": "capital"}}
        ],
        "stop_reason": "tool_use",
    }
)
_WEB_SEARCH_SETTINGS: Final[Mapping[str, object]] = MappingProxyType({"asearch": _search})
_WEATHER: Final = _kw(tools=[dict(WEATHER_TOOL)])

AGENTIC_CASES: Final = (
    Case("agentic_loop_without_tool_use", _WEATHER, loggers=_loggers(_FollowUpOnToolUse)),
    Case(
        "agentic_loop_follow_up",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE), _ok()),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_follow_up_with_request_params",
        _kw(
            tools=[dict(WEATHER_TOOL)],
            system="Answer in one word.",
            temperature=0.3,
            metadata={"user_id": "user-123"},
            thinking={"type": "enabled", "budget_tokens": 2048},
            max_tokens=4096,
            messages=[{"role": "user", "content": "Weather?", "provider_specific_fields": {"x": 1}}],
        ),
        upstream=(_ok(TOOL_USE_RESPONSE), _ok()),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_follow_up_stream",
        _kw(stream=True, tools=[dict(WEATHER_TOOL)]),
        upstream=(_stream(TOOL_STREAM), _stream(TEXT_STREAM)),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_stream_without_tool_use",
        _kw(stream=True, tools=[dict(WEATHER_TOOL)]),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_holds_back_a_server_fulfilled_tool",
        _kw(stream=True, tools=[dict(WEATHER_TOOL)]),
        upstream=(_stream(TOOL_STREAM), _stream(TEXT_STREAM)),
        loggers=_loggers(_HeldBack),
    ),
    Case(
        "agentic_loop_terminates",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE),),
        loggers=_loggers(_Terminates),
    ),
    Case(
        "agentic_loop_response_override",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE),),
        loggers=_loggers(_Overrides),
    ),
    Case(
        "agentic_loop_plan_fails",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE),),
        loggers=_loggers(_Fails),
    ),
    Case(
        "agentic_loop_follow_up_fails",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE), _failure(529, "overloaded_error", "Overloaded")),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_refused_past_max_loops",
        _kw(tools=[dict(WEATHER_TOOL)], max_agentic_loops=1),
        upstream=(_ok(TOOL_USE_RESPONSE), _ok(TOOL_USE_RESPONSE)),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_upstream_error_skips_the_hooks",
        _WEATHER,
        upstream=(_failure(400, "invalid_request_error", "bad"),),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "web_search_interception",
        _kw(tools=[dict(WEB_SEARCH_TOOL)]),
        upstream=(_ok(WEB_SEARCH_CALL_RESPONSE), _ok()),
        settings=_WEB_SEARCH_SETTINGS,
        loggers=_loggers(_web_search),
    ),
    Case(
        "web_search_interception_converted_stream",
        _kw(stream=True, tools=[dict(WEB_SEARCH_TOOL)]),
        upstream=(_ok(WEB_SEARCH_CALL_RESPONSE), _ok()),
        settings=_WEB_SEARCH_SETTINGS,
        loggers=_loggers(_web_search),
    ),
    Case(
        "web_search_interception_without_a_search_call",
        _kw(stream=True, tools=[dict(WEB_SEARCH_TOOL)]),
        settings=_WEB_SEARCH_SETTINGS,
        loggers=_loggers(_web_search),
    ),
    Case(
        "web_search_interception_for_a_provider_it_skips",
        _kw(tools=[dict(WEB_SEARCH_TOOL)]),
        loggers=_loggers(_WebSearch),
    ),
    Case(
        "web_search_interception_rewrites_the_messages_it_hands_the_loop",
        _kw(
            stream=True,
            tools=[dict(WEB_SEARCH_TOOL)],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "Capital?"}]}
            ],
        ),
        upstream=(_ok(WEB_SEARCH_CALL_RESPONSE), _ok()),
        settings=_WEB_SEARCH_SETTINGS,
        loggers=_loggers(_web_search),
    ),
)

CLASSIFIER_MODEL: Final = "openai/parity-classifier"
CLASSIFIER_CALL_TYPE: Final = "acompletion"
NON_CLAUDE: Final = "anthropic/parity-tool-model"
DANGEROUS_TOOL_USE: Final[tuple[JsonValue, ...]] = (
    {
        "type": "dangerous_tool_use",
        "classifier_context": {
            "permission_mode": "auto",
            "platform": "linux",
            "trusted_directories": ["/repo"],
            "git_state": {"cwd": "/repo", "branch": "main", "dirty": True},
            "ignored": "not rendered",
        },
    },
)
FLAGGED: Final = json.dumps({"verdicts": {"toolu_01Parity": {"flagged": True, "explanation": "Sends the weather."}}})
CLEARED: Final = json.dumps({"verdicts": {"toolu_01Parity": {"flagged": False}}})
CUT_OFF_TOOL_STREAM: Final = (*TOOL_STREAM[:-2], *_message_end("max_tokens", 22))
UPSTREAM_ANSWERED: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TOOL_USE_RESPONSE,
        "safeguard_results": [{"type": "dangerous_tool_use", "status": {"type": "unsupported"}}],
    }
)


def _safeguarded(**kwargs: object) -> Mapping[str, object]:
    return _kw(safeguards=list(DANGEROUS_TOOL_USE), messages=list(TOOL_TURNS), tools=[dict(WEATHER_TOOL)], **kwargs)


SAFEGUARDS_CASES: Final = (
    Case(
        "safeguards_classifier_flags_tool_use",
        _safeguarded(litellm_metadata={"user_api_key_end_user_id": "end-user-1", "user_api_key": "hashed-key"}),
        upstream=(_ok(TOOL_USE_RESPONSE),),
        model=NON_CLAUDE,
        classifier=FLAGGED,
    ),
    Case(
        "safeguards_classifier_clears_tool_use",
        _safeguarded(),
        upstream=(_ok(TOOL_USE_RESPONSE),),
        model=NON_CLAUDE,
        classifier=CLEARED,
    ),
    Case(
        "safeguards_classifier_unparsable_reply",
        _safeguarded(),
        upstream=(_ok(TOOL_USE_RESPONSE),),
        model=NON_CLAUDE,
        classifier="I cannot tell.",
    ),
    Case(
        "safeguards_classifier_rejects_long_input",
        _safeguarded(),
        upstream=(_ok(TOOL_USE_RESPONSE),),
        model=NON_CLAUDE,
        classifier="litellm.ContextWindowExceededError",
    ),
    Case("safeguards_classifier_without_tool_use", _safeguarded(), model=NON_CLAUDE, classifier=FLAGGED),
    Case(
        "safeguards_classifier_upstream_answered",
        _safeguarded(),
        upstream=(_ok(UPSTREAM_ANSWERED),),
        model=NON_CLAUDE,
        classifier=FLAGGED,
    ),
    Case(
        "safeguards_classifier_stream_tool_use",
        _safeguarded(stream=True),
        upstream=(_stream(TOOL_STREAM),),
        model=NON_CLAUDE,
        classifier=FLAGGED,
    ),
    Case(
        "safeguards_classifier_stream_cut_off",
        _safeguarded(stream=True),
        upstream=(_stream(CUT_OFF_TOOL_STREAM),),
        model=NON_CLAUDE,
        classifier=FLAGGED,
    ),
    Case(
        "safeguards_classifier_stream_text",
        _safeguarded(stream=True),
        upstream=(_stream(TEXT_STREAM),),
        model=NON_CLAUDE,
        classifier=FLAGGED,
    ),
    Case(
        "safeguards_classifier_upstream_failure",
        _safeguarded(),
        upstream=(_failure(529, "overloaded_error", "Overloaded"),),
        model=NON_CLAUDE,
        classifier=FLAGGED,
    ),
    Case(
        "safeguards_claude_model_is_not_classified",
        _safeguarded(),
        upstream=(_ok(TOOL_USE_RESPONSE),),
        classifier=FLAGGED,
    ),
    Case("safeguards_classifier_sync", _safeguarded(), model=NON_CLAUDE, sync=True, classifier=FLAGGED),
)

CASES: Final = (
    *REQUEST_CASES,
    *AGENTIC_CASES,
    *PROMPT_CACHING_CASES,
    *SYNC_PROMPT_CACHING_CASES,
    *RESPONSE_CASES,
    *FAILURE_CASES,
    *STREAM_CASES,
    *SYNC_CASES,
    *SERVED_NATIVELY_CASES,
    *THINKING_RETRY_CASES,
    *STRICT_TOOLS_RETRY_CASES,
    *FALLBACK_CASES,
    *CACHE_CASES,
    *THREAD_CASES,
    *TRANSPORT_CASES,
    *PRE_REQUEST_HOOK_CASES,
    *SAFEGUARDS_CASES,
)


@dataclass(frozen=True, slots=True)
class Leg:
    fell_back: bool
    requests: tuple[Mapping[str, object], ...]
    outcome: Mapping[str, object]
    hidden_params: Mapping[str, object]
    logged: Mapping[str, object]
    callback_kwargs: Mapping[str, object]
    cached: tuple[str, ...] = ()
    hook_inputs: tuple[object, ...] = ()
    classified: tuple[Mapping[str, object], ...] = ()


@dataclass(slots=True)
class FallbackProbe:
    calls: int = 0


_PYTHON_AMESSAGES: Final = cast(Callable[..., Awaitable[object]], dispatch._PYTHON_AMESSAGES)  # pyright: ignore[reportPrivateUsage]  # the Python baseline under comparison
_PYTHON_MESSAGES: Final = cast(Callable[..., object], dispatch._PYTHON_MESSAGES)  # pyright: ignore[reportPrivateUsage]  # the Python baseline under comparison
_PUBLIC_AMESSAGES: Final = cast(Callable[..., Awaitable[object]], dispatch.anthropic_messages)
_PUBLIC_MESSAGES: Final = cast(Callable[..., object], dispatch.anthropic_messages_handler)


@pytest.fixture
def native_messages() -> Iterator[FallbackProbe]:
    probe: Final = FallbackProbe()

    async def python_afallback(*args: object, **kwargs: object) -> object:
        probe.calls += 1
        return await _PYTHON_AMESSAGES(*args, **kwargs)

    def python_fallback(*args: object, **kwargs: object) -> object:
        probe.calls += 1
        return _PYTHON_MESSAGES(*args, **kwargs)

    with (
        rebound(dispatch, "_PYTHON_AMESSAGES", python_afallback),
        rebound(dispatch, "_PYTHON_MESSAGES", python_fallback),
    ):
        yield probe


@contextmanager
def _environment(env: Mapping[str, str]) -> Generator[None]:
    # LITELLM_RUST overrides the catalog rule, so a shell that sets it would route the native leg to Python.
    names: Final = (
        *env,
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "ANTHROPIC_IDENTITY_SOURCE",
        "LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS",
        "LITELLM_REASONING_AUTO_SUMMARY",
        "LITELLM_RUST",
    )
    with pytest.MonkeyPatch.context() as patch:
        for name in names:
            patch.delenv(name, raising=False)
        for name, value in env.items():
            patch.setenv(name, value)
        yield


@contextmanager
def _settings(settings: Mapping[str, object]) -> Generator[None]:
    saved: Final[dict[str, object]] = {name: cast(object, getattr(litellm, name)) for name in settings}
    for name, value in settings.items():
        setattr(litellm, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(litellm, name, value)


@contextmanager
def _classifier(reply: str | None, calls: list[Mapping[str, object]]) -> Generator[None]:
    if reply is None:
        yield
        return
    acompletion: Final = litellm.acompletion

    async def classify(**kwargs: object) -> object:  # kwargs-ok: stands in for litellm.acompletion
        calls.append(
            {
                "names": sorted(kwargs),
                **{name: kwargs.get(name) for name in ("model", "messages", "max_tokens", "num_retries", "user")},
                "litellm_metadata": _mapping(kwargs.get("litellm_metadata")),
            }
        )
        return await acompletion(**kwargs, mock_response=reply)

    with (
        rebound(proxy_server, "general_settings", {"safeguards_classifier_model": CLASSIFIER_MODEL}),
        rebound(litellm, "acompletion", classify),
    ):
        yield


@contextmanager
def _cache(cache: Callable[[], Cache] | None) -> Generator[None]:
    if cache is None:
        yield
        return
    with rebound(litellm, "cache", cache()):
        yield


def _kwargs(case: Case, server: RecordingServer) -> dict[str, object]:
    # Deep: Python writes into the caller's own `litellm_metadata`, which must not leak into the
    # other leg.
    supplied: Final = copy.deepcopy(dict(case.kwargs))
    suffix: Final = supplied.pop("api_base_suffix", "")
    kwargs: Final[dict[str, object]] = {
        "model": case.model,
        "max_tokens": 64,
        "messages": [dict(USER)],
        "api_key": API_KEY,
        "api_base": f"{server.base_url}{suffix}",
        **supplied,
    }
    return {name: value for name, value in kwargs.items() if value is not None or name in ("safeguards",)}


def _mapping(value: object) -> Mapping[str, object]:
    return _LOGGED.validate_python(value) if isinstance(value, dict) else {}


def _normalized(text: str) -> str:
    for pattern, replacement in _VOLATILE:
        text = pattern.sub(replacement, text)  # rebind-ok: successive substitutions
    return text


def _error(error: BaseException, phase: str, chunks: Iterable[object] = ()) -> Mapping[str, object]:
    header_items: Final = {
        name.lower(): value for name, value in _mapping(getattr(error, "litellm_response_headers", None)).items()
    }
    return {
        "kind": "error",
        "phase": phase,
        "class": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "llm_provider": getattr(error, "llm_provider", None),
        "model": getattr(error, "model", None),
        "message": _normalized(str(error)),
        "details": AnthropicExceptionMapping.embedded_error_details(str(error)),
        "keeps_the_deployment_warm": is_message_thread_not_found(error),
        "headers": {name: header_items.get(name) for name in ("retry-after", "request-id") if name in header_items},
        "partial": _stream_text(chunks),
    }


def _sent_request(response: object) -> Mapping[str, object] | None:
    if not isinstance(response, httpx.Response):
        return None
    try:
        sent: Final = response.request
    except RuntimeError:
        return {"status": response.status_code, "request": None}
    return {
        "status": response.status_code,
        "request": {
            "method": sent.method,
            "path": sent.url.path,
            "anthropic-beta": sent.headers.get("anthropic-beta"),
            "body": json.loads(sent.content),
        },
    }


def _stream_text(chunks: Iterable[object]) -> str:
    return b"".join(chunk if isinstance(chunk, bytes) else str(chunk).encode() for chunk in chunks).decode()


def _scrubbed(value: object) -> object:
    """Masks the random server tool ids inside a response, leaving every other value as it was."""
    match value:
        case str():
            return _SERVER_TOOL_ID.sub("srvtoolu_<id>", value)
        case dict():
            return {name: _scrubbed(item) for name, item in cast(dict[object, object], value).items()}
        case list():
            return [_scrubbed(item) for item in cast(list[object], value)]
        case _:
            return value


def _response(result: object, private_keys: bool) -> Mapping[str, object]:
    assert isinstance(result, dict), f"expected a dict response, got {type(result).__name__}"
    body: Final = {
        name: value
        for name, value in cast(dict[str, object], result).items()
        if private_keys or not name.startswith("_")
    }
    return {"kind": "response", "body": _LOGGED.validate_python(_scrubbed(body))}


def _hidden_params(result: object) -> Mapping[str, object]:
    hidden: Final = dict(_mapping(get_hidden_params_dict(result)))
    headers: Final = _mapping(hidden.pop("additional_headers", None))
    header_names: Final = sorted(name for name in headers if name not in ("x-litellm-rust", "x-litellm-cache-key"))
    return {"keys": sorted(hidden), "additional_headers": header_names}


async def _consume_async(stream: AsyncIterator[object]) -> tuple[list[object], BaseException | None]:
    chunks: Final[list[object]] = []
    try:
        async for chunk in stream:
            chunks.append(chunk)
    except Exception as error:  # noqa: BLE001  # the raised error is part of the compared outcome
        return chunks, error
    return chunks, None


def _consume_sync(stream: Iterator[object]) -> tuple[list[object], BaseException | None]:
    chunks: Final[list[object]] = []
    try:
        for chunk in stream:
            chunks.append(chunk)
    except Exception as error:  # noqa: BLE001  # the raised error is part of the compared outcome
        return chunks, error
    return chunks, None


async def _invoke(
    case: Case, kwargs: Mapping[str, object], native: bool
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if case.sync:
        call = _PUBLIC_MESSAGES if native else _PYTHON_MESSAGES
        try:
            result = await asyncio.to_thread(lambda: call(**kwargs))
        except Exception as error:  # noqa: BLE001  # the raised error is part of the compared outcome
            return _error(error, "call"), {}
        if isinstance(result, dict):
            body = cast(Mapping[str, object], result)
            return _response(body, case.private_keys), _hidden_params(body)
        chunks, failure = await asyncio.to_thread(_consume_sync, cast(Iterator[object], result))
    else:
        acall = _PUBLIC_AMESSAGES if native else _PYTHON_AMESSAGES
        try:
            result = await acall(**kwargs)
        except Exception as error:  # noqa: BLE001  # the raised error is part of the compared outcome
            return _error(error, "call"), {}
        if isinstance(result, dict):
            body = cast(Mapping[str, object], result)
            return _response(body, case.private_keys), _hidden_params(body)
        chunks, failure = await _consume_async(cast(AsyncIterator[object], result))
    hidden: Final = _hidden_params(result)
    if failure is not None:
        return _error(failure, "stream", chunks), hidden
    return {
        "kind": "stream",
        "chunk_types": sorted({type(chunk).__name__ for chunk in chunks}),
        "text": _SERVER_TOOL_ID.sub("srvtoolu_<id>", _stream_text(chunks)),
    }, hidden


async def _terminal_events(recorder: RecordingLogger, since: int = 0) -> tuple[str, ...]:
    for _ in range(100):
        await drain_logging()
        if any(name in _TERMINAL_EVENTS for name in recorder.names[since:]):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)
    await drain_logging()
    return tuple(name for name in recorder.names[since:] if name in _TERMINAL_EVENTS)


def _logged(
    recorder: RecordingLogger, events: tuple[str, ...], since: int = 0
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    terminal: Final = [
        event
        for event in recorder.events[since:]
        if event.name in _TERMINAL_EVENTS and event.call_type != CLASSIFIER_CALL_TYPE
    ]
    hooks: Final = tuple(sorted(set(recorder.names[since:])))
    if not terminal:
        return {"events": events, "hooks": hooks}, {}
    details: Final = _mapping(terminal[-1].kwargs)
    standard: Final = _mapping(details.get("standard_logging_object"))
    error_class: Final = _mapping(standard.get("error_information")).get("error_class")
    litellm_params: Final = _mapping(details.get("litellm_params"))
    metadata: Final = litellm_params.get("metadata")
    metadata_keys: Final = _mapping(metadata).keys()
    logged: Final[Mapping[str, object]] = {
        "events": events,
        "hooks": hooks,
        **{name: standard.get(name) for name in _COMPARED_LOG_FIELDS},
        "error_class": error_class,
    }
    callback_kwargs: Final[Mapping[str, object]] = {
        "logged_metadata": sorted(metadata_keys) if isinstance(metadata, dict) else metadata,
        "logged_model_info": litellm_params.get("model_info"),
        "logged_response_headers": bool(details.get("response_headers")),
        "logged_api_key": details.get("api_key"),
        "logged_call": {
            "custom_llm_provider": details.get("custom_llm_provider"),
            "model": details.get("model"),
            "stream": details.get("stream"),
            "aanthropic_messages": litellm_params.get("aanthropic_messages"),
        },
        "logged_request": {
            "messages": details.get("messages"),
            "thinking": details.get("thinking"),
            "tools": details.get("tools"),
            "complete_input_dict": _mapping(details.get("additional_args")).get("complete_input_dict"),
        },
        "logged_sent_request": _sent_request(details.get("httpx_response")),
    }
    return logged, callback_kwargs


async def _prime(case: Case, server: RecordingServer, recorder: RecordingLogger, native: bool) -> int:
    if case.cache is None:
        return 0
    outcome: Final = await _invoke(case, _kwargs(case, server), native=native)
    assert case.sync or outcome[0]["kind"] != "error", outcome
    await _terminal_events(recorder)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    return len(recorder.events)


async def _leg(case: Case, server: RecordingServer, probe: FallbackProbe | None) -> Leg:
    classified: Final[list[Mapping[str, object]]] = []
    with (
        isolated_callback_registries(),
        _environment(case.env),
        _settings(case.settings),
        _cache(case.cache),
        _classifier(case.classifier, classified),
    ):
        for spec in case.upstream:
            server.enqueue(replace(spec))
        request_count: Final = len(server.requests)
        recorder: Final = RecordingLogger()
        loggers: Final = () if case.loggers is None else case.loggers()
        litellm.callbacks.extend(loggers)
        litellm.callbacks.append(recorder)
        litellm.callbacks.extend(callback() for callback in case.callbacks)
        fallbacks_before: Final = 0 if probe is None else probe.calls
        primed: Final = await _prime(case, server, recorder, native=probe is not None)
        outcome, hidden = await _invoke(case, _kwargs(case, server), native=probe is not None)
        events: Final = await _terminal_events(recorder, primed)
        server.responses.clear()
        logged, callback_kwargs = _logged(recorder, events, primed)
        return Leg(
            fell_back=probe is not None and probe.calls > fallbacks_before,
            requests=tuple(
                {
                    "method": sent.method,
                    "path": sent.path,
                    "body": sent.body if sent.body is not None else sent.raw_body.decode(errors="replace"),
                    "headers": {
                        name: value for name, value in sorted(sent.headers.items()) if name not in _TRANSPORT_HEADERS
                    },
                }
                for sent in server.requests[request_count:]
            ),
            outcome=outcome,
            hidden_params=hidden,
            logged=logged,
            callback_kwargs=callback_kwargs,
            cached=() if litellm.cache is None else tuple(sorted(litellm.cache.cache.cache_dict)),
            hook_inputs=tuple(getattr(logger, "seen", ()) for logger in loggers),
            classified=tuple(classified),
        )


@pytest.fixture
def parity_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.expected_requests = None
    return recording_server


_LEGS: Final[dict[str, tuple[Leg, Leg]]] = {}


async def _legs(case: Case, server: RecordingServer, probe: FallbackProbe) -> tuple[Leg, Leg]:
    if case.name not in _LEGS:
        _LEGS[case.name] = (await _leg(case, server, probe=None), await _leg(case, server, probe=probe))
        _dump(case.name, *_LEGS[case.name])
    return _LEGS[case.name]


def _dump(name: str, python: Leg, native: Leg) -> None:
    """Writes both legs to ``$MESSAGES_PARITY_DUMP/<case>.json`` when that variable is set."""
    dump: Final = os.environ.get("MESSAGES_PARITY_DUMP")
    if not dump:
        return
    os.makedirs(dump, exist_ok=True)
    with open(os.path.join(dump, f"{name}.json"), "w") as out:
        json.dump({"python": asdict(python), "native": asdict(native)}, out, indent=1, default=repr)


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "aspect"), tuple(param for case in CASES for param in case.params()))
async def test_native_messages_match_python(
    case: Case, aspect: str, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python, native = await _legs(case, parity_server, native_messages)
    match aspect:
        case "dispatch":
            assert not native.fell_back
        case "request":
            assert native.requests == python.requests
        case "outcome":
            assert native.outcome == python.outcome
        case "hidden_params":
            assert native.hidden_params == python.hidden_params
        case "logged":
            assert native.logged == python.logged
        case "cached":
            assert native.cached == python.cached
        case "hook_inputs":
            assert native.hook_inputs == python.hook_inputs
        case "classified":
            assert native.classified == python.classified
        case _:
            assert native.callback_kwargs.get(aspect) == python.callback_kwargs.get(aspect)


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.name) for case in (*SERVED_NATIVELY_CASES, *AGENTIC_CASES, *SAFEGUARDS_CASES)],
)
def test_catalog_serves_the_case_natively(case: Case, parity_server: RecordingServer) -> None:
    with (
        isolated_callback_registries(),
        _environment(case.env),
        _settings(case.settings),
        _classifier(case.classifier, []),
    ):
        litellm.callbacks.extend(() if case.loggers is None else case.loggers())
        binding: Final = dispatch._MESSAGES if case.sync else dispatch._AMESSAGES  # pyright: ignore[reportPrivateUsage]  # the public binding the entry routes by
        features: Final = dispatch._FEATURES if case.sync else dispatch._ASYNC_FEATURES  # pyright: ignore[reportPrivateUsage]  # the entry's own detectors
        request: Final = dispatch._public_request(binding, (), _kwargs(case, parity_server))  # pyright: ignore[reportPrivateUsage]  # the public binding the entry routes by
        assert request is not None
        context: Final = dispatch._context(request, features)  # pyright: ignore[reportPrivateUsage]  # the entry's own route context
        assert catalog.rollout(context) is Rollout.RUST_REQUIRED, context.features


@pytest.mark.parametrize("case", [pytest.param(case, id=case.name) for case in TRANSPORT_CASES])
def test_catalog_serves_the_transport_case_natively(case: Case, parity_server: RecordingServer) -> None:
    with isolated_callback_registries(), _environment(case.env), _settings(case.settings):
        request: Final = dispatch._public_request(dispatch._AMESSAGES, (), _kwargs(case, parity_server))  # pyright: ignore[reportPrivateUsage]  # the public binding the async entry routes by
        assert request is not None
        context: Final = dispatch._context(request, dispatch._ASYNC_FEATURES)  # pyright: ignore[reportPrivateUsage]  # the async entry's own route context
        assert catalog.rollout(context) is Rollout.RUST_REQUIRED, context.features


@dataclass(slots=True)
class _Exchanges:
    posts: int = 0
    windows: int = 0


async def _exchanges(
    case: Case, server: RecordingServer, probe: FallbackProbe | None, monkeypatch: pytest.MonkeyPatch
) -> _Exchanges:
    counted: Final = _Exchanges()
    post: Final = AsyncHTTPHandler.post  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the handler's untyped dict params
    record: Final = logging_utils._set_duration_in_model_call_details  # pyright: ignore[reportPrivateUsage]  # the one recorder of an exchange window

    async def posted(
        handler: AsyncHTTPHandler, *args: object, **kwargs: object
    ) -> object:  # kwargs-ok: the handler's own post signature
        counted.posts += 1
        return await post(handler, *args, **kwargs)  # pyright: ignore[reportArgumentType]  # forwarded unchanged

    def windowed(**kwargs: object) -> None:  # kwargs-ok: the recorder's own keywords
        counted.windows += 1
        record(**kwargs)  # pyright: ignore[reportArgumentType]  # forwarded unchanged

    with monkeypatch.context() as patch:
        patch.setattr(AsyncHTTPHandler, "post", posted)
        patch.setattr(logging_utils, "_set_duration_in_model_call_details", windowed)
        await _leg(case, server, probe)
    return counted


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [pytest.param(case, id=case.name) for case in TRANSPORT_CASES if not case.sync])
async def test_transport_cases_post_through_the_handler_python_posts_through(
    case: Case, parity_server: RecordingServer, native_messages: FallbackProbe, monkeypatch: pytest.MonkeyPatch
) -> None:
    python: Final = await _exchanges(case, parity_server, None, monkeypatch)
    native: Final = await _exchanges(case, parity_server, native_messages, monkeypatch)
    ignored: Final = case.name.endswith("_is_ignored")
    assert python.posts == len(case.upstream)
    assert native.posts == (0 if ignored else python.posts)
    assert native.windows == python.windows
    assert native_messages.calls == 0


_COST_CASES: Final = tuple(
    pytest.param(case, id=case.name) for case in SERVED_NATIVELY_CASES if case.name.startswith("stream_cost_injection")
)


def _proxy_relayed(text: str, request_headers: Mapping[str, object]) -> str:
    return "".join(
        str(
            ProxyBaseLLMRequestProcessing.process_chunk_with_cost_injection(
                frame + "\n\n", "proxy-model-alias", None, request_headers=request_headers
            )
        )
        for frame in text.split("\n\n")
        if frame
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _COST_CASES)
async def test_both_lanes_carry_one_cost_per_usage_through_the_proxy_relay(
    case: Case, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python, native = await _legs(case, parity_server, native_messages)
    opted_in: Final = {"x-litellm-include-cost-in-streaming-usage": "true"}
    python_text: Final = str(python.outcome["text"])
    native_text: Final = str(native.outcome["text"])
    assert '"cost"' in python_text
    assert native_text == python_text
    relayed: Final = _proxy_relayed(native_text, opted_in)
    assert relayed == _proxy_relayed(python_text, opted_in)
    assert all(frame.count('"cost"') <= 1 for frame in relayed.split("\n\n"))


def _proxied(kwargs: Mapping[str, object]) -> dict[str, object]:
    logging_obj, data = litellm.utils.function_setup(
        original_function="anthropic_messages",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.datetime.now(),
        is_async_call=True,
        litellm_call_id=str(uuid.uuid4()),
        **kwargs,
    )
    return {**data, "litellm_logging_obj": logging_obj}


async def _through_the_ownership_hook(
    hook: MessageThreadOwnership, owner: UserAPIKeyAuth, data: dict[str, object], native: bool
) -> None:
    acall: Final = _PUBLIC_AMESSAGES if native else _PYTHON_AMESSAGES
    result: Final = await acall(**data)
    if isinstance(result, dict):
        await hook.async_post_call_success_hook(data, owner, cast(AnthropicMessagesResponse, result))
        return
    async for _ in hook.async_post_call_streaming_iterator_hook(owner, cast(AsyncIterator[object], result), data):
        pass


async def _thread_betas(server: RecordingServer, stream: bool, native: bool) -> tuple[str | None, str | None]:
    hook: Final = MessageThreadOwnership(InternalUsageCache(dual_cache=DualCache()))
    owner: Final = UserAPIKeyAuth(token="hashed-owner")
    base: Final[dict[str, object]] = {
        "model": SONNET,
        "max_tokens": 64,
        "messages": [dict(USER)],
        "api_key": API_KEY,
        "api_base": server.base_url,
    }
    server.enqueue(_stream(TEXT_STREAM) if stream else _ok())
    server.enqueue(_ok())
    sent: Final = len(server.requests)
    with isolated_callback_registries(), _environment({}):
        await _through_the_ownership_hook(
            hook,
            owner,
            _proxied({**base, "thread": dict(THREAD_CREATE), "tools": [dict(ADVISOR_TOOL)], "stream": stream}),
            native,
        )
        replied: Final = "msg_01ParityStream" if stream else str(TEXT_RESPONSE["id"])
        continued: Final = _proxied({**base, "thread": {"type": "continue", "previous_message_id": replied}})
        hooked: Final = await hook.async_pre_call_hook(owner, DualCache(), continued, "anthropic_messages")
        await _through_the_ownership_hook(hook, owner, continued if hooked is None else hooked, native)
        await drain_logging()
    server.responses.clear()
    create, resumed = server.requests[sent:]
    return create.headers.get("anthropic-beta"), resumed.headers.get("anthropic-beta")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True), ids=("json", "streamed"))
async def test_both_lanes_resend_a_thread_s_create_betas_through_the_ownership_hook(
    stream: bool, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python: Final = await _thread_betas(parity_server, stream, native=False)
    native: Final = await _thread_betas(parity_server, stream, native=True)

    assert native_messages.calls == 0
    assert native == python
    assert python == ("advisor-tool-2026-03-01,message-threads-2026-08-12",) * 2


_WEB_SEARCH_PRE_HOOK_CASES: Final = tuple(
    pytest.param(case, id=case.name)
    for case in AGENTIC_CASES
    if case.name in {"web_search_interception", "web_search_interception_converted_stream"}
)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _WEB_SEARCH_PRE_HOOK_CASES)
async def test_both_lanes_hand_the_web_search_pre_hook_kwargs_to_the_agentic_loop(
    case: Case, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python, native = await _legs(case, parity_server, native_messages)

    assert native_messages.calls == 0
    assert native.hook_inputs == python.hook_inputs
    [seen] = python.hook_inputs
    first: Final = cast(list[Mapping[str, object]], seen)[0]
    converted: Final = case.name.endswith("_converted_stream")
    assert first["kwargs"] == {
        "max_agentic_loops": 2,
        "_websearch_interception_emit_native_blocks": True,
        **({"_websearch_interception_converted_stream": True} if converted else {}),
    }
    assert [tool.get("name") for tool in cast(list[Mapping[str, object]], first["tools"])] == ["litellm_web_search"]
    assert first["stream"] is False
