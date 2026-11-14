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
import contextlib
import copy
import datetime
import gzip
import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, Final, cast

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.anthropic_interface.exceptions.exception_mapping_utils import AnthropicExceptionMapping
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # a primed entry must land before the measured call reads it
)
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.litellm_core_utils import logging_utils
from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.messages.native import interceptors as messages_interceptors
from litellm.llms.anthropic.messages.native.fake_stream_iterator import FakeAnthropicMessagesStreamIterator
from litellm.llms.anthropic.messages.native.interceptors import MessagesInterceptor
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


WIDE_NUMBERS_REPLY: Final = (
    b'{"id":"msg_01ParityWide","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929",'
    b'"content":[{"type":"tool_use","id":"toolu_01ParityWide","name":"get_ids","input":{"big":18446744073709551616,'
    b'"huge":12345678901234567890123,"negative":-12345678901234567890123,"neg_zero":-0.0,"overflow":1e400,'
    b'"underflow":-1E+400,"exp":1E2,"tiny":5e-324,"precise":0.30000000000000004,"text":"1e400 18446744073709551616"}}],'
    b'"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":9}}'
)
WIDE_INTEGER_REPLY: Final = (
    b'{"id":"msg_01ParityWideCached","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929",'
    b'"content":[{"type":"tool_use","id":"toolu_01ParityWideCached","name":"get_ids","input":{"huge":12345678901234567890123}}],'
    b'"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":9}}'
)

MARKED_TEXTS_REPLY: Final = (
    '{"id":"msg_01ParityMarked","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929",'
    '"content":[{"type":"text","text":"\\uf8fflitellm-raw-json:42"},{"type":"tool_use","id":"toolu_01ParityMarked",'
    '"name":"get_ids","input":{"n":"\uf8fflitellm-raw-json:NaN","s":"\\uF8FFlitellm-raw-json:\\"x\\"",'
    '"\\uf8fflitellm-raw-json:1":true,"big":18446744073709551616}}],'
    '"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":9}}'
).encode()


def _non_string_tool_result_texts(
    *texts: object,
) -> list[dict[str, object]]:  # mutable-ok: the messages kwarg is a list of dicts
    return [
        dict(USER),
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01Texts", "name": "get_ids", "input": {}}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01Texts",
                    "content": [{"type": "text", "text": text} for text in texts],
                },
                {"type": "text", "text": "go on"},
            ],
        },
    ]


_BAD_TRAILER_GZIP: Final = gzip.compress(json.dumps(dict(TEXT_RESPONSE)).encode(), mtime=0)[:-8] + bytes(8)
_NOT_GZIP: Final = b"\x1f\x8b\x08\x00not really gzip data at allnot really gzip data at all"


def _framed_bytes(events: tuple[tuple[str, object], ...]) -> tuple[bytes, ...]:
    return tuple(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in events)


def _encoded(body: bytes, encoding: str) -> ResponseSpec:
    return ResponseSpec(body=body, headers={"content-encoding": encoding})


def _upstream_error(kind: str, message: str) -> Mapping[str, JsonValue]:
    return MappingProxyType({"type": "error", "error": {"type": kind, "message": message}})


def _ok(body: Mapping[str, JsonValue] = TEXT_RESPONSE, **headers: str) -> ResponseSpec:
    return ResponseSpec(body=dict(body), headers=dict(headers))


def _stream(events: tuple[tuple[str, object], ...], chunk_delay: float = 0) -> ResponseSpec:
    return ResponseSpec(body={}, events=events, chunk_delay=chunk_delay)


def _burst(events: tuple[tuple[str, object], ...], chunked: bool = True) -> ResponseSpec:
    return ResponseSpec(body={}, events=events, burst=True, chunked=chunked)


def _chunked(events: tuple[tuple[str, object], ...], chunk_delay: float) -> ResponseSpec:
    return ResponseSpec(body={}, events=events, chunk_delay=chunk_delay, chunked=True)


_PING: Final[tuple[str, JsonValue]] = ("ping", {"type": "ping"})

PINGED_STREAM: Final = _events(
    _message_start("msg_01ParityPinged"),
    *((_PING,) * 50),
    *_text_block(0, "Pa", "ris"),
    *_message_end("end_turn", 4),
)

HEAVILY_PINGED_STREAM: Final = _events(
    _message_start("msg_01ParityHeavilyPinged"),
    *((_PING,) * 1000),
    *_text_block(0, "Pa", "ris"),
    *_message_end("end_turn", 4),
)

LARGE_STREAM: Final = _events(
    _message_start("msg_01ParityLarge"),
    *_text_block(0, "a" * 15000, "b" * 15000, "c" * 15000),
    *_message_end("end_turn", 4),
)


def _raw_stream(*frames: str) -> ResponseSpec:
    return ResponseSpec(body="".join(f"{frame}\n\n" for frame in frames).encode())


def _framed(events: tuple[tuple[str, object], ...], event_line: str = "event: ") -> tuple[str, ...]:
    return tuple(f"{event_line}{event}\ndata: {json.dumps(data)}" for event, data in events)


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
    "logged_model_parameters",
    "logged_error",
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
    interceptor: Callable[[], MessagesInterceptor] | None = None
    gaps: Mapping[str, str | None] = field(default_factory=lambda: MappingProxyType({}))
    loggers: Callable[[], Sequence[CustomLogger]] | None = None
    gateway: bool = False
    racy: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    partitioned: bool = False
    nones: tuple[str, ...] = ()
    busy_before_reading: float = 0
    warned: bool = False

    def aspects(self) -> tuple[str, ...]:
        return (
            *ASPECTS,
            *(("cached",) if self.cache is not None else ()),
            *(("warnings",) if self.warned else ()),
            *(("hook_inputs",) if self.loggers else ()),
            *(("classified",) if self.classifier is not None else ()),
            *(("tool_runs",) if self.gateway else ()),
        )

    def params(self) -> Iterator[object]:
        for aspect in self.aspects():
            gap = {**KNOWN_GAPS.get(self.name, {}), **self.gaps}.get(aspect)
            race = self.racy.get(aspect)
            marks = (
                (pytest.mark.skip(reason=race),)
                if race
                else (() if not gap else (pytest.mark.xfail(reason=gap, strict=True),))
            )
            yield pytest.param(self, aspect, id=f"{self.name}-{aspect}", marks=marks)


def _gaps(**gaps: str | None) -> Mapping[str, str | None]:
    return MappingProxyType(gaps)


def _kw(**kwargs: object) -> Mapping[str, object]:
    return MappingProxyType(kwargs)


def _tls(**settings: object) -> Mapping[str, object]:
    return MappingProxyType({**settings, "in_memory_llm_clients_cache": LLMClientCache()})


E04_HTTPX_CHUNK_READS: Final = "E04: httpx yields each HTTP chunk whole from a 64 KiB socket read; hyper hands a large chunk over in pieces of its smaller read buffer"
A10_NO_MAX_TOKENS: Final = "A10: native admission hands a call without an int max_tokens to Python, whose signature raises the TypeError; only dispatch differs"
C_UNSENT_CALL_FLAG: Final = (
    "C: a call that fails natively before its provider request logs no aanthropic_messages flag, "
    "which Python's handler sets before it shapes the request"
)

F46_SEE_OTHER_CONTENT_TYPE: Final = (
    "F46: httpx keeps content-type on the GET that follows a 303; reqwest's redirect follower drops it with the body"
)
REDIRECTED_SENT_REQUEST: Final = (
    "a followed provider redirect: Python logs the redirected request (its URL, its body already streamed) on the "
    "logged response, the native call logs the request it sent"
)

KNOWN_GAPS: Final[Mapping[str, Mapping[str, str | None]]] = MappingProxyType(
    {
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
                "max_tokens_none",
                "reasoning_effort_unknown",
                "output_config_effort_unknown",
                "reasoning_effort_unknown_on_adaptive_model",
                "reasoning_effort_xhigh_on_model_without_xhigh",
                "int_temperature_on_adaptive_model",
                "int_top_p_on_adaptive_model",
                "int_speed_on_model_without_speed",
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
        warned=True,
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
    Case(
        "response_gzip_under_a_padded_mixed_case_alias",
        upstream=(_encoded(gzip.compress(json.dumps(dict(TEXT_RESPONSE)).encode(), mtime=0), " X-Gzip "),),
    ),
    Case("response_deflate_body_that_is_not_deflate", upstream=(_encoded(b"not deflate at all", "deflate"),)),
    Case("response_gzip_body_with_a_bad_trailer", upstream=(_encoded(_BAD_TRAILER_GZIP, "gzip"),)),
    Case("response_gzip_body_that_is_not_gzip", upstream=(_encoded(_NOT_GZIP, "gzip"),)),
    Case(
        "stream_deflate_body_that_is_not_deflate",
        _kw(stream=True),
        upstream=(_encoded(b"not deflate at all", "deflate"),),
    ),
    Case(
        "stream_gzip_listed_with_an_unknown_coding",
        _kw(stream=True),
        upstream=(_encoded(gzip.compress(b"".join(_framed_bytes(TEXT_STREAM)), mtime=0), "gzip, br"),),
    ),
)

INPUT_VALUE_CASES: Final = (
    Case("max_tokens_numeric_string", _kw(max_tokens="1024")),
    Case("max_tokens_fraction", _kw(max_tokens=1.5)),
    Case("max_tokens_whole_float", _kw(max_tokens=1024.0)),
    Case("max_tokens_bool", _kw(max_tokens=True)),
    Case("max_tokens_beyond_64_bits", _kw(max_tokens=2**64)),
    Case("reasoning_effort_with_fractional_max_tokens", _kw(reasoning_effort="high", max_tokens=1500.5)),
    Case("temperature_numeric_string", _kw(temperature="0.5")),
    Case("temperature_integer_beyond_float_precision", _kw(temperature=2**60 + 1)),
    Case("top_p_integer_beyond_float_precision", _kw(top_p=-9223372036854775807)),
    Case(
        "reasoning_effort_with_numeric_string_max_tokens",
        _kw(reasoning_effort="high", max_tokens="8192"),
        upstream=(),
    ),
    Case("top_k_beyond_64_bits", _kw(top_k=2**70)),
    Case("messages_mapping", _kw(messages={"role": "user", "content": "hi"})),
    Case("messages_string", _kw(messages="hi")),
    Case("messages_with_non_object_entries", _kw(messages=[dict(USER), "stray", 7])),
    Case("system_lone_surrogate", _kw(system="sys \ud800 lone")),
    Case("metadata_user_id_lone_surrogate", _kw(metadata={"user_id": "user-\udc00"})),
    Case("message_text_lone_surrogate", _kw(messages=[{"role": "user", "content": "a\ud800b \u00e9"}])),
    Case("text_starting_with_the_raw_leaf_mark", _kw(system="\uf8fflitellm-raw-json:1")),
    Case("thinking_budget_beyond_64_bits", _kw(thinking={"type": "enabled", "budget_tokens": 2**70}, max_tokens=2048)),
    Case(
        "tool_schema_integers_beyond_64_bits",
        _kw(tools=[{**WEATHER_TOOL, "input_schema": {"type": "integer", "maximum": 2**70, "minimum": -(2**70)}}]),
    ),
    Case("temperature_nan", _kw(temperature=float("nan"))),
    Case(
        "tool_schema_infinities",
        _kw(
            tools=[
                {**WEATHER_TOOL, "input_schema": {"type": "number", "maximum": float("inf"), "minimum": -float("inf")}}
            ]
        ),
    ),
)

LOOSE_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"name": "lookup", "description": "lookup", "input_schema": {"type": "object", "properties": {}}}
)


class _StringStreamHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        kwargs["stream"] = "true"
        return kwargs


class _IntegerStreamHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        kwargs["stream"] = 1
        return kwargs


class _StringParamsHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        kwargs["litellm_params"] = "not-a-mapping"
        return kwargs


class _IntegerProviderHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        params: Final = kwargs.setdefault("litellm_params", {})
        assert isinstance(params, dict)
        params["custom_llm_provider"] = 5
        return kwargs


ARGUMENT_TYPE_CASES: Final = (
    Case("stream_string_true", _kw(stream="true"), upstream=(_stream(TEXT_STREAM),)),
    Case("stream_string_false", _kw(stream="false"), upstream=(_stream(TEXT_STREAM),)),
    Case("stream_zero", _kw(stream=0)),
    Case("stream_one", _kw(stream=1), upstream=(_stream(TEXT_STREAM),)),
    Case("stream_one_sync", _kw(stream=1), upstream=(_stream(TEXT_STREAM),), sync=True),
    Case(
        "pre_request_hook_streams_on_a_string",
        _kw(stream=True),
        callbacks=(_StringStreamHook,),
        upstream=(_stream(TEXT_STREAM),),
    ),
    Case("pre_request_hook_streams_on_an_integer", callbacks=(_IntegerStreamHook,), upstream=(_stream(TEXT_STREAM),)),
    Case("pre_request_hook_sets_string_litellm_params", callbacks=(_StringParamsHook,), model="claude-sonnet-4-5"),
    Case("pre_request_hook_sets_integer_provider", callbacks=(_IntegerProviderHook,), model="claude-sonnet-4-5"),
    Case("model_info_string", _kw(model_info="not-a-mapping")),
    Case("tools_mapping", _kw(tools=dict(LOOSE_TOOL)), upstream=()),
    Case("messages_none", _kw(messages=None), nones=("messages",), upstream=()),
    Case("max_tokens_none", _kw(max_tokens=None), nones=("max_tokens",), upstream=()),
    Case("reasoning_effort_unknown", _kw(reasoning_effort="ultra"), upstream=()),
    Case("reasoning_effort_unknown_on_adaptive_model", _kw(reasoning_effort="ultra"), model=OPUS_ADAPTIVE, upstream=()),
    Case(
        "reasoning_effort_xhigh_on_model_without_xhigh",
        _kw(reasoning_effort="xhigh", max_tokens=8192),
        model="anthropic/claude-opus-4-6",
        upstream=(),
    ),
    Case("int_temperature_on_adaptive_model", _kw(temperature=0), model=OPUS_ADAPTIVE, upstream=()),
    Case("int_top_p_on_adaptive_model", _kw(top_p=1), model=OPUS_ADAPTIVE, upstream=()),
    Case("int_speed_on_model_without_speed", _kw(speed=2), upstream=()),
    Case("string_output_format_dropped", _kw(output_format="json", drop_params=True)),
    Case("string_output_format_kept_without_drop_params", _kw(output_format="json")),
    Case("non_string_reasoning_effort_warns", _kw(reasoning_effort=3), warned=True),
    Case(
        "reasoning_effort_that_cannot_fit_max_tokens_warns",
        _kw(reasoning_effort="high", max_tokens=500),
        warned=True,
    ),
    Case(
        "adaptive_thinking_that_cannot_fit_max_tokens_warns",
        _kw(thinking={"type": "adaptive"}, output_config={"effort": "high"}, max_tokens=1024),
        warned=True,
    ),
    Case(
        "adaptive_thinking_on_model_without_reasoning_warns",
        _kw(thinking={"type": "adaptive"}),
        model="anthropic/claude-3-haiku-20240307",
        warned=True,
    ),
    Case("output_config_effort_unknown", _kw(output_config={"effort": "ultra"})),
    Case(
        "global_drop_params_string_true",
        _kw(tools=[{**LOOSE_TOOL, "custom": {"note": "opaque"}}]),
        settings=_kw(drop_params="true"),
    ),
    Case(
        "global_drop_params_string_false",
        _kw(tools=[{**LOOSE_TOOL, "custom": {"note": "opaque"}}]),
        settings=_kw(drop_params="false"),
    ),
    Case(
        "request_drop_params_string_true",
        _kw(drop_params="true", tools=[{**LOOSE_TOOL, "custom": {"note": "opaque"}}]),
    ),
    Case(
        "additional_drop_params_tool_custom",
        _kw(
            additional_drop_params=["tools[*].custom"],
            tools=[{**LOOSE_TOOL, "custom": {"defer_loading": True, "note": "x"}}],
        ),
    ),
    Case(
        "additional_drop_params_tool_input_examples",
        _kw(
            additional_drop_params=["tools[*].input_examples", "metadata.user_id", "fz.nope"],
            metadata={"user_id": "u1"},
            tools=[{**LOOSE_TOOL, "input_examples": [{"q": "x"}]}],
        ),
    ),
    Case(
        "response_without_stop_reason",
        upstream=(_ok({key: value for key, value in TEXT_RESPONSE.items() if key != "stop_reason"}),),
    ),
    Case(
        "response_without_stop_fields",
        upstream=(
            _ok({key: value for key, value in TEXT_RESPONSE.items() if key not in ("stop_reason", "stop_sequence")}),
        ),
    ),
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
    ),
    Case(
        "response_body_stalls_past_timeout",
        _kw(timeout=0.5),
        upstream=(ResponseSpec(body=dict(TEXT_RESPONSE), chunk_delay=1.0, body_chunks=2),),
    ),
    Case(
        "upstream_500_body_cut",
        upstream=(
            ResponseSpec(body=dict(_upstream_error("api_error", "cut")), status=500, body_chunks=2, cut_after=1),
        ),
    ),
    Case("response_body_cut", upstream=(ResponseSpec(body=dict(TEXT_RESPONSE), body_chunks=2, cut_after=1),)),
    Case("gzip_response", upstream=(ResponseSpec(body=dict(TEXT_RESPONSE), gzip=True),)),
    Case("upstream_500_gzip", upstream=(replace(_failure(500, "api_error", "internal"), gzip=True),)),
    Case("missing_max_tokens", _kw(max_tokens=None), upstream=()),
    Case("missing_api_key", _kw(api_key=None), upstream=()),
    Case(
        "api_key_with_a_trailing_newline",
        _kw(api_key=None),
        env=MappingProxyType({"ANTHROPIC_API_KEY": f"{API_KEY}\n"}),
    ),
    Case(
        "api_key_with_a_trailing_crlf_streamed",
        _kw(api_key=None, stream=True),
        upstream=(_stream(TEXT_STREAM),),
        env=MappingProxyType({"ANTHROPIC_API_KEY": f"{API_KEY}\r\n"}),
    ),
    Case(
        "api_key_padded_with_whitespace",
        _kw(api_key=None),
        env=MappingProxyType({"ANTHROPIC_API_KEY": f"  {API_KEY}\t\n"}),
    ),
    Case(
        "api_key_padded_with_separators",
        _kw(api_key=None),
        env=MappingProxyType({"ANTHROPIC_API_KEY": f"\x1f{API_KEY}\x1f"}),
    ),
    Case(
        "api_key_of_separators_falls_to_the_auth_token",
        _kw(api_key=None),
        env=MappingProxyType({"ANTHROPIC_API_KEY": "\x1c\x1f\n", "ANTHROPIC_AUTH_TOKEN": "oauth-token"}),
    ),
    Case(
        "auth_token_with_a_trailing_newline",
        _kw(api_key=None),
        env=MappingProxyType({"ANTHROPIC_AUTH_TOKEN": "oauth-token\n"}),
    ),
    Case("caller_api_key_padded_with_separators", _kw(api_key=f"\x1f{API_KEY}\x1f")),
    Case("caller_api_key_with_a_trailing_newline", _kw(api_key=f"{API_KEY}\n")),
    Case(
        "auth_token_padded_with_whitespace",
        _kw(api_key=None),
        env=MappingProxyType({"ANTHROPIC_AUTH_TOKEN": "  fz-env-auth-token  "}),
    ),
    Case(
        "provider_redirect_is_followed",
        gaps=_gaps(logged_sent_request=REDIRECTED_SENT_REQUEST),
        upstream=(
            ResponseSpec(body=b"moved", status=307, headers={"location": "/redirected/v1/messages"}),
            ResponseSpec(body=dict(TEXT_RESPONSE)),
        ),
    ),
    Case(
        "provider_redirect_chain_of_eleven_hops",
        gaps=_gaps(logged_sent_request=REDIRECTED_SENT_REQUEST),
        upstream=(
            *(
                ResponseSpec(body=b"moved", status=307, headers={"location": f"/hop{hop}/v1/messages"})
                for hop in range(11)
            ),
            ResponseSpec(body=dict(TEXT_RESPONSE)),
        ),
    ),
    Case(
        "provider_redirect_chain_past_twenty_hops",
        upstream=tuple(
            ResponseSpec(body=b"moved", status=307, headers={"location": f"/hop{hop}/v1/messages"}) for hop in range(21)
        ),
    ),
    Case(
        "provider_see_other_redirect",
        gaps=_gaps(logged_sent_request=REDIRECTED_SENT_REQUEST, request=F46_SEE_OTHER_CONTENT_TYPE),
        upstream=(
            ResponseSpec(body=b"see other", status=303, headers={"location": "/other/v1/messages"}),
            ResponseSpec(body=dict(TEXT_RESPONSE)),
        ),
    ),
    Case("extra_header_lone_surrogate", _kw(extra_headers={"x-fz-surrogate": "h\ud800"}), upstream=()),
    Case(
        "provider_specific_header_lone_surrogate",
        _kw(
            provider_specific_header={
                "custom_llm_provider": "anthropic",
                "extra_headers": {"x-fz-surrogate": "h\ud800"},
            }
        ),
        upstream=(),
    ),
    Case("extra_header_non_ascii", _kw(extra_headers={"x-fz-unicode": "caf\u00e9"}), upstream=()),
    Case("caller_api_key_non_ascii", _kw(api_key="sk-ant-\u200djoined"), upstream=()),
    Case("caller_api_key_non_ascii_stream", _kw(api_key="sk-ant-\u200djoined", stream=True), upstream=()),
    Case("api_base_with_a_space_and_no_scheme", _kw(api_base="event: message_stop"), upstream=()),
    Case("api_base_with_a_space_and_no_scheme_stream", _kw(api_base="data: [DONE]", stream=True), upstream=()),
    Case("api_base_with_a_nul", _kw(api_base="a\x00b"), upstream=()),
    Case("api_base_non_ascii_without_a_scheme", _kw(api_base="h\u00e9llo"), upstream=()),
    Case("api_base_with_an_unclosed_ipv6_bracket", _kw(api_base_suffix="data: [DONE]"), upstream=()),
    Case("provider_specific_header_as_a_string", _kw(provider_specific_header="anthropic"), upstream=()),
    Case("response_header_latin_1", upstream=(_ok(**{"x-fz-unicode": "caf\u00e9"}),)),
    Case("response_header_utf_8", upstream=(_ok(**{"x-fz-unicode": "caf\u00c3\u00a9"}),)),
    Case(
        "response_header_utf_8_httpx_transport",
        upstream=(_ok(**{"x-fz-unicode": "caf\u00c3\u00a9"}),),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
    ),
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
    Case("stream_events_in_one_write", _kw(stream=True), upstream=(_burst(TEXT_STREAM),), partitioned=True),
    Case(
        "stream_body_in_one_write",
        _kw(stream=True),
        upstream=(_burst(TEXT_STREAM, chunked=False),),
        partitioned=True,
    ),
    Case(
        "stream_events_in_chunks_written_apart",
        _kw(stream=True),
        upstream=(_chunked(TEXT_STREAM, chunk_delay=0.05),),
        partitioned=True,
    ),
    Case(
        "stream_events_written_apart",
        _kw(stream=True),
        upstream=(_stream(TEXT_STREAM, chunk_delay=0.05),),
        partitioned=True,
    ),
    Case(
        "stream_pings_after_message_start_in_one_write",
        _kw(stream=True),
        upstream=(_burst(PINGED_STREAM),),
        partitioned=True,
    ),
    Case(
        "stream_pings_after_message_start_written_apart",
        _kw(stream=True),
        upstream=(_stream(PINGED_STREAM, chunk_delay=0.01),),
        partitioned=True,
    ),
    Case("stream_burst_past_one_read", _kw(stream=True), upstream=(_burst(LARGE_STREAM),), partitioned=True),
    Case(
        "stream_thousand_pings_in_one_write",
        _kw(stream=True),
        upstream=(_burst(HEAVILY_PINGED_STREAM),),
        partitioned=True,
    ),
    Case(
        "stream_error_event_in_one_write",
        _kw(stream=True),
        upstream=(_burst(ERROR_MID_STREAM),),
        partitioned=True,
    ),
    Case(
        "stream_events_in_one_write_httpx_transport",
        _kw(stream=True),
        upstream=(_burst(TEXT_STREAM),),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
        partitioned=True,
    ),
    Case(
        "stream_burst_past_one_read_httpx_transport",
        _kw(stream=True),
        upstream=(_burst(LARGE_STREAM),),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
        partitioned=True,
        gaps=_gaps(outcome=E04_HTTPX_CHUNK_READS),
    ),
    Case(
        "stream_events_written_apart_httpx_transport",
        _kw(stream=True),
        upstream=(_stream(TEXT_STREAM, chunk_delay=0.05),),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
        partitioned=True,
    ),
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
    Case(
        "stream_upstream_500_body_cut",
        _kw(stream=True),
        upstream=(
            ResponseSpec(body=dict(_upstream_error("api_error", "cut")), status=500, body_chunks=2, cut_after=1),
        ),
    ),
    Case(
        "stream_cut_after_four_events",
        _kw(stream=True),
        upstream=(replace(_stream(TEXT_STREAM), cut_after=4, cut_delay=0.1),),
    ),
    Case(
        "stream_cut_after_message_start",
        _kw(stream=True),
        upstream=(replace(_stream(TEXT_STREAM), cut_after=1, cut_delay=0.1),),
    ),
    Case(
        "stream_cut_after_events_written_apart",
        _kw(stream=True),
        upstream=(replace(_stream(TEXT_STREAM, chunk_delay=0.05), cut_after=6, cut_delay=0.1),),
    ),
    Case(
        "stream_chunked_cut_after_four_events",
        _kw(stream=True),
        upstream=(replace(_chunked(TEXT_STREAM, chunk_delay=0), cut_after=4, cut_delay=0.1),),
    ),
    Case(
        "stream_cut_while_the_caller_is_busy",
        _kw(stream=True),
        upstream=(replace(_chunked(TEXT_STREAM, chunk_delay=0), cut_after=4, cut_delay=0.1),),
        busy_before_reading=0.4,
    ),
    Case(
        "stream_cut_after_message_start_while_the_caller_is_busy",
        _kw(stream=True),
        upstream=(replace(_stream(TEXT_STREAM), cut_after=1, cut_delay=0.1),),
        busy_before_reading=0.4,
    ),
    Case(
        "stream_upstream_500_body_stalls",
        _kw(stream=True, timeout=8),
        upstream=(
            ResponseSpec(body=dict(_upstream_error("api_error", "stalled")), status=500, chunk_delay=6, body_chunks=2),
        ),
    ),
    Case(
        "stream_upstream_429_body_stalls_past_timeout",
        _kw(stream=True, timeout=1),
        upstream=(
            ResponseSpec(
                body=dict(_upstream_error("rate_limit_error", "stalled")), status=429, chunk_delay=2, body_chunks=2
            ),
        ),
    ),
    Case("stream_gzip_response", _kw(stream=True), upstream=(replace(_stream(TEXT_STREAM), gzip=True),)),
    Case(
        "stream_read_stalls_past_timeout_httpx_transport",
        _kw(stream=True, timeout=0.5),
        upstream=(_stream(SLOW_STREAM, chunk_delay=1.0),),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
    ),
    Case(
        "stream_response_header_utf_8",
        _kw(stream=True),
        upstream=(replace(_stream(TEXT_STREAM), headers={"x-fz-unicode": "caf\u00c3\u00a9"}),),
    ),
    Case(
        "stream_response_header_utf_8_httpx_transport",
        _kw(stream=True),
        upstream=(replace(_stream(TEXT_STREAM), headers={"x-fz-unicode": "caf\u00c3\u00a9"}),),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
    ),
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
    Case("reply_with_wide_numbers", upstream=(ResponseSpec(body=WIDE_NUMBERS_REPLY),)),
    Case("reply_with_a_wide_integer_cached", upstream=(ResponseSpec(body=WIDE_INTEGER_REPLY),), cache=Cache),
    Case("reply_with_raw_leaf_marked_texts", upstream=(ResponseSpec(body=MARKED_TEXTS_REPLY),)),
    Case("reply_with_raw_leaf_marked_texts_cached", upstream=(ResponseSpec(body=MARKED_TEXTS_REPLY),), cache=Cache),
    Case(
        "modify_params_clamped_stream_hits_the_cache",
        _kw(max_tokens=200_000, stream=True),
        upstream=(_stream(TEXT_STREAM), _stream(TEXT_STREAM)),
        settings=_MODIFY_PARAMS,
        cache=Cache,
    ),
    Case(
        "drop_params_drops_non_string_tool_result_texts",
        _kw(
            messages=_non_string_tool_result_texts(5, None, {"a": 1}, 8.5, 2**70, float("nan"), "kept"),
            drop_params=True,
        ),
    ),
    Case(
        "global_drop_params_drops_every_tool_result_text",
        _kw(messages=_non_string_tool_result_texts(5, None)),
        settings=MappingProxyType({"drop_params": True}),
    ),
    Case(
        "drop_params_keeps_string_tool_result_texts",
        _kw(messages=_non_string_tool_result_texts("\uf8fflitellm-raw-json:1", "kept"), drop_params=True),
    ),
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
    *(
        Case(name, _kw(stream=True), upstream=(_raw_stream(*frames), _raw_stream(*frames)), cache=Cache)
        for name, frames in (
            ("cache_stream_data_only_frames", tuple(f"data: {json.dumps(data)}" for _, data in TEXT_STREAM)),
            ("cache_stream_event_without_space", _framed(TEXT_STREAM, "event:")),
            ("cache_stream_done_sentinel", (*_framed(TEXT_STREAM), "data: [DONE]")),
            (
                "cache_stream_without_output",
                _framed(
                    _events(_message_start("msg_01ParityEmpty"), *_text_block(0, ""), *_message_end("end_turn", 1))
                ),
            ),
            (
                "cache_stream_error_after_stop",
                (
                    *_framed(TEXT_STREAM),
                    *_framed((("error", dict(_upstream_error("overloaded_error", "Overloaded"))),)),
                ),
            ),
        )
    ),
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
    Case("tls_ecdh_curve_x448", settings=_tls(ssl_ecdh_curve="X448")),
    Case("tls_ecdh_curve_upstream_500", upstream=(_failure(500, "api_error", "internal"),), settings=_TLS_CURVE),
    Case(
        "tls_security_level_from_environment",
        env=MappingProxyType({"SSL_SECURITY_LEVEL": "DEFAULT@SECLEVEL=1"}),
        settings=_tls(),
    ),
    Case("tls_security_level_python_cannot_apply", upstream=(), settings=_tls(ssl_security_level="NOT-A-CIPHER")),
    Case("tls_ecdh_curve_nist_name", settings=_tls(ssl_ecdh_curve="P-256")),
    Case("tls_ecdh_curve_padded_lowercase", settings=_tls(ssl_ecdh_curve=" x25519 ")),
    Case("negative_timeout", _kw(timeout=-1), upstream=(ResponseSpec(body=dict(TEXT_RESPONSE), delay=0.3),)),
    Case(
        "negative_timeout_stream",
        _kw(stream=True, timeout=-0.5),
        upstream=(replace(_stream(TEXT_STREAM), delay=0.3),),
    ),
    Case(
        "zero_timeout_httpx_transport",
        _kw(timeout=0),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
    ),
    Case(
        "negative_timeout_httpx_transport",
        _kw(timeout=-1),
        env=MappingProxyType({"DISABLE_AIOHTTP_TRANSPORT": "true"}),
        settings=_tls(),
    ),
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


class _DeploymentHooks(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes untyped kwargs
        self.seen: list[Mapping[str, object]] = []

    async def async_post_call_success_deployment_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> object:
        self.seen.append({"hook": "success", "call_type": str(call_type)})
        return None

    async def async_post_call_failure_deployment_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self, request_data: Mapping[str, object], exception: Exception, call_type: object
    ) -> None:
        self.seen.append({"hook": "failure", "error": type(exception).__name__, "call_type": str(call_type)})


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
    *(
        Case(
            f"agentic_loop_follow_up_keeps_{name}",
            _kw(**{"tools": [dict(WEATHER_TOOL)], **params}),
            upstream=(_ok(TOOL_USE_RESPONSE), _ok()),
            model=model,
            loggers=_loggers(_FollowUpOnToolUse),
        )
        for name, params, model in (
            (
                "output_format",
                {
                    "output_format": {
                        "type": "json_schema",
                        "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                    }
                },
                SONNET,
            ),
            ("reasoning_effort", {"reasoning_effort": "high", "max_tokens": 8192}, SONNET),
            ("output_config_effort", {"output_config": {"effort": "medium"}}, OPUS_ADAPTIVE),
            (
                "injected_cache_points",
                {"cache_control_injection_points": [{"location": "message", "role": "user"}]},
                SONNET,
            ),
            ("context_management", {"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}, SONNET),
            ("speed", {"speed": "fast"}, OPUS_FAST),
            ("service_tier", {"service_tier": "auto"}, SONNET),
            ("inference_geo", {"inference_geo": "us"}, SONNET),
            (
                "container",
                {"container": {"skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]}},
                SONNET,
            ),
            ("custom_tool", {"tools": [{**WEATHER_TOOL, "type": "custom"}]}, SONNET),
            ("metadata_extras", {"metadata": {"user_id": "user-123", "session": "dropped"}}, SONNET),
        )
    ),
    Case(
        "agentic_loop_prices_the_final_round",
        _kw(tools=[dict(WEATHER_TOOL)]),
        upstream=(_ok(TOOL_USE_RESPONSE), _ok({**TEXT_RESPONSE, "usage": {"input_tokens": 900, "output_tokens": 300}})),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_prices_the_final_round_stream",
        _kw(stream=True, tools=[dict(WEATHER_TOOL)]),
        upstream=(
            _stream(TOOL_STREAM),
            _stream(
                _events(_message_start("msg_01ParityLate"), *_text_block(0, "Paris"), *_message_end("end_turn", 300))
            ),
        ),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_follow_up_stream",
        _kw(stream=True, tools=[dict(WEATHER_TOOL)]),
        upstream=(_stream(TOOL_STREAM), _stream(TEXT_STREAM)),
        loggers=_loggers(_FollowUpOnToolUse),
    ),
    Case(
        "agentic_loop_follow_up_stream_in_one_write",
        _kw(stream=True, tools=[dict(WEATHER_TOOL)]),
        upstream=(_burst(TOOL_STREAM), _burst(TEXT_STREAM)),
        loggers=_loggers(_FollowUpOnToolUse),
        partitioned=True,
    ),
    Case(
        "agentic_loop_stream_without_tool_use_in_one_write",
        _kw(stream=True, tools=[dict(WEATHER_TOOL)]),
        upstream=(_burst(TEXT_STREAM),),
        loggers=_loggers(_FollowUpOnToolUse),
        partitioned=True,
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
        "agentic_loop_plan_fails_runs_the_failure_deployment_hook_once",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE),),
        loggers=_loggers(_Fails, _DeploymentHooks),
    ),
    Case(
        "agentic_loop_follow_up_fails_runs_the_failure_deployment_hook_per_round",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE), _failure(529, "overloaded_error", "Overloaded")),
        loggers=_loggers(_FollowUpOnToolUse, _DeploymentHooks),
    ),
    Case(
        "agentic_loop_follow_up_runs_the_success_deployment_hook_per_round",
        _WEATHER,
        upstream=(_ok(TOOL_USE_RESPONSE), _ok()),
        loggers=_loggers(_FollowUpOnToolUse, _DeploymentHooks),
    ),
    Case(
        "pre_request_hook_raises_runs_the_failure_deployment_hook_once",
        callbacks=(_RaisingHook,),
        upstream=(),
        loggers=_loggers(_DeploymentHooks),
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

INTERCEPTED_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": "parity_intercept", "name": "intercept"})
_ECHOED_KWARGS: Final = (
    "model",
    "messages",
    "tools",
    "stream",
    "max_tokens",
    "custom_llm_provider",
    "api_key",
    "metadata",
    "system",
    "temperature",
    "tool_choice",
)


class _Intercepting(MessagesInterceptor):
    def can_handle(self, tools: Sequence[Mapping[str, object]] | None, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == "anthropic" and any(
            tool.get("type") == INTERCEPTED_TOOL["type"] for tool in tools or ()
        )


class _Answering(_Intercepting):
    async def handle(self, **kwargs: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: the interceptor contract's keyword call
        echoed: Final = {name: kwargs.get(name) for name in _ECHOED_KWARGS}
        answer: Final = {
            **TEXT_RESPONSE,
            "id": "msg_01ParityIntercepted",
            "content": [{"type": "text", "text": json.dumps(echoed, sort_keys=True, default=repr)}],
            "extra_keys": sorted(name for name in kwargs if name not in _ECHOED_KWARGS),
        }
        if kwargs.get("stream"):
            return FakeAnthropicMessagesStreamIterator(
                cast(AnthropicMessagesResponse, answer)
            )  # cast-ok: a synthetic Messages response
        return answer


class _Delegating(_Intercepting):
    async def handle(self, **kwargs: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: the interceptor contract's keyword call
        from litellm.messages import anthropic_messages

        tools: Final = cast(list[Mapping[str, object]], kwargs["tools"])  # cast-ok: the caller's tools
        return await anthropic_messages(
            model=kwargs["model"],
            messages=kwargs["messages"],
            max_tokens=kwargs["max_tokens"],
            tools=[tool for tool in tools if tool.get("type") != INTERCEPTED_TOOL["type"]],
            api_key=kwargs["api_key"],
            api_base=kwargs["api_base"],
            stream=kwargs["stream"],
        )


class _Refusing(_Intercepting):
    async def handle(self, **kwargs: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: the interceptor contract's keyword call
        raise ValueError("the interceptor refused the call")


def _intercepted(**kwargs: object) -> Mapping[str, object]:
    return _kw(tools=[dict(INTERCEPTED_TOOL), dict(WEATHER_TOOL)], **kwargs)


class _InterceptedToolHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        return {**kwargs, "tools": [dict(INTERCEPTED_TOOL), *cast(list[object], kwargs.get("tools") or [])]}


class _UninterceptedToolHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        return {**kwargs, "tools": [dict(WEATHER_TOOL)]}


class _DeploymentHookRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes untyped kwargs
        self.seen: list[Mapping[str, object]] = []

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self.seen.append({"logging_obj": type(kwargs.get("litellm_logging_obj")).__name__, "call_type": str(call_type)})


class _DeploymentResponseRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes untyped kwargs
        self.seen: list[Mapping[str, object]] = []

    async def async_post_call_success_deployment_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> object:
        self.seen.append({"response": type(response).__name__, "call_type": str(call_type)})
        return None


DEPLOYMENT_HOOK_CASES: Final = (
    Case(
        "deployment_hook_sees_the_stream_of_a_string_flag",
        _kw(stream="true"),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_DeploymentResponseRecorder),
    ),
    Case(
        "deployment_hook_sees_the_stream_of_an_integer_flag",
        _kw(stream=1),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_DeploymentResponseRecorder),
    ),
    Case(
        "deployment_hook_sees_the_stream_of_a_bool_flag",
        _kw(stream=True),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_DeploymentResponseRecorder),
    ),
    Case(
        "deployment_hook_sees_a_stream_a_hook_made",
        upstream=(_stream(TEXT_STREAM),),
        callbacks=(_UnstreamingHook,),
        loggers=_loggers(_DeploymentResponseRecorder),
    ),
    Case(
        "deployment_hook_sees_a_stream_a_hook_made_on_an_integer",
        upstream=(_stream(TEXT_STREAM),),
        callbacks=(_IntegerStreamHook,),
        loggers=_loggers(_DeploymentResponseRecorder),
    ),
    Case("deployment_hook_sees_the_logging_object", loggers=_loggers(_DeploymentHookRecorder)),
    Case(
        "deployment_hook_sees_the_logging_object_stream",
        _kw(stream=True),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_DeploymentHookRecorder),
    ),
)

INTERCEPTOR_CASES: Final = (
    Case("interceptor_answers", _intercepted(), upstream=(), interceptor=_Answering),
    Case(
        "pre_request_hook_adds_a_tool_an_interceptor_owns",
        _kw(tools=[dict(WEATHER_TOOL)]),
        upstream=(),
        callbacks=(_InterceptedToolHook,),
        interceptor=_Answering,
    ),
    Case(
        "pre_request_hook_adds_a_tool_an_interceptor_owns_stream",
        _kw(tools=[dict(WEATHER_TOOL)], stream=True),
        upstream=(),
        callbacks=(_InterceptedToolHook,),
        interceptor=_Answering,
    ),
    Case(
        "pre_request_hook_unstreams_a_call_an_interceptor_owns",
        _intercepted(stream=True),
        upstream=(),
        callbacks=(_UnstreamingHook,),
        interceptor=_Answering,
    ),
    Case(
        "pre_request_hook_removes_the_tool_an_interceptor_owns",
        _intercepted(),
        callbacks=(_UninterceptedToolHook,),
        interceptor=_Answering,
    ),
    Case(
        "interceptor_answers_with_system_and_metadata",
        _intercepted(system="Be brief.", metadata={"user_id": "user-123"}, temperature=0.5),
        upstream=(),
        interceptor=_Answering,
    ),
    Case("interceptor_answers_stream", _intercepted(stream=True), upstream=(), interceptor=_Answering),
    Case(
        "interceptor_wins_over_mock_response",
        _intercepted(mock_response="Hi from the mock"),
        upstream=(),
        interceptor=_Answering,
    ),
    Case(
        "interceptor_answers_bare_model",
        _intercepted(custom_llm_provider="anthropic"),
        upstream=(),
        model="claude-sonnet-4-5",
        interceptor=_Answering,
    ),
    Case("interceptor_delegates_to_the_public_api", _intercepted(), interceptor=_Delegating),
    Case(
        "interceptor_delegates_a_stream",
        _intercepted(stream=True),
        upstream=(_stream(TEXT_STREAM),),
        interceptor=_Delegating,
    ),
    Case("interceptor_raises", _intercepted(), upstream=(), interceptor=_Refusing),
    Case("interceptor_is_not_consulted_on_a_sync_call", _intercepted(), upstream=(), sync=True, interceptor=_Answering),
    Case(
        "pre_request_hook_is_not_run_on_a_sync_call_an_interceptor_owns",
        _intercepted(),
        upstream=(),
        sync=True,
        callbacks=(_RaisingHook,),
        interceptor=_Answering,
    ),
    Case("pre_request_hook_is_not_run_on_a_sync_call", upstream=(), sync=True, callbacks=(_RaisingHook,)),
    Case(
        "interceptor_raises_runs_the_failure_deployment_hook_once",
        _intercepted(),
        upstream=(),
        interceptor=_Refusing,
        loggers=_loggers(_DeploymentHooks),
    ),
    Case("interceptor_answer_is_cached", _intercepted(), upstream=(), cache=Cache, interceptor=_Answering),
    Case(
        "interceptor_stream_answer_with_cache",
        _intercepted(stream=True),
        upstream=(),
        cache=Cache,
        interceptor=_Answering,
    ),
    Case("interceptor_ignores_other_calls", _kw(tools=[dict(WEATHER_TOOL)]), interceptor=_Answering),
)

SHORT_CIRCUITED: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "id": "msg_01ParityShortCircuit",
        "content": [{"type": "text", "text": "Paris is the capital."}],
    }
)


class _ShortCircuit(WebSearchInterceptionLogger):
    """A web search interception whose replaced short-circuit records its inputs and answers every call."""

    answer: Mapping[str, JsonValue] | None = SHORT_CIRCUITED

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # the logger annotates bare containers
        self.seen: list[Mapping[str, object]] = []

    async def try_short_circuit_search(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        custom_llm_provider: str | None,
        kwargs: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        passed: Final = kwargs or {}
        self.seen.append(
            {
                "model": model,
                "messages": _OBJECTS.validate_python(messages),
                "tools": tools,
                "custom_llm_provider": custom_llm_provider,
                "kwargs": sorted(name for name in passed if name not in _VOLATILE_HOOK_KWARGS),
                "metadata": passed.get("metadata"),
                "safeguards": passed.get("safeguards"),
            }
        )
        return None if self.answer is None else _OBJECT.validate_python(copy.deepcopy(dict(self.answer)))


class _ShortCircuitPassing(_ShortCircuit):
    answer = None


class _ShortCircuitToolUse(_ShortCircuit):
    answer = TOOL_USE_RESPONSE


class _ShortCircuitRefusing(_ShortCircuit):
    async def try_short_circuit_search(  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        self,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        custom_llm_provider: str | None,
        kwargs: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        await super().try_short_circuit_search(model, messages, tools, custom_llm_provider, kwargs)
        raise ValueError("the short-circuit search failed")


def _searched(**kwargs: object) -> Mapping[str, object]:
    return _kw(tools=[dict(WEB_SEARCH_TOOL)], metadata={"user_id": "parity-user"}, **kwargs)


SHORT_CIRCUIT_CASES: Final = (
    Case("short_circuit_answers", _searched(), upstream=(), loggers=_loggers(_ShortCircuit)),
    Case("short_circuit_answers_stream", _searched(stream=True), upstream=(), loggers=_loggers(_ShortCircuit)),
    Case("short_circuit_raises", _searched(), upstream=(), loggers=_loggers(_ShortCircuitRefusing)),
    Case("short_circuit_raises_stream", _searched(stream=True), upstream=(), loggers=_loggers(_ShortCircuitRefusing)),
    Case("short_circuit_passes_the_call_on", _searched(), loggers=_loggers(_ShortCircuitPassing)),
    Case(
        "short_circuit_passes_a_stream_on",
        _searched(stream=True),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_ShortCircuitPassing),
    ),
    Case(
        "short_circuit_first_answer_wins",
        _searched(),
        upstream=(),
        loggers=_loggers(_ShortCircuitPassing, _ShortCircuit, _ShortCircuitRefusing),
    ),
    Case(
        "short_circuit_answers_before_the_interceptor",
        _intercepted(),
        upstream=(),
        loggers=_loggers(_ShortCircuit),
        interceptor=_Answering,
    ),
    Case(
        "short_circuit_passes_on_to_the_interceptor",
        _intercepted(),
        upstream=(),
        loggers=_loggers(_ShortCircuitPassing),
        interceptor=_Answering,
    ),
    Case(
        "short_circuit_answer_with_cache",
        _searched(),
        upstream=(),
        cache=Cache,
        loggers=_loggers(_ShortCircuit),
    ),
    Case(
        "short_circuit_answer_is_classified",
        _safeguarded(),
        upstream=(),
        model=NON_CLAUDE,
        classifier=FLAGGED,
        loggers=_loggers(_ShortCircuitToolUse),
    ),
    Case(
        "short_circuit_stream_answer_is_classified",
        _safeguarded(stream=True),
        upstream=(),
        model=NON_CLAUDE,
        classifier=FLAGGED,
        loggers=_loggers(_ShortCircuitToolUse),
    ),
    Case("short_circuit_is_not_consulted_on_a_sync_call", _searched(), sync=True, loggers=_loggers(_ShortCircuit)),
)

MCP_REFERENCE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "mcp",
        "server_label": "litellm",
        "server_url": "litellm_proxy/mcp/deepwiki",
        "require_approval": "never",
    }
)
MCP_NEEDS_APPROVAL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {key: value for key, value in MCP_REFERENCE.items() if key != "require_approval"}
)
WIKI_TOOL_CALL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "tool_use", "id": "toolu_01Wiki", "name": "read_wiki", "input": {"repo": "a/b"}}
)
WIKI_TOOL_USE_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "id": "msg_01ParityWiki",
        "content": [{"type": "text", "text": "Reading the wiki."}, dict(WIKI_TOOL_CALL)],
        "stop_reason": "tool_use",
    }
)
WIKI_TOOL_STREAM: Final = _events(
    _message_start("msg_01ParityWikiStream"),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_01Wiki", "name": "read_wiki", "input": {}},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"repo": "a/b"}'},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    *_message_end("tool_use", 9),
)


def _wiki(**kwargs: object) -> Mapping[str, object]:
    return _kw(**{"tools": [dict(MCP_REFERENCE)], **kwargs})


class _McpToolHook(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        return {**kwargs, "tools": [dict(MCP_REFERENCE), *cast(list[object], kwargs.get("tools") or [])]}


_SHARED_LOGGER: Final = (
    "the gateway's follow-up call rewrites the shared Logging object while the first call's async success callback "
    "is still queued, on Python and Rust alike, so the callback logs whichever call it saw"
)
SHARED_LOGGER_RACE: Final = MappingProxyType(
    {
        "logged_request": _SHARED_LOGGER,
        "logged_metadata": _SHARED_LOGGER,
        "logged_response_headers": _SHARED_LOGGER,
        "logged_sent_request": _SHARED_LOGGER,
    }
)


MCP_CASES: Final = (
    Case(
        "mcp_tool_runs_then_answers",
        _wiki(),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        gateway=True,
        racy=SHARED_LOGGER_RACE,
    ),
    Case(
        "mcp_tool_runs_then_answers_stream",
        _wiki(stream=True),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        gateway=True,
        racy=SHARED_LOGGER_RACE,
    ),
    Case("mcp_answers_without_a_tool", _wiki(), gateway=True),
    Case(
        "mcp_tool_beside_a_native_tool",
        _wiki(tools=[dict(MCP_REFERENCE), dict(WEATHER_TOOL)], system="Be terse.", temperature=0.2),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        gateway=True,
        racy=SHARED_LOGGER_RACE,
    ),
    Case(
        "mcp_whole_gateway",
        _wiki(tools=[{**MCP_REFERENCE, "server_url": "litellm_proxy"}]),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        gateway=True,
        racy=SHARED_LOGGER_RACE,
    ),
    Case(
        "mcp_tool_needs_approval",
        _wiki(tools=[dict(MCP_NEEDS_APPROVAL)]),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE),),
        gateway=True,
    ),
    Case(
        "mcp_tool_needs_approval_stream",
        _wiki(tools=[dict(MCP_NEEDS_APPROVAL)], stream=True),
        upstream=(_stream(WIKI_TOOL_STREAM),),
        gateway=True,
    ),
    Case(
        "mcp_follow_up_fails",
        _wiki(),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _failure(429, "rate_limit_error", "slow down")),
        gateway=True,
        racy=SHARED_LOGGER_RACE,
    ),
    Case(
        "mcp_first_call_fails",
        _wiki(),
        upstream=(_failure(400, "invalid_request_error", "max_tokens: too large"),),
        gateway=True,
    ),
    Case(
        "mcp_metadata_and_litellm_metadata",
        _wiki(metadata={"user_id": "user-123"}, litellm_metadata={"user_api_key": "hashed", "tags": ["t1"]}),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        gateway=True,
        racy=SHARED_LOGGER_RACE,
    ),
    Case(
        "mcp_unserved_proxy_path_passes_through",
        _wiki(tools=[{**MCP_REFERENCE, "server_url": "https://gateway.example/mcp/unknown"}]),
        gateway=True,
    ),
    Case(
        "mcp_skip_flag_passes_the_reference_through",
        _wiki(_skip_mcp_handler=True),
        gateway=True,
    ),
    Case(
        "mcp_tool_execution_raises",
        _wiki(),
        upstream=(_ok({**WIKI_TOOL_USE_RESPONSE, "content": [{**WIKI_TOOL_CALL, "input": {"repo": "boom"}}]}),),
        gateway=True,
    ),
    Case(
        "mcp_cache_hit",
        _wiki(),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok(), _ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        cache=Cache,
        gateway=True,
    ),
    Case("mcp_sync", _wiki(), upstream=(), sync=True, gateway=True),
    Case("mcp_mock_answers_first", _wiki(mock_response="Hi from the mock"), upstream=(), gateway=True),
    Case("mcp_sync_mock_answers_first", _wiki(mock_response="Hi from the mock"), upstream=(), sync=True, gateway=True),
    Case(
        "pre_request_hook_adds_an_mcp_tool",
        _kw(tools=[dict(WEATHER_TOOL)]),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        callbacks=(_McpToolHook,),
        gateway=True,
        racy=SHARED_LOGGER_RACE,
    ),
    Case(
        "pre_request_hook_removes_the_mcp_tool",
        _wiki(),
        callbacks=(_UninterceptedToolHook,),
        gateway=True,
    ),
    Case(
        "interceptor_answers_before_the_mcp_gateway",
        _kw(tools=[dict(INTERCEPTED_TOOL), dict(MCP_REFERENCE)]),
        upstream=(),
        interceptor=_Answering,
        gateway=True,
    ),
)


class _OrderedMemory(InMemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: records the backend writes in order
        self.ops.append(f"set {key}")
        await super().async_set_cache(key, value, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # InMemoryCache leaves its signature untyped

    async def async_get_cache(self, key: str, **kwargs: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: records the backend reads in order
        self.ops.append(f"get {key}")
        return await super().async_get_cache(key, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # InMemoryCache leaves its signature untyped


def _ordered_cache() -> Cache:
    cache: Final = Cache()
    cache.cache = _OrderedMemory()
    return cache


EMPTY_CONTENT_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType({**TEXT_RESPONSE, "content": []})
EMPTY_TEXT_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {**TEXT_RESPONSE, "content": [{"type": "text", "text": ""}]}
)

HOOKED_STREAM_AND_CACHE_CASES: Final = (
    Case(
        "pre_request_hook_streams_runs_the_deployment_success_hook",
        callbacks=(_UnstreamingHook,),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_DeploymentHooks),
    ),
    Case(
        "pre_request_hook_streams_behind_a_cache",
        callbacks=(_UnstreamingHook,),
        upstream=(_stream(TEXT_STREAM), _stream(TEXT_STREAM)),
        cache=Cache,
    ),
    Case(
        "stream_one_runs_the_deployment_success_hook",
        _kw(stream=1),
        upstream=(_stream(TEXT_STREAM),),
        loggers=_loggers(_DeploymentHooks),
    ),
    Case(
        "stream_one_behind_a_cache", _kw(stream=1), upstream=(_stream(TEXT_STREAM), _stream(TEXT_STREAM)), cache=Cache
    ),
    Case(
        "stream_one_sync_behind_a_cache",
        _kw(stream=1),
        upstream=(_stream(TEXT_STREAM), _stream(TEXT_STREAM)),
        cache=Cache,
        sync=True,
    ),
    Case(
        "stream_string_true_behind_a_cache",
        _kw(stream="true"),
        upstream=(_stream(TEXT_STREAM), _stream(TEXT_STREAM)),
        cache=Cache,
    ),
    Case(
        "pre_request_hook_streams_on_an_integer_behind_a_cache",
        callbacks=(_IntegerStreamHook,),
        upstream=(_stream(TEXT_STREAM), _stream(TEXT_STREAM)),
        cache=Cache,
    ),
    Case(
        "cache_empty_content_is_not_stored",
        upstream=(_ok(EMPTY_CONTENT_RESPONSE), _ok(EMPTY_CONTENT_RESPONSE)),
        cache=Cache,
    ),
    Case(
        "cache_empty_text_block_is_not_stored",
        upstream=(_ok(EMPTY_TEXT_RESPONSE), _ok(EMPTY_TEXT_RESPONSE)),
        cache=Cache,
    ),
    Case(
        "mcp_cache_hit_stream",
        _wiki(stream=True),
        upstream=(_ok(WIKI_TOOL_USE_RESPONSE), _ok(), _ok(WIKI_TOOL_USE_RESPONSE), _ok()),
        cache=Cache,
        gateway=True,
    ),
    Case(
        "safeguards_classifier_stream_cached",
        _safeguarded(stream=True),
        upstream=(_stream(TOOL_STREAM), _stream(TOOL_STREAM)),
        model=NON_CLAUDE,
        classifier=FLAGGED,
        cache=_ordered_cache,
    ),
)

CASES: Final = (
    *REQUEST_CASES,
    *INPUT_VALUE_CASES,
    *ARGUMENT_TYPE_CASES,
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
    *INTERCEPTOR_CASES,
    *SHORT_CIRCUIT_CASES,
    *MCP_CASES,
    *DEPLOYMENT_HOOK_CASES,
    *HOOKED_STREAM_AND_CACHE_CASES,
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
    warnings: tuple[str, ...] = ()
    hook_inputs: tuple[object, ...] = ()
    classified: tuple[Mapping[str, object], ...] = ()
    tool_runs: tuple[Mapping[str, object], ...] = ()


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
def _interceptor(interceptor: Callable[[], MessagesInterceptor] | None) -> Generator[None]:
    if interceptor is None:
        yield
        return
    with rebound(messages_interceptors, "_interceptors", [interceptor(), *messages_interceptors._interceptors]):  # pyright: ignore[reportPrivateUsage]  # registers a test interceptor ahead of the shipped ones
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
    return {name: value for name, value in kwargs.items() if value is not None or name in ("safeguards", *case.nones)}


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
            "body": _sent_body(sent),
        },
    }


def _sent_body(sent: httpx.Request) -> object:
    try:
        sent_body: Final = sent.read()
    except (httpx.RequestNotRead, httpx.StreamClosed):
        return None
    return json.loads(sent_body) if sent_body else None


def _partition(chunks: Iterable[object]) -> tuple[int, ...]:
    return tuple(len(chunk if isinstance(chunk, bytes) else str(chunk).encode()) for chunk in chunks)


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
    header_names: Final = sorted(name for name in headers if name != "x-litellm-rust")
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
        time.sleep(case.busy_before_reading)  # noqa: ASYNC251  # holds the event loop, as a caller's own work between the call and its first read does
        chunks, failure = await _consume_async(cast(AsyncIterator[object], result))
    hidden: Final = _hidden_params(result)
    partition: Final = {"partition": _partition(chunks)} if case.partitioned else {}
    if failure is not None:
        return {**_error(failure, "stream", chunks), **partition}, hidden
    return {
        "kind": "stream",
        "chunk_types": sorted({type(chunk).__name__ for chunk in chunks}),
        "text": _SERVER_TOOL_ID.sub("srvtoolu_<id>", _stream_text(chunks)),
        **partition,
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
        "logged_response_headers": {
            name: value for name, value in _mapping(details.get("response_headers")).items() if name.lower() != "date"
        },
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
        "logged_model_parameters": standard.get("model_parameters"),
        "logged_error": _logged_error(details, standard),
    }
    return logged, callback_kwargs


def _logged_error(details: Mapping[str, object], standard: Mapping[str, object]) -> Mapping[str, object]:
    exception: Final = details.get("exception")
    information: Final = _mapping(standard.get("error_information"))
    return {
        "error_str": _normalized(str(standard.get("error_str"))),
        "error_message": _normalized(str(information.get("error_message"))),
        "error_code": information.get("error_code"),
        "llm_provider": information.get("llm_provider"),
        "exception": None if exception is None else _normalized(str(exception)),
        "exception_model": getattr(exception, "model", None),
        "exception_provider": getattr(exception, "llm_provider", None),
    }


async def _prime(case: Case, server: RecordingServer, recorder: RecordingLogger, native: bool) -> int:
    if case.cache is None:
        return 0
    outcome: Final = await _invoke(case, _kwargs(case, server), native=native)
    assert case.sync or outcome[0]["kind"] != "error", outcome
    await _terminal_events(recorder)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    return len(recorder.events)


WIKI_MCP_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "name": "read_wiki",
        "description": "Read a repository wiki",
        "inputSchema": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    }
)


@contextmanager
def _gateway(case: Case, runs: list[Mapping[str, object]]) -> Generator[None]:  # mutable-ok: records the tool runs
    if not case.gateway:
        yield
        return
    from mcp.types import Tool as MCPTool

    from litellm.responses.mcp import tool_precheck
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    split_mcp_tools: Final = LiteLLM_Proxy_MCP_Handler.split_mcp_tools

    async def served(names: Iterable[str]) -> frozenset[str]:
        return frozenset({"deepwiki"}) & frozenset(names)

    async def split(tools: Iterable[Mapping[str, object]] | None) -> object:
        return await split_mcp_tools(tools, served)

    async def listed(**_: object) -> tuple[list[MCPTool], list[str]]:
        return [MCPTool.model_validate(dict(WIKI_MCP_TOOL))], ["deepwiki"]

    async def executed(**kwargs: object) -> list[dict[str, object]]:
        calls: Final = cast(list[Mapping[str, object]], kwargs["tool_calls"])
        runs.append(
            {
                "tool_calls": [dict(call) for call in calls],
                "tool_server_map": kwargs["tool_server_map"],
                "request_tags": kwargs["request_tags"],
                "guardrail_context": json.loads(json.dumps(kwargs["guardrail_context"], default=repr)),
            }
        )
        if any(call.get("input") == {"repo": "boom"} for call in calls):
            raise RuntimeError("the wiki server went away")
        return [{"tool_call_id": call.get("id"), "result": "9 sections", "name": call.get("name")} for call in calls]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tool_precheck, "_gateway_registry_loads", lambda: True)
        patch.setattr(LiteLLM_Proxy_MCP_Handler, "split_mcp_tools", staticmethod(split))
        patch.setattr(LiteLLM_Proxy_MCP_Handler, "_get_mcp_tools_from_manager", staticmethod(listed))
        patch.setattr(LiteLLM_Proxy_MCP_Handler, "execute_tool_calls", staticmethod(executed))
        yield


async def _leg(case: Case, server: RecordingServer, probe: FallbackProbe | None) -> Leg:
    classified: Final[list[Mapping[str, object]]] = []
    runs: Final[list[Mapping[str, object]]] = []
    with (
        isolated_callback_registries(),
        _environment(case.env),
        _settings(case.settings),
        _cache(case.cache),
        _classifier(case.classifier, classified),
        _interceptor(case.interceptor),
        _gateway(case, runs),
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
        with _warnings() as warned:
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
            cached=() if litellm.cache is None else _cache_entries(litellm.cache),
            warnings=tuple(warned),
            hook_inputs=tuple(getattr(logger, "seen", ()) for logger in loggers),
            classified=tuple(classified),
            tool_runs=tuple(runs),
        )


class _WarningLines(logging.Handler):
    def __init__(self, lines: list[str]) -> None:  # mutable-ok: the handler appends each warning as it is logged
        super().__init__(logging.WARNING)
        self.lines: Final = lines

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@contextmanager
def _warnings() -> Generator[list[str]]:  # mutable-ok: filled while the call runs
    lines: Final[list[str]] = []
    handler: Final = _WarningLines(lines)
    verbose_logger.addHandler(handler)
    try:
        yield lines
    finally:
        verbose_logger.removeHandler(handler)


def _cache_entries(cache: Cache) -> tuple[str, ...]:
    backend: Final = cache.cache
    keys: Final = tuple(sorted(backend.cache_dict)) if isinstance(backend, InMemoryCache) else ()
    return (*keys, *backend.ops) if isinstance(backend, _OrderedMemory) else keys


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
            assert _comparable(native.requests) == _comparable(python.requests)
        case "outcome":
            assert native.outcome == python.outcome
        case "hidden_params":
            assert native.hidden_params == python.hidden_params
        case "logged":
            assert native.logged == python.logged
        case "cached":
            assert native.cached == python.cached
        case "warnings":
            assert native.warnings == python.warnings
        case "hook_inputs":
            assert native.hook_inputs == python.hook_inputs
        case "classified":
            assert native.classified == python.classified
        case "tool_runs":
            assert native.tool_runs == python.tool_runs
        case _:
            assert _comparable(native.callback_kwargs.get(aspect)) == _comparable(python.callback_kwargs.get(aspect))


def _comparable(value: object) -> object:
    match value:
        case float() if math.isnan(value):
            return "<NaN>"
        case Mapping():
            return {name: _comparable(item) for name, item in cast(Mapping[object, object], value).items()}
        case list() | tuple():
            return [_comparable(item) for item in cast(Sequence[object], value)]
        case _:
            return value


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.name)
        for case in (*SERVED_NATIVELY_CASES, *AGENTIC_CASES, *SAFEGUARDS_CASES, *SHORT_CIRCUIT_CASES)
    ],
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
        features: Final = dispatch._FEATURES  # pyright: ignore[reportPrivateUsage]  # the entry's own detectors
        request: Final = dispatch._public_request(binding, (), _kwargs(case, parity_server))  # pyright: ignore[reportPrivateUsage]  # the public binding the entry routes by
        assert request is not None
        context: Final = dispatch._context(request, features)  # pyright: ignore[reportPrivateUsage]  # the entry's own route context
        assert catalog.rollout(context) is Rollout.RUST_REQUIRED, context.features


@pytest.mark.parametrize("case", [pytest.param(case, id=case.name) for case in TRANSPORT_CASES])
def test_catalog_serves_the_transport_case_natively(case: Case, parity_server: RecordingServer) -> None:
    with isolated_callback_registries(), _environment(case.env), _settings(case.settings):
        request: Final = dispatch._public_request(dispatch._AMESSAGES, (), _kwargs(case, parity_server))  # pyright: ignore[reportPrivateUsage]  # the public binding the async entry routes by
        assert request is not None
        context: Final = dispatch._context(request, dispatch._FEATURES)  # pyright: ignore[reportPrivateUsage]  # the async entry's own route context
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


class _CountingAnswering(_Answering):
    def __init__(self) -> None:
        self.asked = 0

    def can_handle(self, tools: Sequence[Mapping[str, object]] | None, custom_llm_provider: str | None) -> bool:
        self.asked += 1
        return super().can_handle(tools, custom_llm_provider)


async def _interceptor_asks(server: RecordingServer, hooked: bool, native: bool) -> tuple[int, tuple[object, ...]]:
    counting: Final = _CountingAnswering()
    acall: Final = _PUBLIC_AMESSAGES if native else _PYTHON_AMESSAGES
    kwargs: Final[dict[str, object]] = {
        "model": SONNET,
        "max_tokens": 64,
        "messages": [dict(USER)],
        "api_key": API_KEY,
        "api_base": server.base_url,
        "tools": [dict(WEATHER_TOOL)] if hooked else [dict(INTERCEPTED_TOOL), dict(WEATHER_TOOL)],
    }
    answers: Final[list[object]] = []
    with (
        isolated_callback_registries(),
        _environment({}),
        _cache(Cache),
        _interceptor(lambda: counting),
    ):
        if hooked:
            litellm.callbacks.append(_InterceptedToolHook())
        for _ in range(2):
            answers.append(_response(await acall(**copy.deepcopy(kwargs)), private_keys=False))
            await drain_logging()
            await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    return counting.asked, tuple(answers)


@pytest.mark.asyncio
@pytest.mark.parametrize("hooked", (False, True), ids=("plain", "pre_request_hooked"))
async def test_both_lanes_ask_the_interceptor_only_on_a_cache_miss(
    hooked: bool, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python: Final = await _interceptor_asks(parity_server, hooked, native=False)
    native: Final = await _interceptor_asks(parity_server, hooked, native=True)

    assert native_messages.calls == 0
    assert not parity_server.requests
    assert native == python
    assert python[0] == 1


class _ServiceRecorder:
    def __init__(self) -> None:
        self.events: list[Mapping[str, object]] = []  # mutable-ok: appended per service hook

    async def async_service_success_hook(self, **kwargs: object) -> None:  # kwargs-ok: the service hook's keyword call
        self.events.append(
            {
                "service": kwargs.get("service"),
                "call_type": kwargs.get("call_type"),
                "timed": isinstance(kwargs.get("duration"), float),
                "parent_otel_span": kwargs.get("parent_otel_span"),
            }
        )


async def _llm_api_timings(server: RecordingServer, stream: bool, native: bool) -> tuple[Mapping[str, object], ...]:
    recorder: Final = _ServiceRecorder()
    server.enqueue(_stream(TEXT_STREAM) if stream else _ok())
    acall: Final = _PUBLIC_AMESSAGES if native else _PYTHON_AMESSAGES
    with isolated_callback_registries(), _environment({}), rebound(logging_utils, "_service_logger", recorder):
        result: Final = await acall(
            model=SONNET,
            max_tokens=64,
            messages=[dict(USER)],
            api_key=API_KEY,
            api_base=server.base_url,
            stream=stream,
        )
        if not isinstance(result, dict):
            await _consume_async(cast(AsyncIterator[object], result))
        await drain_logging()
        await asyncio.sleep(0)
    server.responses.clear()
    return tuple(recorder.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True), ids=("json", "streamed"))
async def test_both_lanes_time_the_provider_exchange_as_one_llm_service_event(
    stream: bool, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python: Final = await _llm_api_timings(parity_server, stream, native=False)
    native: Final = await _llm_api_timings(parity_server, stream, native=True)

    assert native_messages.calls == 0
    assert native == python
    assert [event["call_type"] for event in python] == ["post <- track_llm_api_timing"]


class _AttemptRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes untyped kwargs
        self.seen: list[Mapping[str, object]] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        logged: Final = _LOGGED.validate_python(kwargs.get("standard_logging_object") or {})
        self.seen.append(
            {
                "event": "success",
                "model_id": logged.get("model_id"),
                "completion_tokens": logged.get("completion_tokens"),
            }
        )

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows the hook's types
        logged: Final = _LOGGED.validate_python(kwargs.get("standard_logging_object") or {})
        self.seen.append(
            {
                "event": "failure",
                "model_id": logged.get("model_id"),
                "exception": type(kwargs.get("exception")).__name__,
            }
        )


async def _stream_fallback_attempts(
    server: RecordingServer, native: bool
) -> tuple[tuple[Mapping[str, object], ...], str]:
    overloaded: Final = _events(
        _message_start("msg_01ParityOverloaded"), ("error", dict(_upstream_error("overloaded_error", "Overloaded")))
    )
    server.enqueue(_stream(overloaded))
    server.enqueue(_stream(TEXT_STREAM))
    recorder: Final = _AttemptRecorder()
    deployment: Final[Mapping[str, object]] = {"model": SONNET, "api_base": server.base_url, "api_key": API_KEY}
    with contextlib.ExitStack() as stack:
        stack.enter_context(isolated_callback_registries())
        stack.enter_context(_environment({}))
        if not native:
            stack.enter_context(rebound(litellm, "anthropic_messages", _PYTHON_AMESSAGES))
        litellm.callbacks.append(recorder)
        router: Final = litellm.Router(
            model_list=[
                {"model_name": "primary", "litellm_params": dict(deployment), "model_info": {"id": "dep-primary"}},
                {"model_name": "fallback", "litellm_params": dict(deployment), "model_info": {"id": "dep-fallback"}},
            ],
            fallbacks=[{"primary": ["fallback"]}],
            num_retries=0,
        )
        result: Final = await router.aanthropic_messages(
            model="primary", max_tokens=64, messages=[dict(USER)], stream=True
        )
        chunks, failure = await _consume_async(cast(AsyncIterator[object], result))
        assert failure is None
        for _ in range(100):
            await drain_logging()
            if len(recorder.seen) >= 2:
                break
            await asyncio.sleep(0.05)
    server.responses.clear()
    return tuple(recorder.seen), _stream_text(chunks)


@pytest.mark.asyncio
async def test_both_lanes_log_a_stream_superseded_by_a_router_fallback_as_its_frame_s_failure(
    parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python: Final = await _stream_fallback_attempts(parity_server, native=False)
    native: Final = await _stream_fallback_attempts(parity_server, native=True)

    assert native_messages.calls == 0
    assert native == python
    assert sorted(str(event["event"]) for event in python[0]) == ["failure", "success"]


async def _router_stream(
    server: RecordingServer, native: bool, upstream: tuple[ResponseSpec, ...]
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
    for response in upstream:
        server.enqueue(response)
    recorder: Final = _AttemptRecorder()
    deployment: Final[Mapping[str, object]] = {"model": SONNET, "api_base": server.base_url, "api_key": API_KEY}
    with contextlib.ExitStack() as stack:
        stack.enter_context(isolated_callback_registries())
        stack.enter_context(_environment({}))
        if not native:
            stack.enter_context(rebound(litellm, "anthropic_messages", _PYTHON_AMESSAGES))
        litellm.callbacks.append(recorder)
        router: Final = litellm.Router(
            model_list=[
                {"model_name": "primary", "litellm_params": dict(deployment), "model_info": {"id": "dep-primary"}},
                {"model_name": "fallback", "litellm_params": dict(deployment), "model_info": {"id": "dep-fallback"}},
            ],
            fallbacks=[{"primary": ["fallback"]}],
            num_retries=0,
        )
        result: Final = await router.aanthropic_messages(
            model="primary", max_tokens=64, messages=[dict(USER)], stream=True
        )
        chunks, failure = await _consume_async(cast(AsyncIterator[object], result))
        assert failure is None
        for _ in range(100):
            await drain_logging()
            if recorder.seen:
                break
            await asyncio.sleep(0.05)
    server.responses.clear()
    return tuple(_stream_text((chunk,)) for chunk in chunks), tuple(recorder.seen)


OVERLOADED_AFTER_PINGS: Final = _events(
    _message_start("msg_01ParityPingedOverloaded"),
    *((_PING,) * 50),
    ("error", dict(_upstream_error("overloaded_error", "Overloaded"))),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "upstream",
    [
        pytest.param((_burst(PINGED_STREAM),), id="pings_after_message_start"),
        pytest.param((_burst(HEAVILY_PINGED_STREAM),), id="a_thousand_pings_after_message_start"),
        pytest.param((_burst(OVERLOADED_AFTER_PINGS), _burst(TEXT_STREAM)), id="pings_then_overloaded_then_fallback"),
    ],
)
async def test_both_lanes_stream_a_router_burst_in_the_same_chunks(
    parity_server: RecordingServer, native_messages: FallbackProbe, upstream: tuple[ResponseSpec, ...]
) -> None:
    python: Final = await _router_stream(parity_server, native=False, upstream=upstream)
    native: Final = await _router_stream(parity_server, native=True, upstream=upstream)

    assert native_messages.calls == 0
    assert native == python


async def _abandoned_stream(
    server: RecordingServer, native: bool, upstream: ResponseSpec
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
    server.enqueue(upstream)
    recorder: Final = _AttemptRecorder()
    acall: Final = _PUBLIC_AMESSAGES if native else _PYTHON_AMESSAGES
    with contextlib.ExitStack() as stack:
        stack.enter_context(isolated_callback_registries())
        stack.enter_context(_environment({}))
        litellm.callbacks.append(recorder)
        stream: Final = cast(
            AsyncGenerator[object],
            await acall(
                model=SONNET,
                max_tokens=64,
                messages=[dict(USER)],
                api_key=API_KEY,
                api_base=server.base_url,
                stream=True,
            ),
        )
        first: Final = await anext(stream)
        await stream.aclose()
        for _ in range(100):
            await drain_logging()
            if recorder.seen:
                break
            await asyncio.sleep(0.05)
    server.responses.clear()
    return (_stream_text((first,)),), tuple(recorder.seen)


@pytest.mark.asyncio
async def test_both_lanes_bill_a_stream_abandoned_after_its_first_chunk_alike(
    parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python: Final = await _abandoned_stream(parity_server, native=False, upstream=_burst(TEXT_STREAM))
    native: Final = await _abandoned_stream(parity_server, native=True, upstream=_burst(TEXT_STREAM))

    assert native_messages.calls == 0
    assert native == python
