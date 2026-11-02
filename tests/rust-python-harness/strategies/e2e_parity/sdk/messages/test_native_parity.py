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
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Final, cast

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.messages import dispatch
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

SONNET: Final = "anthropic/claude-sonnet-4-5"
OPUS_ADAPTIVE: Final = "anthropic/claude-opus-4-7"
OPUS_FAST: Final = "anthropic/claude-opus-5-5"
API_KEY: Final = "sk-ant-parity-test"
USER: Final[Mapping[str, JsonValue]] = MappingProxyType({"role": "user", "content": "What is the capital of France?"})

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_LOGGED: Final = TypeAdapter(dict[str, object])
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
    gaps: Mapping[str, str | None] = field(default_factory=lambda: MappingProxyType({}))

    def params(self) -> Iterator[object]:
        for aspect in ASPECTS:
            gap = {**KNOWN_GAPS.get(self.name, {}), **self.gaps}.get(aspect)
            marks = () if not gap else (pytest.mark.xfail(reason=gap, strict=True),)
            yield pytest.param(self, aspect, id=f"{self.name}-{aspect}", marks=marks)


def _gaps(**gaps: str | None) -> Mapping[str, str | None]:
    return MappingProxyType(gaps)


def _kw(**kwargs: object) -> Mapping[str, object]:
    return MappingProxyType(kwargs)


A2_BARE_MODEL: Final = "A2: native resolves the provider from the model prefix only, so a bare Claude model name fails"
A10_NO_MAX_TOKENS: Final = "A10: native admission declines a call without an int max_tokens and Python serves it"
R3_LEGACY_THINKING: Final = "R3: native maps a float legacy thinking budget to effort low instead of Python's tier"
R4_BINDING_BETA: Final = "R4: native omits the thinking-binding-controls beta for block_binding"
R5_DEPLOYMENT_EFFORT: Final = "R5: native ignores the deployment's reasoning_effort_levels from model_info"
R7_DROP_ORDER: Final = "R7: native applies additional_drop_params before Python's unsupported-param check"
R8_VERSION_HEADER: Final = "R8: Python sends anthropic-version twice when the caller capitalises it; native once"
R9_QUERY_PATH: Final = "R9: native appends /v1/messages again to an api_base that ends in /v1/messages?query"
R11_PARAM_ERROR: Final = "R11: native raises BadRequestError where Python raises UnsupportedParamsError"
R12_METADATA_ERROR: Final = (
    "R12: Python raises a pydantic ValidationError for a non-string metadata.user_id; native a BadRequestError"
)
S1_S2_ERROR_HEADERS: Final = (
    "S1/S2: native drops upstream error headers and wraps the error body in a (status, body) tuple"
)
S2_ERROR_MESSAGE: Final = (
    "S2: native wraps the upstream error body in a (status, body) tuple and strips the provider from model"
)
S2_S12_ERROR_STATUS: Final = "S2/S12: native keeps the upstream status where Python maps it (413 to 400, 529 to 500)"
S3_TIMEOUT: Final = "S3: native raises APIConnectionError (500) for a timeout where Python raises Timeout (408)"
S5_STREAM_DEADLINE: Final = "S5: native applies the timeout to the whole stream instead of to each read"
S6_PRIVATE_KEYS: Final = "S6: native writes _hidden_params into the response dict"
S8_TYPED_RESPONSE: Final = "S8: native re-serializes the response and drops null usage and container"
S9_SYNC: Final = "S9: Python's sync handler raises ValueError for native providers; native serves the call"
S13_INVALID_JSON: Final = "S13: native reports an unparseable success body as an internal error, not AnthropicException"
S14_MISSING_KEY: Final = (
    "S14: native raises APIConnectionError for a missing API key where Python raises AuthenticationError"
)
C4_METADATA: Final = "C4: native callbacks get the raw Anthropic metadata, not Python's merged litellm metadata"
C6_RESPONSE_HEADERS: Final = "C6: native success logs carry no response_headers"
C7_API_KEY: Final = "C7: native callbacks see the caller's api_key where Python logs an empty string"
C8_STREAM_HIDDEN: Final = "C8: native streams lack the call metadata @client stamps into _hidden_params"
F_MOCK: Final = "F: mock_response is Python-only (detector mock_response)"
F_THINKING_RETRY: Final = (
    "F: retry without thinking after a rejected signature is Python-only (detector thinking_signature_retry)"
)
F_CACHE_INJECTION: Final = "F: prompt cache_control injection is Python-only (detector cache_control_injection)"
F_TOOL_CUSTOM: Final = "F: hoisting a tool's custom field is Python-only (detector tool_custom_field)"
F_STREAM_COST: Final = "F: cost injection into streamed usage is Python-only (detector stream_cost_injection)"

KNOWN_GAPS: Final[Mapping[str, Mapping[str, str | None]]] = MappingProxyType(
    {
        "plain_text": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "system_string": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "system_blocks_with_cache_control": _gaps(
            logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "multi_turn_tool_result": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "tools_with_tool_choice": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "tool_choice_none": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "tool_input_examples": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "sampling_params": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "stop_sequences": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "metadata_user_id": _gaps(
            logged_metadata=C4_METADATA, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "metadata_extra_keys_are_filtered": _gaps(
            logged_metadata=C4_METADATA, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "metadata_user_id_not_a_string": _gaps(outcome=R12_METADATA_ERROR, logged=R12_METADATA_ERROR),
        "service_tier": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "thinking_enabled": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "thinking_disabled": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "legacy_thinking_on_adaptive_model_keeps_display": _gaps(
            logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "legacy_thinking_float_budget_on_adaptive_model": _gaps(
            request=R3_LEGACY_THINKING, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "adaptive_thinking_with_block_binding": _gaps(
            request=R4_BINDING_BETA, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "reasoning_effort": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "reasoning_effort_on_adaptive_model": _gaps(
            logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "output_config_effort": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "output_format_json_schema": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "speed_on_model_without_speed": _gaps(outcome=R11_PARAM_ERROR, logged=R11_PARAM_ERROR),
        "speed_on_model_without_speed_drop_params": _gaps(
            logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "speed_on_fast_model": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "temperature_on_adaptive_model": _gaps(outcome=R11_PARAM_ERROR, logged=R11_PARAM_ERROR),
        "top_k_on_adaptive_model": _gaps(outcome=R11_PARAM_ERROR, logged=R11_PARAM_ERROR),
        "top_k_on_adaptive_model_drop_params": _gaps(
            logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "additional_drop_params_hides_rejected_temperature": _gaps(
            request=R7_DROP_ORDER,
            outcome=R7_DROP_ORDER,
            hidden_params=R7_DROP_ORDER,
            logged=R7_DROP_ORDER,
            logged_metadata=R7_DROP_ORDER,
            logged_api_key=R7_DROP_ORDER,
            logged_call=R7_DROP_ORDER,
        ),
        "additional_drop_params_nested_path": _gaps(
            logged_metadata=C4_METADATA, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "context_management_edits": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "container_skills": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "remote_mcp_servers": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "inference_geo": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "top_level_cache_control": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "null_safeguards_are_stripped": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "empty_text_blocks_are_stripped": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "tool_use_ids_are_sanitized": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "provider_specific_fields_are_stripped": _gaps(
            logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "image_block": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "document_block": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "server_web_search_tool": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "extra_headers": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "anthropic_beta_from_headers_and_extra_headers": _gaps(
            logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "caller_anthropic_version_capitalised": _gaps(
            request=R8_VERSION_HEADER, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "provider_specific_header": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "explicit_custom_llm_provider": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "bare_model_name": _gaps(**dict.fromkeys(ASPECTS[1:], A2_BARE_MODEL) | {"logged_model_info": None}),
        "api_base_with_messages_path": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "api_base_with_messages_path_and_query": _gaps(
            request=R9_QUERY_PATH,
            logged=R9_QUERY_PATH,
            logged_response_headers=C6_RESPONSE_HEADERS,
            logged_api_key=C7_API_KEY,
        ),
        "api_key_from_environment": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "auth_token_from_environment": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "reasoning_auto_summary": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "global_drop_params": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "request_timeout_setting": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "litellm_metadata_and_metadata": _gaps(
            logged_metadata=C4_METADATA, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "model_info_with_id": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "deployment_effort_levels": _gaps(
            request=R5_DEPLOYMENT_EFFORT, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "response_with_tool_use": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "response_with_thinking": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "response_with_cache_usage": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "response_with_null_usage_and_container": _gaps(outcome=S8_TYPED_RESPONSE),
        "response_without_stop_sequence": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "response_with_unknown_field": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "response_key_order": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "response_upstream_headers": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "response_private_keys": _gaps(
            outcome=S6_PRIVATE_KEYS, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "upstream_400": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "upstream_401": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "upstream_403": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "upstream_404": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "upstream_413": _gaps(outcome=S2_S12_ERROR_STATUS, logged_api_key=C7_API_KEY),
        "upstream_429_with_retry_after": _gaps(outcome=S1_S2_ERROR_HEADERS, logged_api_key=C7_API_KEY),
        "upstream_500": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "upstream_529_overloaded": _gaps(outcome=S2_S12_ERROR_STATUS, logged_api_key=C7_API_KEY),
        "upstream_400_context_window": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "upstream_non_json_error": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "upstream_invalid_json_success": _gaps(
            outcome=S13_INVALID_JSON, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "upstream_timeout": _gaps(outcome=S3_TIMEOUT, logged=S3_TIMEOUT, logged_api_key=C7_API_KEY),
        "missing_max_tokens": _gaps(dispatch=A10_NO_MAX_TOKENS),
        "missing_api_key": _gaps(outcome=S14_MISSING_KEY, logged=S14_MISSING_KEY),
        "stream_text": _gaps(hidden_params=C8_STREAM_HIDDEN, logged_api_key=C7_API_KEY),
        "stream_tool_use": _gaps(hidden_params=C8_STREAM_HIDDEN, logged_api_key=C7_API_KEY),
        "stream_thinking": _gaps(hidden_params=C8_STREAM_HIDDEN, logged_api_key=C7_API_KEY),
        "stream_error_event_mid_stream": _gaps(hidden_params=C8_STREAM_HIDDEN, logged_api_key=C7_API_KEY),
        "stream_longer_than_timeout": _gaps(
            outcome=S5_STREAM_DEADLINE,
            hidden_params=S5_STREAM_DEADLINE,
            logged=S5_STREAM_DEADLINE,
            logged_metadata=S5_STREAM_DEADLINE,
            logged_api_key=S5_STREAM_DEADLINE,
        ),
        "stream_upstream_429": _gaps(outcome=S1_S2_ERROR_HEADERS, logged_api_key=C7_API_KEY),
        "stream_false_explicit": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "sync_text": _gaps(
            request=S9_SYNC,
            outcome=S9_SYNC,
            hidden_params=S9_SYNC,
            logged=S9_SYNC,
            logged_metadata=S9_SYNC,
            logged_response_headers=S9_SYNC,
            logged_api_key=S9_SYNC,
            logged_call=S9_SYNC,
        ),
        "sync_stream": _gaps(
            request=S9_SYNC,
            outcome=S9_SYNC,
            hidden_params=S9_SYNC,
            logged=S9_SYNC,
            logged_metadata=S9_SYNC,
            logged_response_headers=S9_SYNC,
            logged_api_key=S9_SYNC,
            logged_call=S9_SYNC,
        ),
        "sync_upstream_429": _gaps(
            request=S9_SYNC,
            outcome=S9_SYNC,
            logged=S9_SYNC,
            logged_response_headers=S9_SYNC,
            logged_api_key=S9_SYNC,
            logged_call=S9_SYNC,
        ),
        "mock_response": _gaps(
            request=F_MOCK,
            outcome=F_MOCK,
            hidden_params=F_MOCK,
            logged=F_MOCK,
            logged_metadata=F_MOCK,
            logged_api_key=F_MOCK,
            logged_call=F_MOCK,
        ),
        "mock_response_rate_limit": _gaps(
            request=F_MOCK, outcome=F_MOCK, logged=F_MOCK, logged_api_key=F_MOCK, logged_call=F_MOCK
        ),
        "thinking_replay": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "thinking_signature_rejected_then_retried": _gaps(
            request=F_THINKING_RETRY,
            outcome=F_THINKING_RETRY,
            hidden_params=F_THINKING_RETRY,
            logged=F_THINKING_RETRY,
            logged_metadata=C4_METADATA,
            logged_response_headers=C6_RESPONSE_HEADERS,
            logged_api_key=C7_API_KEY,
        ),
        "cache_control_injection_points": _gaps(
            request=F_CACHE_INJECTION, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "tool_custom_field": _gaps(
            request=F_TOOL_CUSTOM, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "strict_tools_rejected_then_retried": _gaps(outcome=S2_ERROR_MESSAGE, logged_api_key=C7_API_KEY),
        "modify_params": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "stream_cost_injection": _gaps(outcome=F_STREAM_COST, hidden_params=F_STREAM_COST, logged_api_key=C7_API_KEY),
        "enable_prompt_caching": _gaps(
            request=F_CACHE_INJECTION, logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY
        ),
        "request_timeout_kwarg": _gaps(logged_response_headers=C6_RESPONSE_HEADERS, logged_api_key=C7_API_KEY),
        "stream_timeout_kwarg": _gaps(hidden_params=C8_STREAM_HIDDEN, logged_api_key=C7_API_KEY),
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
    Case("service_tier", _kw(service_tier="auto"), gaps=_gaps(request="R1: native drops service_tier")),
    Case("thinking_enabled", _kw(thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096)),
    Case("thinking_disabled", _kw(thinking={"type": "disabled"})),
    Case(
        "legacy_thinking_on_adaptive_model_keeps_display",
        _kw(thinking={"type": "enabled", "budget_tokens": 2048, "display": "summarized"}, max_tokens=4096),
        model=OPUS_ADAPTIVE,
        gaps=_gaps(request="R3: native replaces legacy thinking with a bare adaptive block and drops display"),
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
        gaps=_gaps(request="R2: native keeps the last anthropic-beta source instead of the union"),
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

FALLBACK_CASES: Final = (
    Case("mock_response", _kw(mock_response="Hi from the mock"), upstream=()),
    Case("mock_response_rate_limit", _kw(mock_response="litellm.RateLimitError"), upstream=()),
    Case(
        "thinking_replay",
        _kw(messages=list(THINKING_REPLAY), thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096),
    ),
    Case(
        "thinking_signature_rejected_then_retried",
        _kw(messages=list(THINKING_REPLAY), thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096),
        upstream=(
            _failure(400, "invalid_request_error", "messages.1.content.0: Invalid `signature` in `thinking` block"),
            _ok(),
        ),
    ),
    Case(
        "cache_control_injection_points",
        _kw(system="Answer in one word.", cache_control_injection_points=[{"location": "message", "role": "system"}]),
    ),
    Case("tool_custom_field", _kw(tools=[{**WEATHER_TOOL, "custom": {"defer_loading": True}}])),
    Case(
        "strict_tools_rejected_then_retried",
        _kw(tools=[{**WEATHER_TOOL, "strict": True}], drop_params=True),
        upstream=(
            _failure(400, "invalid_request_error", "tools.0.strict: too many strict tools"),
            _ok(),
        ),
    ),
    Case("modify_params", _kw(max_tokens=1), settings=MappingProxyType({"modify_params": True})),
    Case(
        "stream_cost_injection",
        _kw(stream=True),
        upstream=(_stream(TEXT_STREAM),),
        settings=MappingProxyType({"include_cost_in_streaming_usage": True}),
    ),
    Case("enable_prompt_caching", _kw(system="Answer in one word.", enable_prompt_caching=True)),
    Case("request_timeout_kwarg", _kw(request_timeout=30)),
    Case("stream_timeout_kwarg", _kw(stream=True, stream_timeout=30), upstream=(_stream(TEXT_STREAM),)),
)

CASES: Final = (*REQUEST_CASES, *RESPONSE_CASES, *FAILURE_CASES, *STREAM_CASES, *SYNC_CASES, *FALLBACK_CASES)


@dataclass(frozen=True, slots=True)
class Leg:
    fell_back: bool
    requests: tuple[Mapping[str, object], ...]
    outcome: Mapping[str, object]
    hidden_params: Mapping[str, object]
    logged: Mapping[str, object]
    callback_kwargs: Mapping[str, object]


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
        rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_OUT), *catalog.RULES)),
        rebound(dispatch, "_PYTHON_AMESSAGES", python_afallback),
        rebound(dispatch, "_PYTHON_MESSAGES", python_fallback),
    ):
        yield probe


@contextmanager
def _environment(env: Mapping[str, str]) -> Generator[None]:
    names: Final = (*env, "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "LITELLM_REASONING_AUTO_SUMMARY")
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


def _kwargs(case: Case, server: RecordingServer) -> dict[str, object]:
    supplied: Final = dict(case.kwargs)
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
        "headers": {name: header_items.get(name) for name in ("retry-after", "request-id") if name in header_items},
        "partial": _stream_text(chunks),
    }


def _stream_text(chunks: Iterable[object]) -> str:
    return b"".join(chunk if isinstance(chunk, bytes) else str(chunk).encode() for chunk in chunks).decode()


def _response(result: object, private_keys: bool) -> Mapping[str, object]:
    assert isinstance(result, dict), f"expected a dict response, got {type(result).__name__}"
    body: Final = {
        name: value
        for name, value in cast(dict[str, object], result).items()
        if private_keys or not name.startswith("_")
    }
    return {"kind": "response", "body": _OBJECT.validate_python(body)}


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
        "text": _stream_text(chunks),
    }, hidden


async def _terminal_events(recorder: RecordingLogger) -> tuple[str, ...]:
    for _ in range(100):
        await drain_logging()
        if any(name in _TERMINAL_EVENTS for name in recorder.names):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)
    await drain_logging()
    return tuple(name for name in recorder.names if name in _TERMINAL_EVENTS)


def _logged(recorder: RecordingLogger, events: tuple[str, ...]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    terminal: Final = [event for event in recorder.events if event.name in _TERMINAL_EVENTS]
    if not terminal:
        return {"events": events}, {}
    details: Final = _mapping(terminal[-1].kwargs)
    standard: Final = _mapping(details.get("standard_logging_object"))
    error_class: Final = _mapping(standard.get("error_information")).get("error_class")
    litellm_params: Final = _mapping(details.get("litellm_params"))
    metadata: Final = litellm_params.get("metadata")
    metadata_keys: Final = _mapping(metadata).keys()
    logged: Final[Mapping[str, object]] = {
        "events": events,
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
        },
    }
    return logged, callback_kwargs


async def _leg(case: Case, server: RecordingServer, probe: FallbackProbe | None) -> Leg:
    with isolated_callback_registries(), _environment(case.env), _settings(case.settings):
        recorder: Final = RecordingLogger()
        litellm.callbacks.append(recorder)
        for spec in case.upstream:
            server.enqueue(replace(spec))
        request_count: Final = len(server.requests)
        fallbacks_before: Final = 0 if probe is None else probe.calls
        outcome, hidden = await _invoke(case, _kwargs(case, server), native=probe is not None)
        events: Final = await _terminal_events(recorder)
        server.responses.clear()
        logged, callback_kwargs = _logged(recorder, events)
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
        case _:
            assert native.callback_kwargs.get(aspect) == python.callback_kwargs.get(aspect)
