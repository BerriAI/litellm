"""Scripted provider sidecar for the cost-calculation integration suite.

A standalone process (``python -m integration._support.scripted_provider``) that
pretends to be an LLM provider for the proxy under test. The suite registers a
Scenario over a small control API; the provider wire routes then answer the
proxy's upstream calls with the scripted usage figures, in the exact wire shape
the real provider would emit (OpenAI chat completions, OpenAI Responses,
Anthropic Messages, Gemini generateContent, or the OpenAI-compatible Together /
Fireworks surfaces). Because the usage is scripted, expected spend is literal
arithmetic on the test cost map's rates, with no dependency on what a real
provider would report.

Layout on one port:

- ``GET  /health``                      liveness
- ``POST /_scenarios``                  register a Scenario JSON, returns its id
- ``DELETE /_scenarios/<id>``           remove it
- ``POST /_oauth/token``              fake Google OAuth token endpoint for the
  Vertex service-account credential's refresh call
- ``POST /<id>/<mount>/<provider path>`` provider wire; mount is one of
  ``openai``, ``anthropic``, ``gemini``, ``together``, ``fireworks``, ``azure``,
  ``bedrock``, ``vertex`` and the remainder is whatever path the provider
  client appends (``chat/completions``, ``responses``, ``v1/messages``,
  ``models/<m>:generateContent`` ...). Vertex appends ``:generateContent`` /
  ``:streamGenerateContent`` to the mount segment itself, and Bedrock Converse
  targets ``model/<modelId>/converse`` / ``converse-stream``

A request carrying ``"stream": true`` (or the ``:streamGenerateContent`` Gemini
verb) gets an SSE answer; ``stream_usage`` on the Scenario decides whether the
final stream chunk carries usage or the provider reports none.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

Wire: TypeAlias = Literal[
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate",
    "together_chat",
    "fireworks_chat",
    "azure_chat",
    "bedrock_converse",
    "vertex_generate",
]

WIRE_MOUNTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai_chat": "openai",
        "openai_responses": "openai",
        "anthropic_messages": "anthropic",
        "gemini_generate": "gemini",
        "together_chat": "together",
        "fireworks_chat": "fireworks",
        "azure_chat": "azure",
        "bedrock_converse": "bedrock",
        "vertex_generate": "vertex",
    }
)

StreamUsage: TypeAlias = Literal["final_chunk", "absent"]
ServiceTier: TypeAlias = Literal["flex", "priority"]
TerminalKind: TypeAlias = Literal["completed", "incomplete", "unvalidated", "prompt_blocked"]

# Which terminal variant each wire can represent.
_TERMINAL_CAPS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "openai_responses": frozenset({"incomplete", "unvalidated"}),
        "gemini_generate": frozenset({"prompt_blocked"}),
        "vertex_generate": frozenset({"prompt_blocked"}),
    }
)


_BASE_USAGE_FIELDS: Final = frozenset({"fresh_input_tokens", "output_tokens"})
_OPENAI_FAMILY_USAGE: Final = frozenset(
    {
        "cache_read_tokens",
        "reasoning_tokens",
        "audio_input_tokens",
        "audio_output_tokens",
        "web_search_calls",
    }
)
_CACHE_WRITE_USAGE: Final = frozenset({"cache_write_5m_tokens", "cache_write_1h_tokens"})
_GEMINI_USAGE: Final = frozenset(
    {
        "cache_read_tokens",
        "reasoning_tokens",
        "audio_input_tokens",
        "audio_output_tokens",
        "image_input_tokens",
        "video_input_tokens",
        "web_search_calls",
        "google_maps_calls",
    }
)

_USAGE_CAPS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        wire: usage
        for wire, usage in (
            ("openai_chat", _OPENAI_FAMILY_USAGE),
            ("azure_chat", _OPENAI_FAMILY_USAGE),
            ("together_chat", _OPENAI_FAMILY_USAGE),
            ("fireworks_chat", _OPENAI_FAMILY_USAGE),
            (
                "openai_responses",
                frozenset(
                    {"cache_read_tokens", "reasoning_tokens", "web_search_calls", "file_search_calls"}
                ),
            ),
            (
                "anthropic_messages",
                frozenset({"cache_read_tokens", "web_search_calls"}) | _CACHE_WRITE_USAGE,
            ),
            ("bedrock_converse", frozenset({"cache_read_tokens"}) | _CACHE_WRITE_USAGE),
            ("gemini_generate", _GEMINI_USAGE),
            ("vertex_generate", _GEMINI_USAGE),
        )
    }
)


class ScriptedToolCall(BaseModel):
    """A single function call the scripted output emits instead of text.
    ``arguments`` is the wire's JSON string (~250 chars), sliced into deltas
    for streams."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: str


class ScriptedUsage(BaseModel):
    """Physical token counts the scripted response reports. ``fresh_input_tokens``
    is the uncached, never-written, non-audio input count; ``output_tokens`` is
    the non-reasoning, non-audio output count. Renderers add the cached, written,
    audio, and reasoning counts into the wire's total fields the way the real
    provider does (inside prompt_tokens for OpenAI/Gemini, as uncached-only
    input_tokens for Anthropic)."""

    model_config = ConfigDict(frozen=True)

    fresh_input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    reasoning_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    image_input_tokens: int = 0
    video_input_tokens: int = 0
    web_search_calls: int = 0
    google_maps_calls: int = 0
    file_search_calls: int = 0


class ScriptedOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    finish_reason: str = "stop"
    # When set, emitted verbatim as the response's model field, letting a test
    # prove the biller prices the provider-reported model.
    response_model: str | None = None
    # OpenAI-compatible providers can report a provider-computed cost; emitted as
    # the top-level "cost" field on the together/fireworks wire.
    provider_cost: float | None = None
    # When set, the response is a tool call only: no text content on any wire.
    tool_call: ScriptedToolCall | None = None
    # Terminal shape: "unvalidated" makes the Responses terminal response fail
    # pydantic validation so the proxy takes its model_construct dict path;
    # "prompt_blocked" is a Gemini promptFeedback-only body.
    terminal: TerminalKind = "completed"


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    wire: Wire
    usage: ScriptedUsage
    output: ScriptedOutput
    # The bare provider-facing model name the renderer echoes when the request
    # carries no model of its own (Vertex and Bedrock name the model in the URL
    # path, not the body).
    model: str
    stream_usage: StreamUsage = "final_chunk"
    service_tier: ServiceTier | None = None
    # Anthropic fast mode and US inference geography; emitted on the anthropic
    # usage object only (litellm reads them there), so they are response-side.
    speed: Literal["fast"] | None = None
    inference_geo: Literal["us"] | None = None

    @model_validator(mode="after")
    def _check_terminal_supported(self) -> Scenario:
        if (
            self.output.terminal != "completed"
            and self.output.terminal not in _TERMINAL_CAPS.get(self.wire, frozenset())
        ):
            raise ValueError(
                f"wire {self.wire} cannot emit terminal={self.output.terminal}"
            )
        unsupported: Final = frozenset(
            field
            for field in self.usage.model_fields_set
            if getattr(self.usage, field)
            and field not in (_USAGE_CAPS.get(self.wire, frozenset()) | _BASE_USAGE_FIELDS)
        )
        if unsupported:
            raise ValueError(
                f"wire {self.wire} cannot express usage fields {sorted(unsupported)}"
            )
        if (self.speed or self.inference_geo) and self.wire != "anthropic_messages":
            raise ValueError(
                f"wire {self.wire} cannot emit speed/inference_geo (anthropic usage fields)"
            )
        return self

    @property
    def mount(self) -> str:
        return WIRE_MOUNTS[self.wire]


class ScenarioRegistered(BaseModel):
    scenario_id: str


class ScenarioDeleted(BaseModel):
    deleted: bool


class HealthStatus(BaseModel):
    status: str


@dataclass(frozen=True, slots=True)
class RenderedResponse:
    status_code: int
    content_type: str
    body: bytes


def _jobj(*pairs: tuple[str, object]) -> Mapping[str, object]:
    """A JSON object payload built in one shot and frozen."""
    return MappingProxyType(dict(pairs))


def _jobj_opt(*pairs: tuple[str, object] | None) -> Mapping[str, object]:
    """``_jobj`` where a ``None`` pair means the field is absent."""
    return MappingProxyType(dict(pair for pair in pairs if pair is not None))


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, default=dict).encode("utf-8")


def _sse_frame(event_name: str | None, data: Mapping[str, object] | str) -> str:
    head: Final = f"event: {event_name}\n" if event_name is not None else ""
    payload: Final = data if isinstance(data, str) else json.dumps(data, default=dict)
    return f"{head}data: {payload}\n\n"


def _sse(events: tuple[tuple[str | None, Mapping[str, object] | str], ...]) -> bytes:
    return "".join(_sse_frame(event_name, data) for event_name, data in events).encode("utf-8")


# ---------- per-wire usage shapes ----------


def _openai_usage(u: ScriptedUsage) -> Mapping[str, object]:
    prompt_tokens: Final = u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens
    completion_tokens: Final = u.output_tokens + u.reasoning_tokens + u.audio_output_tokens
    prompt_details: Final = _jobj_opt(
        ("cached_tokens", u.cache_read_tokens) if u.cache_read_tokens else None,
        ("audio_tokens", u.audio_input_tokens) if u.audio_input_tokens else None,
    )
    completion_details: Final = _jobj_opt(
        ("reasoning_tokens", u.reasoning_tokens) if u.reasoning_tokens else None,
        ("audio_tokens", u.audio_output_tokens) if u.audio_output_tokens else None,
    )
    return _jobj_opt(
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("total_tokens", prompt_tokens + completion_tokens),
        ("prompt_tokens_details", prompt_details) if prompt_details else None,
        ("completion_tokens_details", completion_details) if completion_details else None,
    )


def _anthropic_usage(scenario: Scenario) -> Mapping[str, object]:
    # Anthropic reports uncached-only input_tokens; cache reads and writes ride
    # top-level fields, with the 5m/1h write split under cache_creation.
    u: Final = scenario.usage
    return _jobj_opt(
        ("input_tokens", u.fresh_input_tokens),
        ("output_tokens", u.output_tokens),
        ("service_tier", scenario.service_tier) if scenario.service_tier else None,
        ("speed", scenario.speed) if scenario.speed else None,
        ("inference_geo", scenario.inference_geo) if scenario.inference_geo else None,
        ("cache_read_input_tokens", u.cache_read_tokens) if u.cache_read_tokens else None,
        (
            ("cache_creation_input_tokens", u.cache_write_5m_tokens + u.cache_write_1h_tokens)
            if u.cache_write_5m_tokens or u.cache_write_1h_tokens
            else None
        ),
        (
            (
                "cache_creation",
                _jobj(
                    ("ephemeral_5m_input_tokens", u.cache_write_5m_tokens),
                    ("ephemeral_1h_input_tokens", u.cache_write_1h_tokens),
                ),
            )
            if u.cache_write_5m_tokens or u.cache_write_1h_tokens
            else None
        ),
        (
            ("server_tool_use", _jobj(("web_search_requests", u.web_search_calls)))
            if u.web_search_calls
            else None
        ),
    )


def _gemini_usage(scenario: Scenario) -> Mapping[str, object]:
    # Real generateContent accounting: promptTokenCount carries the cached count
    # inside it (TEXT modality is the cached-inclusive text count so litellm's
    # implicit-caching subtraction lands on the fresh figure), candidatesTokenCount
    # excludes thoughts, thoughtsTokenCount reports them separately, and
    # totalTokenCount sums all three. Image/video input ride promptTokensDetails.
    u: Final = scenario.usage
    prompt_tokens: Final = (
        u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens
        + u.image_input_tokens + u.video_input_tokens
    )
    candidates: Final = u.output_tokens + u.audio_output_tokens
    return _jobj_opt(
        ("promptTokenCount", prompt_tokens),
        ("candidatesTokenCount", candidates),
        ("thoughtsTokenCount", u.reasoning_tokens) if u.reasoning_tokens else None,
        ("totalTokenCount", prompt_tokens + candidates + u.reasoning_tokens),
        ("cachedContentTokenCount", u.cache_read_tokens) if u.cache_read_tokens else None,
        (
            "promptTokensDetails",
            (
                _jobj(("modality", "TEXT"), ("tokenCount", u.fresh_input_tokens + u.cache_read_tokens)),
                *(
                    (_jobj(("modality", "AUDIO"), ("tokenCount", u.audio_input_tokens)),)
                    if u.audio_input_tokens
                    else ()
                ),
                *(
                    (_jobj(("modality", "IMAGE"), ("tokenCount", u.image_input_tokens)),)
                    if u.image_input_tokens
                    else ()
                ),
                *(
                    (_jobj(("modality", "VIDEO"), ("tokenCount", u.video_input_tokens)),)
                    if u.video_input_tokens
                    else ()
                ),
            ),
        ),
        (
            (
                "candidatesTokensDetails",
                (
                    _jobj(("modality", "TEXT"), ("tokenCount", u.output_tokens)),
                    _jobj(("modality", "AUDIO"), ("tokenCount", u.audio_output_tokens)),
                ),
            )
            if u.audio_output_tokens
            else None
        ),
        (
            (
                "trafficType",
                {"flex": "ON_DEMAND_FLEX", "priority": "ON_DEMAND_PRIORITY"}[
                    scenario.service_tier
                ],
            )
            if scenario.service_tier
            else None
        ),
    )


def _gemini_grounding_metadata(scenario: Scenario) -> Mapping[str, object] | None:
    """groundingMetadata for the search/Maps flags. Maps items carry maps
    chunks and googleMapsWidgetContextToken so litellm bills them as Maps
    queries, not web search."""
    u: Final = scenario.usage
    if not u.web_search_calls and not u.google_maps_calls:
        return None
    if u.google_maps_calls:
        return _jobj(
            (
                "webSearchQueries",
                tuple(f"maps query {i}" for i in range(u.google_maps_calls)),
            ),
            (
                "groundingChunks",
                tuple(
                    _jobj(("maps", _jobj(("uri", f"https://maps.google.com/?cid={i}"))))
                    for i in range(u.google_maps_calls)
                ),
            ),
            ("googleMapsWidgetContextToken", f"token_{scenario.scenario_id}"),
        )
    return _jobj(
        ("webSearchQueries", tuple(f"query {i}" for i in range(u.web_search_calls))),
    )


def _responses_usage(u: ScriptedUsage) -> Mapping[str, object]:
    input_tokens: Final = u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens
    output_tokens: Final = u.output_tokens + u.reasoning_tokens + u.audio_output_tokens
    input_details: Final = _jobj_opt(
        ("cached_tokens", u.cache_read_tokens) if u.cache_read_tokens else None,
    )
    return _jobj_opt(
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("total_tokens", input_tokens + output_tokens),
        ("input_tokens_details", input_details) if input_details else None,
        (
            ("output_tokens_details", _jobj(("reasoning_tokens", u.reasoning_tokens)))
            if u.reasoning_tokens
            else None
        ),
    )


# ---------- per-wire responses ----------


def _split_arguments(arguments: str) -> tuple[str, ...]:
    """Slice a tool-call arguments JSON string into 2-3 streamed deltas."""
    third: Final = max(1, len(arguments) // 3)
    return tuple(
        slice_
        for slice_ in (arguments[:third], arguments[third : 2 * third], arguments[2 * third :])
        if slice_
    )


def _openai_message(scenario: Scenario) -> Mapping[str, object]:
    tool_call: Final = scenario.output.tool_call
    return _jobj_opt(
        ("role", "assistant"),
        ("content", None if tool_call is not None else scenario.output.text),
        (
            (
                "tool_calls",
                (
                    _jobj(
                        ("id", f"call_{scenario.scenario_id}"),
                        ("type", "function"),
                        (
                            "function",
                            _jobj(("name", tool_call.name), ("arguments", tool_call.arguments)),
                        ),
                    ),
                ),
            )
            if tool_call is not None
            else None
        ),
        (
            (
                "annotations",
                tuple(
                    _jobj(
                        ("type", "url_citation"),
                        (
                            "url_citation",
                            _jobj(
                                ("url", "https://scripted.example/source"),
                                ("title", "scripted source"),
                                ("start_index", 0),
                                ("end_index", 1),
                            ),
                        ),
                    )
                    for _ in range(scenario.usage.web_search_calls)
                ),
            )
            if scenario.usage.web_search_calls
            else None
        ),
    )


def _openai_chat_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    return _jobj_opt(
        ("id", f"chatcmpl-{scenario.scenario_id}"),
        ("object", "chat.completion"),
        ("created", int(time.time())),
        ("model", scenario.output.response_model or requested_model),
        (
            "choices",
            (
                _jobj(
                    ("index", 0),
                    ("message", _openai_message(scenario)),
                    (
                        "finish_reason",
                        "tool_calls"
                        if scenario.output.tool_call is not None
                        else scenario.output.finish_reason,
                    ),
                ),
            ),
        ),
        ("usage", _openai_usage(scenario.usage)),
        ("service_tier", scenario.service_tier) if scenario.service_tier is not None else None,
        ("cost", scenario.output.provider_cost) if scenario.output.provider_cost is not None else None,
    )


def _openai_chunk(
    scenario: Scenario,
    requested_model: str,
    choices: tuple[Mapping[str, object], ...] = (),
    usage: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    return _jobj_opt(
        ("id", f"chatcmpl-{scenario.scenario_id}"),
        ("object", "chat.completion.chunk"),
        ("created", int(time.time())),
        ("model", scenario.output.response_model or requested_model),
        ("choices", choices),
        ("usage", usage),
    )


def _openai_chat_sse(scenario: Scenario, requested_model: str) -> bytes:
    tool_call: Final = scenario.output.tool_call
    delta: Final = _jobj_opt(
        ("role", "assistant"),
        ("content", scenario.output.text),
        (
            ("annotations", _openai_message(scenario)["annotations"])
            if scenario.usage.web_search_calls
            else None
        ),
    )
    body_deltas: Final[tuple[Mapping[str, object], ...]] = (
        (
            _jobj(
                ("role", "assistant"),
                (
                    "tool_calls",
                    (
                        _jobj(
                            ("index", 0),
                            ("id", f"call_{scenario.scenario_id}"),
                            ("type", "function"),
                            (
                                "function",
                                _jobj(("name", tool_call.name), ("arguments", "")),
                            ),
                        ),
                    ),
                ),
            ),
            *(
                _jobj(
                    (
                        "tool_calls",
                        (
                            _jobj(
                                ("index", 0),
                                ("function", _jobj(("arguments", arguments_slice))),
                            ),
                        ),
                    )
                )
                for arguments_slice in _split_arguments(tool_call.arguments)
            ),
        )
        if tool_call is not None
        else (delta,)
    )
    return _sse(
        (
            (
                None,
                _openai_chunk(
                    scenario,
                    requested_model,
                    choices=(_jobj(("index", 0), ("delta", _jobj(("role", "assistant"))), ("finish_reason", None)),),
                ),
            ),
            *(
                (
                    None,
                    _openai_chunk(
                        scenario,
                        requested_model,
                        choices=(_jobj(("index", 0), ("delta", body_delta), ("finish_reason", None)),),
                    ),
                )
                for body_delta in body_deltas
            ),
            (
                None,
                _openai_chunk(
                    scenario,
                    requested_model,
                    choices=(
                        _jobj(
                            ("index", 0),
                            ("delta", _jobj()),
                            (
                                "finish_reason",
                                "tool_calls"
                                if tool_call is not None
                                else scenario.output.finish_reason,
                            ),
                        ),
                    ),
                ),
            ),
            *(
                ((None, _openai_chunk(scenario, requested_model, usage=_openai_usage(scenario.usage))),)
                if scenario.stream_usage == "final_chunk"
                else ()
            ),
            (None, "[DONE]"),
        )
    )


def _anthropic_content(scenario: Scenario) -> tuple[Mapping[str, object], ...]:
    tool_call: Final = scenario.output.tool_call
    if tool_call is not None:
        return (
            _jobj(
                ("type", "tool_use"),
                ("id", f"toolu_{scenario.scenario_id}"),
                ("name", tool_call.name),
                ("input", json.loads(tool_call.arguments)),
            ),
        )
    return (_jobj(("type", "text"), ("text", scenario.output.text)),)


def _anthropic_stop_reason(scenario: Scenario) -> str:
    if scenario.output.tool_call is not None:
        return "tool_use"
    return "end_turn" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason


def _anthropic_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    return _jobj(
        ("id", f"msg_{scenario.scenario_id}"),
        ("type", "message"),
        ("role", "assistant"),
        ("model", scenario.output.response_model or requested_model),
        ("content", _anthropic_content(scenario)),
        ("stop_reason", _anthropic_stop_reason(scenario)),
        ("usage", _anthropic_usage(scenario)),
    )


def _anthropic_sse(scenario: Scenario, requested_model: str) -> bytes:
    emit_usage: Final = scenario.stream_usage == "final_chunk"
    input_usage: Final = _jobj(
        *(
            (key, value)
            for key, value in _anthropic_usage(scenario).items()
            if key != "output_tokens"
        )
    )
    message_start: Final = _jobj(
        ("type", "message_start"),
        (
            "message",
            _jobj_opt(
                ("id", f"msg_{scenario.scenario_id}"),
                ("type", "message"),
                ("role", "assistant"),
                ("model", scenario.output.response_model or requested_model),
                ("content", ()),
                ("stop_reason", None),
                ("usage", input_usage) if emit_usage else None,
            ),
        ),
    )
    message_delta: Final = _jobj_opt(
        ("type", "message_delta"),
        (
            "delta",
            _jobj(("stop_reason", _anthropic_stop_reason(scenario))),
        ),
        (
            ("usage", _jobj(("output_tokens", scenario.usage.output_tokens)))
            if emit_usage
            else None
        ),
    )
    return _sse(
        (
            ("message_start", message_start),
            (
                "content_block_start",
                _jobj(
                    ("type", "content_block_start"),
                    ("index", 0),
                    (
                        "content_block",
                        _jobj(
                            ("type", "tool_use"),
                            ("id", f"toolu_{scenario.scenario_id}"),
                            ("name", scenario.output.tool_call.name),
                            ("input", _jobj()),
                        )
                        if scenario.output.tool_call is not None
                        else _jobj(("type", "text"), ("text", "")),
                    ),
                ),
            ),
            *(
                tuple(
                    (
                        "content_block_delta",
                        _jobj(
                            ("type", "content_block_delta"),
                            ("index", 0),
                            (
                                "delta",
                                _jobj(("type", "input_json_delta"), ("partial_json", arguments_slice)),
                            ),
                        ),
                    )
                    for arguments_slice in _split_arguments(scenario.output.tool_call.arguments)
                )
                if scenario.output.tool_call is not None
                else (
                    (
                        "content_block_delta",
                        _jobj(
                            ("type", "content_block_delta"),
                            ("index", 0),
                            ("delta", _jobj(("type", "text_delta"), ("text", scenario.output.text))),
                        ),
                    ),
                )
            ),
            ("content_block_stop", _jobj(("type", "content_block_stop"), ("index", 0))),
            ("message_delta", message_delta),
            ("message_stop", _jobj(("type", "message_stop"))),
        )
    )


def _gemini_prompt_blocked_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    return _jobj(
        (
            "promptFeedback",
            _jobj(
                ("blockReason", "SAFETY"),
                (
                    "safetyRatings",
                    (
                        _jobj(
                            ("category", "HARM_CATEGORY_HARASSMENT"),
                            ("probability", "HIGH"),
                            ("blocked", True),
                        ),
                    ),
                ),
            ),
        ),
        ("usageMetadata", _gemini_usage(scenario)),
        ("modelVersion", scenario.output.response_model or requested_model),
    )


def _gemini_parts(scenario: Scenario) -> tuple[Mapping[str, object], ...]:
    tool_call: Final = scenario.output.tool_call
    if tool_call is not None:
        return (
            _jobj(
                (
                    "functionCall",
                    _jobj(
                        ("name", tool_call.name),
                        ("args", json.loads(tool_call.arguments)),
                    ),
                )
            ),
        )
    return (_jobj(("text", scenario.output.text)),)


def _gemini_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    if scenario.output.terminal == "prompt_blocked":
        return _gemini_prompt_blocked_body(scenario, requested_model)
    return _jobj(
        (
            "candidates",
            (
                _jobj_opt(
                    (
                        "content",
                        _jobj(
                            ("parts", _gemini_parts(scenario)),
                            ("role", "model"),
                        ),
                    ),
                    (
                        "finishReason",
                        "STOP" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason.upper(),
                    ),
                    ("index", 0),
                    (
                        ("groundingMetadata", _gemini_grounding_metadata(scenario))
                        if _gemini_grounding_metadata(scenario) is not None
                        else None
                    ),
                ),
            ),
        ),
        ("usageMetadata", _gemini_usage(scenario)),
        ("modelVersion", scenario.output.response_model or requested_model),
    )


def _gemini_sse(scenario: Scenario, requested_model: str) -> bytes:
    emit_usage: Final = scenario.stream_usage == "final_chunk"
    first: Final = _jobj(
        *((key, value) for key, value in _gemini_body(scenario, requested_model).items() if key != "usageMetadata")
    )
    return _sse(
        (
            (None, first),
            *(
                (
                    (
                        None,
                        _jobj(
                            ("candidates", ()),
                            ("usageMetadata", _gemini_usage(scenario)),
                            ("modelVersion", scenario.output.response_model or requested_model),
                        ),
                    ),
                )
                if emit_usage
                else ()
            ),
        )
    )


def _responses_output(scenario: Scenario) -> tuple[Mapping[str, object], ...]:
    tool_call: Final = scenario.output.tool_call
    return (
        *(
            (
                _jobj(("type", "scripted_future_item"), ("id", f"fut_{scenario.scenario_id}"), ("status", "completed")),
            )
            if scenario.output.terminal == "unvalidated"
            else ()
        ),
        *(
            _jobj(("type", "web_search_call"), ("id", f"ws_{i}"), ("status", "completed"))
            for i in range(scenario.usage.web_search_calls)
        ),
        *(
            _jobj(
                ("type", "file_search_call"),
                ("id", f"fs_{i}"),
                ("status", "completed"),
                ("queries", (f"query {i}",)),
                ("results", ()),
            )
            for i in range(scenario.usage.file_search_calls)
        ),
        _jobj(
            ("type", "function_call"),
            ("id", f"fc_{scenario.scenario_id}"),
            ("call_id", f"call_{scenario.scenario_id}"),
            ("name", tool_call.name),
            ("arguments", tool_call.arguments),
            ("status", "completed"),
        )
        if tool_call is not None
        else _jobj(
            ("type", "message"),
            ("id", f"msg_{scenario.scenario_id}"),
            ("status", "completed"),
            ("role", "assistant"),
            (
                "content",
                (
                    _jobj(
                        ("type", "output_text"),
                        ("text", scenario.output.text),
                        ("annotations", ()),
                    ),
                ),
            ),
        ),
    )


def _responses_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    incomplete: Final = scenario.output.terminal == "incomplete"
    return _jobj_opt(
        ("id", f"resp_{scenario.scenario_id}"),
        ("object", "response"),
        (
            "created_at",
            "not-a-number" if scenario.output.terminal == "unvalidated" else int(time.time()),
        ),
        ("status", "incomplete" if incomplete else "completed"),
        (
            ("incomplete_details", _jobj(("reason", "max_output_tokens")))
            if incomplete
            else None
        ),
        ("model", scenario.output.response_model or requested_model),
        ("output", _responses_output(scenario)),
        ("usage", _responses_usage(scenario.usage)),
    )


def _responses_sse(scenario: Scenario, requested_model: str) -> bytes:
    tool_call: Final = scenario.output.tool_call
    terminal: Final = (
        _jobj(*((key, value) for key, value in _responses_body(scenario, requested_model).items() if key != "usage"))
        if scenario.stream_usage == "absent"
        else _responses_body(scenario, requested_model)
    )
    created: Final = _jobj(
        *((key, value) for key, value in terminal.items() if key not in ("status", "usage")),
        ("status", "in_progress"),
        ("usage", None),
    )
    terminal_event: Final = (
        "response.incomplete" if scenario.output.terminal == "incomplete" else "response.completed"
    )
    output_index: Final = (
        scenario.usage.web_search_calls
        + scenario.usage.file_search_calls
        + (1 if scenario.output.terminal == "unvalidated" else 0)
    )
    file_search_events: Final[tuple[tuple[str, Mapping[str, object]], ...]] = tuple(
        event
        for i in range(scenario.usage.file_search_calls)
        for event in (
            (
                "response.output_item.added",
                _jobj(
                    ("type", "response.output_item.added"),
                    ("output_index", i),
                    (
                        "item",
                        _jobj(
                            ("type", "file_search_call"),
                            ("id", f"fs_{i}"),
                            ("status", "in_progress"),
                            ("queries", ()),
                        ),
                    ),
                ),
            ),
            (
                "response.output_item.done",
                _jobj(
                    ("type", "response.output_item.done"),
                    ("output_index", i),
                    (
                        "item",
                        _jobj(
                            ("type", "file_search_call"),
                            ("id", f"fs_{i}"),
                            ("status", "completed"),
                            ("queries", (f"query {i}",)),
                            ("results", ()),
                        ),
                    ),
                ),
            ),
        )
    )
    call_events: Final[tuple[tuple[str, Mapping[str, object]], ...]] = (
        (
            (
                "response.output_item.added",
                _jobj(
                    ("type", "response.output_item.added"),
                    ("output_index", output_index),
                    (
                        "item",
                        _jobj(
                            ("type", "function_call"),
                            ("id", f"fc_{scenario.scenario_id}"),
                            ("call_id", f"call_{scenario.scenario_id}"),
                            ("name", tool_call.name),
                            ("arguments", ""),
                            ("status", "in_progress"),
                        ),
                    ),
                ),
            ),
            *(
                (
                    "response.function_call_arguments.delta",
                    _jobj(
                        ("type", "response.function_call_arguments.delta"),
                        ("item_id", f"fc_{scenario.scenario_id}"),
                        ("output_index", output_index),
                        ("delta", arguments_slice),
                    ),
                )
                for arguments_slice in _split_arguments(tool_call.arguments)
            ),
            (
                "response.function_call_arguments.done",
                _jobj(
                    ("type", "response.function_call_arguments.done"),
                    ("item_id", f"fc_{scenario.scenario_id}"),
                    ("output_index", output_index),
                    ("arguments", tool_call.arguments),
                ),
            ),
        )
        if tool_call is not None
        else (
            (
                "response.output_text.delta",
                _jobj(
                    ("type", "response.output_text.delta"),
                    ("item_id", f"msg_{scenario.scenario_id}"),
                    ("output_index", output_index),
                    ("content_index", 0),
                    ("delta", scenario.output.text),
                ),
            ),
        )
    )
    middle_events: Final[tuple[tuple[str, Mapping[str, object]], ...]] = (
        *file_search_events,
        *call_events,
    )
    return _sse(
        (
            ("response.created", _jobj(("type", "response.created"), ("response", created))),
            *middle_events,
            (terminal_event, _jobj(("type", terminal_event), ("response", terminal))),
        )
    )


def _bedrock_usage(u: ScriptedUsage) -> Mapping[str, object]:
    # Converse reports uncached input in inputTokens and rides cache reads and
    # writes on top-level fields; totalTokens covers every input kind + output.
    cache_writes: Final = u.cache_write_5m_tokens + u.cache_write_1h_tokens
    return _jobj_opt(
        ("inputTokens", u.fresh_input_tokens),
        ("outputTokens", u.output_tokens),
        (
            "totalTokens",
            u.fresh_input_tokens + u.cache_read_tokens + cache_writes + u.output_tokens,
        ),
        ("cacheReadInputTokens", u.cache_read_tokens) if u.cache_read_tokens else None,
        ("cacheWriteInputTokens", cache_writes) if cache_writes else None,
        (
            (
                "cacheDetails",
                tuple(
                    _jobj(("inputTokens", count), ("ttl", ttl))
                    for count, ttl in (
                        (u.cache_write_5m_tokens, "5m"),
                        (u.cache_write_1h_tokens, "1h"),
                    )
                    if count
                ),
            )
            if cache_writes
            else None
        ),
    )


def _bedrock_stop_reason(scenario: Scenario) -> str:
    if scenario.output.tool_call is not None:
        return "tool_use"
    return "end_turn" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason


def _bedrock_content(scenario: Scenario) -> tuple[Mapping[str, object], ...]:
    tool_call: Final = scenario.output.tool_call
    if tool_call is not None:
        return (
            _jobj(
                (
                    "toolUse",
                    _jobj(
                        ("toolUseId", f"tooluse_{scenario.scenario_id}"),
                        ("name", tool_call.name),
                        ("input", json.loads(tool_call.arguments)),
                    ),
                ),
            ),
        )
    return (_jobj(("text", scenario.output.text)),)


def _bedrock_body(scenario: Scenario) -> Mapping[str, object]:
    return _jobj_opt(
        (
            "output",
            _jobj(
                (
                    "message",
                    _jobj(
                        ("role", "assistant"),
                        ("content", _bedrock_content(scenario)),
                    ),
                ),
            ),
        ),
        ("stopReason", _bedrock_stop_reason(scenario)),
        ("usage", _bedrock_usage(scenario.usage)),
        ("metrics", _jobj(("latencyMs", 42))),
        (
            ("serviceTier", _jobj(("type", scenario.service_tier)))
            if scenario.service_tier
            else None
        ),
    )


def _aws_str_header(name: str, value: str) -> bytes:
    """One eventstream header: 1-byte name len + name + type-7 marker + value."""
    name_b: Final = name.encode()
    value_b: Final = value.encode()
    return (
        struct.pack("!B", len(name_b))
        + name_b
        + struct.pack("!B", 7)
        + struct.pack("!H", len(value_b))
        + value_b
    )


def _aws_event_frame(event_type: str, payload: Mapping[str, object]) -> bytes:
    """One application/vnd.amazon.eventstream frame: prelude + prelude CRC32 +
    headers + JSON payload + message CRC32, matching botocore EventStreamBuffer."""
    payload_bytes: Final = json.dumps(payload, default=dict, separators=(",", ":")).encode()
    headers_bytes: Final = (
        _aws_str_header(":event-type", event_type)
        + _aws_str_header(":content-type", "application/json")
        + _aws_str_header(":message-type", "event")
    )
    total_length: Final = 12 + len(headers_bytes) + len(payload_bytes) + 4
    prelude: Final = struct.pack("!II", total_length, len(headers_bytes))
    prelude_crc: Final = struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message: Final = prelude + prelude_crc + headers_bytes + payload_bytes
    return message + struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)


def _bedrock_eventstream(scenario: Scenario) -> bytes:
    tool_call: Final = scenario.output.tool_call
    block_start: Final[tuple[bytes, ...]] = (
        (
            _aws_event_frame(
                "contentBlockStart",
                _jobj(
                    (
                        "start",
                        _jobj(
                            (
                                "toolUse",
                                _jobj(
                                    ("toolUseId", f"tooluse_{scenario.scenario_id}"),
                                    ("name", tool_call.name),
                                ),
                            ),
                        ),
                    ),
                    ("contentBlockIndex", 0),
                ),
            ),
        )
        if tool_call is not None
        else ()
    )
    deltas: Final[tuple[bytes, ...]] = (
        tuple(
            _aws_event_frame(
                "contentBlockDelta",
                _jobj(
                    ("delta", _jobj(("toolUse", _jobj(("input", arguments_slice))))),
                    ("contentBlockIndex", 0),
                ),
            )
            for arguments_slice in _split_arguments(tool_call.arguments)
        )
        if tool_call is not None
        else (
            _aws_event_frame(
                "contentBlockDelta",
                _jobj(
                    ("delta", _jobj(("text", scenario.output.text))),
                    ("contentBlockIndex", 0),
                ),
            ),
        )
    )
    return b"".join(
        (
            _aws_event_frame("messageStart", _jobj(("role", "assistant"))),
            *block_start,
            *deltas,
            _aws_event_frame("contentBlockStop", _jobj(("contentBlockIndex", 0))),
            _aws_event_frame("messageStop", _jobj(("stopReason", _bedrock_stop_reason(scenario)))),
            *(
                (
                    _aws_event_frame(
                        "metadata",
                        _jobj_opt(
                            ("usage", _bedrock_usage(scenario.usage)),
                            ("metrics", _jobj(("latencyMs", 42))),
                            (
                                ("serviceTier", _jobj(("type", scenario.service_tier)))
                                if scenario.service_tier
                                else None
                            ),
                        ),
                    ),
                )
                if scenario.stream_usage == "final_chunk"
                else ()
            ),
        )
    )


def _render(
    scenario: Scenario, *, stream: bool, requested_model: str, path_tail: str
) -> RenderedResponse:
    # Azure bridges gpt-5.4+ chat requests carrying function tools onto the
    # Responses API, which lands on the same mount at openai/responses.
    if scenario.wire == "azure_chat" and path_tail.endswith("openai/responses"):
        if stream:
            return RenderedResponse(
                200, "text/event-stream", _responses_sse(scenario, requested_model)
            )
        return RenderedResponse(
            200, "application/json", _json_bytes(_responses_body(scenario, requested_model))
        )
    if scenario.wire == "bedrock_converse":
        if stream:
            return RenderedResponse(
                200, "application/vnd.amazon.eventstream", _bedrock_eventstream(scenario)
            )
        return RenderedResponse(200, "application/json", _json_bytes(_bedrock_body(scenario)))
    if scenario.wire == "vertex_generate":
        if stream:
            return RenderedResponse(200, "text/event-stream", _gemini_sse(scenario, requested_model))
        return RenderedResponse(200, "application/json", _json_bytes(_gemini_body(scenario, requested_model)))
    if scenario.wire == "anthropic_messages":
        if stream:
            return RenderedResponse(200, "text/event-stream", _anthropic_sse(scenario, requested_model))
        return RenderedResponse(200, "application/json", _json_bytes(_anthropic_body(scenario, requested_model)))
    if scenario.wire == "gemini_generate":
        if stream:
            return RenderedResponse(200, "text/event-stream", _gemini_sse(scenario, requested_model))
        return RenderedResponse(200, "application/json", _json_bytes(_gemini_body(scenario, requested_model)))
    if scenario.wire == "openai_responses":
        if stream:
            return RenderedResponse(200, "text/event-stream", _responses_sse(scenario, requested_model))
        return RenderedResponse(200, "application/json", _json_bytes(_responses_body(scenario, requested_model)))
    # openai_chat, together_chat, fireworks_chat and azure_chat share the
    # OpenAI chat shape.
    if stream:
        return RenderedResponse(200, "text/event-stream", _openai_chat_sse(scenario, requested_model))
    return RenderedResponse(200, "application/json", _json_bytes(_openai_chat_body(scenario, requested_model)))


# ---------- registry + request routing ----------


class _ScenarioStore:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._scenarios: dict[str, Scenario] = {}  # mutable-ok: server state, guarded by _lock

    def put(self, scenario: Scenario) -> None:
        with self._lock:
            self._scenarios[scenario.scenario_id] = scenario

    def drop(self, scenario_id: str) -> bool:
        with self._lock:
            return self._scenarios.pop(scenario_id, None) is not None

    def get(self, scenario_id: str) -> Scenario | None:
        with self._lock:
            return self._scenarios.get(scenario_id)


_REQUEST_BODY: Final = TypeAdapter(dict[str, object])


def _request_body(body: bytes) -> Mapping[str, object]:
    try:
        return _REQUEST_BODY.validate_json(body)
    except ValueError:
        return MappingProxyType({})


def _request_wants_stream(mount_endpoint: str | None, path_tail: str, body: bytes) -> bool:
    if mount_endpoint == "streamGenerateContent" or ":streamGenerateContent" in path_tail:
        return True
    if path_tail.endswith("converse-stream"):
        return True
    if not body:
        return False
    return _request_body(body).get("stream") is True


def _request_model(body: bytes, path_tail: str, scenario: Scenario) -> str:
    model: Final = _request_body(body).get("model")
    if isinstance(model, str):
        return model
    # Bedrock Converse names the model in the path: model/<modelId>/converse[-stream].
    if path_tail.startswith("model/"):
        path_model: Final = path_tail.split("/", 2)[1] if path_tail.count("/") >= 2 else ""
        if path_model:
            return unquote(path_model)
    # Vertex names it in the URL too, but the mount segment swallowed it when
    # the api_base carried a path; fall back to the scenario's declared model.
    return scenario.model


def handle_request(store: _ScenarioStore, method: str, raw_path: str, body: bytes) -> RenderedResponse:
    path: Final = urlsplit(raw_path).path
    segments: Final = tuple(segment for segment in path.split("/") if segment)
    if method == "GET" and segments == ("health",):
        return RenderedResponse(200, "application/json", _json_bytes(_jobj(("status", "ok"))))
    if method == "GET" and segments == ("_cost_map",):
        return RenderedResponse(
            200,
            "application/json",
            (Path(__file__).resolve().parents[1] / "cost_calculation" / "cost_map.json").read_bytes(),
        )
    if segments and segments[0] == "_oauth":
        if method == "POST" and segments == ("_oauth", "token"):
            return RenderedResponse(
                200,
                "application/json",
                _json_bytes(
                    _jobj(
                        ("access_token", "scripted-token"),
                        ("token_type", "Bearer"),
                        ("expires_in", 3600),
                    )
                ),
            )
        return RenderedResponse(
            404, "application/json", _json_bytes(_jobj(("error", "unknown control route")))
        )
    if segments and segments[0] == "_scenarios":
        if method == "POST" and len(segments) == 1:
            try:
                scenario: Final = Scenario.model_validate_json(body)
            except ValidationError as exc:
                return RenderedResponse(
                    400, "application/json", _json_bytes(_jobj(("error", str(exc))))
                )
            store.put(scenario)
            return RenderedResponse(
                200, "application/json", _json_bytes(_jobj(("scenario_id", scenario.scenario_id)))
            )
        if method == "DELETE" and len(segments) == 2:
            deleted: Final = store.drop(segments[1])
            return RenderedResponse(
                200 if deleted else 404,
                "application/json",
                _json_bytes(_jobj(("deleted", deleted))),
            )
        return RenderedResponse(
            404, "application/json", _json_bytes(_jobj(("error", "unknown control route")))
        )
    if len(segments) < 2 or method != "POST":
        return RenderedResponse(
            404, "application/json", _json_bytes(_jobj(("error", f"no route for {method} {path}")))
        )
    scenario_id: Final = segments[0]
    # Vertex builds {api_base}:{endpoint}, so the mount segment can carry a
    # :generateContent / :streamGenerateContent suffix.
    mount_segment: Final = segments[1]
    mount, mount_endpoint = (
        mount_segment.split(":", 1)
        if ":" in mount_segment
        else (mount_segment, None)
    )
    found: Final = store.get(scenario_id)
    if found is None:
        return RenderedResponse(
            404, "application/json", _json_bytes(_jobj(("error", f"unknown scenario {scenario_id}")))
        )
    if found.mount != mount:
        return RenderedResponse(
            400,
            "application/json",
            _json_bytes(
                _jobj(("error", f"scenario {scenario_id} is wire {found.wire}, not mount {mount}"))
            ),
        )
    tail: Final = "/".join(segments[2:])
    return _render(
        found,
        stream=_request_wants_stream(mount_endpoint, tail, body),
        requested_model=_request_model(body, tail, found),
        path_tail=tail,
    )


class _ScriptedHandler(BaseHTTPRequestHandler):
    store: Final[_ScenarioStore] = _ScenarioStore()

    def _dispatch(self, method: str) -> None:
        length: Final = int(self.headers.get("content-length") or 0)
        body: Final = self.rfile.read(length) if length else b""
        rendered: Final = handle_request(self.store, method, self.path, body)
        self.send_response(rendered.status_code)
        self.send_header("content-type", rendered.content_type)
        self.send_header("content-length", str(len(rendered.body)))
        self.end_headers()
        self.wfile.write(rendered.body)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")



DEFAULT_PORT: Final = 8191


def serve(port: int = DEFAULT_PORT, bind_host: str = "127.0.0.1") -> None:
    server: Final = ThreadingHTTPServer((bind_host, port), _ScriptedHandler)
    sys.stderr.write(f"scripted-provider listening on http://{bind_host}:{port}\n")
    server.serve_forever()


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8191)
    serve(port=cast(int, parser.parse_args().port))
