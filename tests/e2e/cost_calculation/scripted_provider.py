"""Scripted provider sidecar for the cost-calculation e2e suite.

A standalone process (``python -m cost_calculation.scripted_provider``) that
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
- ``POST /<id>/<mount>/<provider path>`` provider wire; mount is one of
  ``openai``, ``anthropic``, ``gemini``, ``together``, ``fireworks`` and the
  remainder is whatever path the provider client appends (``chat/completions``,
  ``responses``, ``v1/messages``, ``models/<m>:generateContent`` ...)

A request carrying ``"stream": true`` (or the ``:streamGenerateContent`` Gemini
verb) gets an SSE answer; ``stream_usage`` on the Scenario decides whether the
final stream chunk carries usage or the provider reports none.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

Wire: TypeAlias = Literal[
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate",
    "together_chat",
    "fireworks_chat",
]

WIRE_MOUNTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai_chat": "openai",
        "openai_responses": "openai",
        "anthropic_messages": "anthropic",
        "gemini_generate": "gemini",
        "together_chat": "together",
        "fireworks_chat": "fireworks",
    }
)

StreamUsage: TypeAlias = Literal["final_chunk", "absent"]
ServiceTier: TypeAlias = Literal["flex", "priority"]


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
    web_search_calls: int = 0


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


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    wire: Wire
    usage: ScriptedUsage
    output: ScriptedOutput
    stream_usage: StreamUsage = "final_chunk"
    service_tier: ServiceTier | None = None

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
    prompt_tokens: Final = (
        u.fresh_input_tokens
        + u.cache_read_tokens
        + u.cache_write_5m_tokens
        + u.cache_write_1h_tokens
        + u.audio_input_tokens
    )
    completion_tokens: Final = u.output_tokens + u.reasoning_tokens + u.audio_output_tokens
    prompt_details: Final = _jobj_opt(
        ("cached_tokens", u.cache_read_tokens) if u.cache_read_tokens else None,
        (
            ("cache_write_tokens", u.cache_write_5m_tokens + u.cache_write_1h_tokens)
            if u.cache_write_5m_tokens or u.cache_write_1h_tokens
            else None
        ),
        (
            (
                "cache_creation_token_details",
                _jobj(
                    ("ephemeral_5m_input_tokens", u.cache_write_5m_tokens),
                    ("ephemeral_1h_input_tokens", u.cache_write_1h_tokens),
                ),
            )
            if u.cache_write_5m_tokens or u.cache_write_1h_tokens
            else None
        ),
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


def _anthropic_usage(u: ScriptedUsage) -> Mapping[str, object]:
    # Anthropic reports uncached-only input_tokens; cache reads and writes ride
    # top-level fields, with the 5m/1h write split under cache_creation.
    return _jobj_opt(
        ("input_tokens", u.fresh_input_tokens),
        ("output_tokens", u.output_tokens),
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


def _gemini_usage(u: ScriptedUsage) -> Mapping[str, object]:
    # promptTokenCount carries the cached count inside it; TEXT modality is the
    # cached-inclusive text count so litellm's implicit-caching subtraction lands
    # on the fresh figure. candidatesTokenCount includes reasoning + audio.
    prompt_tokens: Final = u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens
    candidates: Final = u.output_tokens + u.reasoning_tokens + u.audio_output_tokens
    return _jobj_opt(
        ("promptTokenCount", prompt_tokens),
        ("candidatesTokenCount", candidates),
        ("totalTokenCount", prompt_tokens + candidates),
        ("cachedContentTokenCount", u.cache_read_tokens) if u.cache_read_tokens else None,
        ("thoughtsTokenCount", u.reasoning_tokens) if u.reasoning_tokens else None,
        (
            "promptTokensDetails",
            (
                _jobj(("modality", "TEXT"), ("tokenCount", u.fresh_input_tokens + u.cache_read_tokens)),
                *(
                    (_jobj(("modality", "AUDIO"), ("tokenCount", u.audio_input_tokens)),)
                    if u.audio_input_tokens
                    else ()
                ),
            ),
        ),
        (
            (
                "candidatesTokensDetails",
                (
                    _jobj(("modality", "TEXT"), ("tokenCount", u.output_tokens + u.reasoning_tokens)),
                    _jobj(("modality", "AUDIO"), ("tokenCount", u.audio_output_tokens)),
                ),
            )
            if u.audio_output_tokens
            else None
        ),
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


def _openai_message(scenario: Scenario) -> Mapping[str, object]:
    return _jobj_opt(
        ("role", "assistant"),
        ("content", scenario.output.text),
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
                    ("finish_reason", scenario.output.finish_reason),
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
    delta: Final = _jobj_opt(
        ("role", "assistant"),
        ("content", scenario.output.text),
        (
            ("annotations", _openai_message(scenario)["annotations"])
            if scenario.usage.web_search_calls
            else None
        ),
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
            (
                None,
                _openai_chunk(
                    scenario,
                    requested_model,
                    choices=(_jobj(("index", 0), ("delta", delta), ("finish_reason", None)),),
                ),
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
                            ("finish_reason", scenario.output.finish_reason),
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


def _anthropic_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    return _jobj(
        ("id", f"msg_{scenario.scenario_id}"),
        ("type", "message"),
        ("role", "assistant"),
        ("model", scenario.output.response_model or requested_model),
        ("content", (_jobj(("type", "text"), ("text", scenario.output.text)),)),
        (
            "stop_reason",
            "end_turn" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason,
        ),
        ("usage", _anthropic_usage(scenario.usage)),
    )


def _anthropic_sse(scenario: Scenario, requested_model: str) -> bytes:
    emit_usage: Final = scenario.stream_usage == "final_chunk"
    input_usage: Final = _jobj(
        *(
            (key, value)
            for key, value in _anthropic_usage(scenario.usage).items()
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
            _jobj(
                (
                    "stop_reason",
                    "end_turn" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason,
                )
            ),
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
                    ("content_block", _jobj(("type", "text"), ("text", ""))),
                ),
            ),
            (
                "content_block_delta",
                _jobj(
                    ("type", "content_block_delta"),
                    ("index", 0),
                    ("delta", _jobj(("type", "text_delta"), ("text", scenario.output.text))),
                ),
            ),
            ("content_block_stop", _jobj(("type", "content_block_stop"), ("index", 0))),
            ("message_delta", message_delta),
            ("message_stop", _jobj(("type", "message_stop"))),
        )
    )


def _gemini_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    return _jobj(
        (
            "candidates",
            (
                _jobj_opt(
                    (
                        "content",
                        _jobj(
                            ("parts", (_jobj(("text", scenario.output.text)),)),
                            ("role", "model"),
                        ),
                    ),
                    (
                        "finishReason",
                        "STOP" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason.upper(),
                    ),
                    ("index", 0),
                    (
                        (
                            "groundingMetadata",
                            _jobj(
                                (
                                    "webSearchQueries",
                                    tuple(f"query {i}" for i in range(scenario.usage.web_search_calls)),
                                )
                            ),
                        )
                        if scenario.usage.web_search_calls
                        else None
                    ),
                ),
            ),
        ),
        ("usageMetadata", _gemini_usage(scenario.usage)),
        ("modelVersion", scenario.output.response_model or requested_model),
    )


def _gemini_sse(scenario: Scenario, requested_model: str) -> bytes:
    emit_usage: Final = scenario.stream_usage == "final_chunk"
    first: Final = (
        _jobj(*((key, value) for key, value in _gemini_body(scenario, requested_model).items() if key != "usageMetadata"))
        if scenario.stream_usage == "absent"
        else _gemini_body(scenario, requested_model)
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
                            ("usageMetadata", _gemini_usage(scenario.usage)),
                            ("modelVersion", scenario.output.response_model or requested_model),
                        ),
                    ),
                )
                if emit_usage
                else ()
            ),
        )
    )


def _responses_body(scenario: Scenario, requested_model: str) -> Mapping[str, object]:
    return _jobj(
        ("id", f"resp_{scenario.scenario_id}"),
        ("object", "response"),
        ("created_at", int(time.time())),
        ("status", "completed"),
        ("model", scenario.output.response_model or requested_model),
        (
            "output",
            (
                *(
                    _jobj(("type", "web_search_call"), ("id", f"ws_{i}"), ("status", "completed"))
                    for i in range(scenario.usage.web_search_calls)
                ),
                _jobj(
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
            ),
        ),
        ("usage", _responses_usage(scenario.usage)),
    )


def _responses_sse(scenario: Scenario, requested_model: str) -> bytes:
    completed: Final = (
        _jobj(*((key, value) for key, value in _responses_body(scenario, requested_model).items() if key != "usage"))
        if scenario.stream_usage == "absent"
        else _responses_body(scenario, requested_model)
    )
    created: Final = _jobj(
        *((key, value) for key, value in completed.items() if key not in ("status", "usage")),
        ("status", "in_progress"),
        ("usage", None),
    )
    return _sse(
        (
            ("response.created", _jobj(("type", "response.created"), ("response", created))),
            (
                "response.output_text.delta",
                _jobj(
                    ("type", "response.output_text.delta"),
                    ("item_id", f"msg_{scenario.scenario_id}"),
                    ("output_index", scenario.usage.web_search_calls),
                    ("content_index", 0),
                    ("delta", scenario.output.text),
                ),
            ),
            ("response.completed", _jobj(("type", "response.completed"), ("response", completed))),
        )
    )


def _render(scenario: Scenario, *, stream: bool, requested_model: str) -> RenderedResponse:
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
    # openai_chat, together_chat, fireworks_chat share the OpenAI chat shape.
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


def _request_wants_stream(path_tail: str, body: bytes) -> bool:
    if ":streamGenerateContent" in path_tail:
        return True
    if not body:
        return False
    return _request_body(body).get("stream") is True


def _request_model(body: bytes) -> str:
    model: Final = _request_body(body).get("model")
    return model if isinstance(model, str) else "unknown"


def handle_request(store: _ScenarioStore, method: str, raw_path: str, body: bytes) -> RenderedResponse:
    path: Final = urlsplit(raw_path).path
    segments: Final = tuple(segment for segment in path.split("/") if segment)
    if method == "GET" and segments == ("health",):
        return RenderedResponse(200, "application/json", _json_bytes(_jobj(("status", "ok"))))
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
    scenario_id, mount = segments[0], segments[1]
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
    return _render(found, stream=_request_wants_stream(tail, body), requested_model=_request_model(body))


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



DEFAULT_PORT: Final = 9100


def serve(port: int = DEFAULT_PORT, bind_host: str = "127.0.0.1") -> None:
    server: Final = ThreadingHTTPServer((bind_host, port), _ScriptedHandler)
    sys.stderr.write(f"scripted-provider listening on http://{bind_host}:{port}\n")
    server.serve_forever()


if __name__ == "__main__":
    port_arg: Final = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    serve(port=port_arg)
