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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

Wire = Literal[
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate",
    "together_chat",
    "fireworks_chat",
]

_WIRE_MOUNTS: Final[dict[str, str]] = {
    "openai_chat": "openai",
    "openai_responses": "openai",
    "anthropic_messages": "anthropic",
    "gemini_generate": "gemini",
    "together_chat": "together",
    "fireworks_chat": "fireworks",
}

StreamUsage = Literal["final_chunk", "absent"]
ServiceTier = Literal["flex", "priority"]


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
        return _WIRE_MOUNTS[self.wire]


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


def _json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _sse(events: tuple[tuple[str | None, dict[str, object] | str], ...]) -> bytes:
    frames: list[str] = []
    for event_name, data in events:
        head = f"event: {event_name}\n" if event_name is not None else ""
        payload = data if isinstance(data, str) else json.dumps(data)
        frames.append(f"{head}data: {payload}\n\n")
    return "".join(frames).encode("utf-8")


# ---------- per-wire usage shapes ----------


def _openai_usage(u: ScriptedUsage) -> dict[str, object]:
    prompt_tokens = (
        u.fresh_input_tokens
        + u.cache_read_tokens
        + u.cache_write_5m_tokens
        + u.cache_write_1h_tokens
        + u.audio_input_tokens
    )
    completion_tokens = u.output_tokens + u.reasoning_tokens + u.audio_output_tokens
    prompt_details: dict[str, object] = {}
    if u.cache_read_tokens:
        prompt_details["cached_tokens"] = u.cache_read_tokens
    if u.cache_write_5m_tokens or u.cache_write_1h_tokens:
        prompt_details["cache_write_tokens"] = u.cache_write_5m_tokens + u.cache_write_1h_tokens
        prompt_details["cache_creation_token_details"] = {
            "ephemeral_5m_input_tokens": u.cache_write_5m_tokens,
            "ephemeral_1h_input_tokens": u.cache_write_1h_tokens,
        }
    if u.audio_input_tokens:
        prompt_details["audio_tokens"] = u.audio_input_tokens
    completion_details: dict[str, object] = {}
    if u.reasoning_tokens:
        completion_details["reasoning_tokens"] = u.reasoning_tokens
    if u.audio_output_tokens:
        completion_details["audio_tokens"] = u.audio_output_tokens
    usage: dict[str, object] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if prompt_details:
        usage["prompt_tokens_details"] = prompt_details
    if completion_details:
        usage["completion_tokens_details"] = completion_details
    return usage


def _anthropic_usage(u: ScriptedUsage) -> dict[str, object]:
    # Anthropic reports uncached-only input_tokens; cache reads and writes ride
    # top-level fields, with the 5m/1h write split under cache_creation.
    usage: dict[str, object] = {
        "input_tokens": u.fresh_input_tokens,
        "output_tokens": u.output_tokens,
    }
    if u.cache_read_tokens:
        usage["cache_read_input_tokens"] = u.cache_read_tokens
    if u.cache_write_5m_tokens or u.cache_write_1h_tokens:
        usage["cache_creation_input_tokens"] = u.cache_write_5m_tokens + u.cache_write_1h_tokens
        usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": u.cache_write_5m_tokens,
            "ephemeral_1h_input_tokens": u.cache_write_1h_tokens,
        }
    if u.web_search_calls:
        usage["server_tool_use"] = {"web_search_requests": u.web_search_calls}
    return usage


def _gemini_usage(u: ScriptedUsage) -> dict[str, object]:
    # promptTokenCount carries the cached count inside it; TEXT modality is the
    # cached-inclusive text count so litellm's implicit-caching subtraction lands
    # on the fresh figure. candidatesTokenCount includes reasoning + audio.
    prompt_tokens = u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens
    candidates = u.output_tokens + u.reasoning_tokens + u.audio_output_tokens
    usage: dict[str, object] = {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": candidates,
        "totalTokenCount": prompt_tokens + candidates,
    }
    if u.cache_read_tokens:
        usage["cachedContentTokenCount"] = u.cache_read_tokens
    if u.reasoning_tokens:
        usage["thoughtsTokenCount"] = u.reasoning_tokens
    prompt_details = [{"modality": "TEXT", "tokenCount": u.fresh_input_tokens + u.cache_read_tokens}]
    if u.audio_input_tokens:
        prompt_details.append({"modality": "AUDIO", "tokenCount": u.audio_input_tokens})
    usage["promptTokensDetails"] = prompt_details
    if u.audio_output_tokens:
        usage["candidatesTokensDetails"] = [
            {"modality": "TEXT", "tokenCount": u.output_tokens + u.reasoning_tokens},
            {"modality": "AUDIO", "tokenCount": u.audio_output_tokens},
        ]
    return usage


def _responses_usage(u: ScriptedUsage) -> dict[str, object]:
    input_tokens = u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens
    output_tokens = u.output_tokens + u.reasoning_tokens + u.audio_output_tokens
    usage: dict[str, object] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    input_details: dict[str, object] = {}
    if u.cache_read_tokens:
        input_details["cached_tokens"] = u.cache_read_tokens
    if input_details:
        usage["input_tokens_details"] = input_details
    if u.reasoning_tokens:
        usage["output_tokens_details"] = {"reasoning_tokens": u.reasoning_tokens}
    return usage


# ---------- per-wire responses ----------


def _openai_message(scenario: Scenario) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": scenario.output.text}
    if scenario.usage.web_search_calls:
        message["annotations"] = [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://scripted.example/source",
                    "title": "scripted source",
                    "start_index": 0,
                    "end_index": 1,
                },
            }
            for _ in range(scenario.usage.web_search_calls)
        ]
    return message


def _openai_chat_body(scenario: Scenario, requested_model: str) -> dict[str, object]:
    body: dict[str, object] = {
        "id": f"chatcmpl-{scenario.scenario_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": scenario.output.response_model or requested_model,
        "choices": [
            {
                "index": 0,
                "message": _openai_message(scenario),
                "finish_reason": scenario.output.finish_reason,
            }
        ],
        "usage": _openai_usage(scenario.usage),
    }
    if scenario.service_tier is not None:
        body["service_tier"] = scenario.service_tier
    if scenario.output.provider_cost is not None:
        body["cost"] = scenario.output.provider_cost
    return body


def _openai_chunk(scenario: Scenario, requested_model: str, **kw: object) -> dict[str, object]:
    chunk: dict[str, object] = {
        "id": f"chatcmpl-{scenario.scenario_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": scenario.output.response_model or requested_model,
    }
    chunk.update(kw)
    return chunk


def _openai_chat_sse(scenario: Scenario, requested_model: str) -> bytes:
    _EMPTY_DELTA: Final[dict[str, object]] = {}
    delta: dict[str, object] = {"role": "assistant", "content": scenario.output.text}
    if scenario.usage.web_search_calls:
        delta["annotations"] = _openai_message(scenario)["annotations"]
    events: list[tuple[str | None, dict[str, object] | str]] = [
        (
            None,
            _openai_chunk(
                scenario,
                requested_model,
                choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            ),
        ),
        (
            None,
            _openai_chunk(
                scenario,
                requested_model,
                choices=[{"index": 0, "delta": delta, "finish_reason": None}],
            ),
        ),
        (
            None,
            _openai_chunk(
                scenario,
                requested_model,
                choices=[
                    {
                        "index": 0,
                        "delta": _EMPTY_DELTA,
                        "finish_reason": scenario.output.finish_reason,
                    }
                ],
            ),
        ),
    ]
    if scenario.stream_usage == "final_chunk":
        events.append(
            (None, _openai_chunk(scenario, requested_model, choices=(), usage=_openai_usage(scenario.usage)))
        )
    events.append((None, "[DONE]"))
    return _sse(tuple(events))


def _anthropic_body(scenario: Scenario, requested_model: str) -> dict[str, object]:
    return {
        "id": f"msg_{scenario.scenario_id}",
        "type": "message",
        "role": "assistant",
        "model": scenario.output.response_model or requested_model,
        "content": [{"type": "text", "text": scenario.output.text}],
        "stop_reason": "end_turn" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason,
        "usage": _anthropic_usage(scenario.usage),
    }


def _anthropic_sse(scenario: Scenario, requested_model: str) -> bytes:
    emit_usage = scenario.stream_usage == "final_chunk"
    input_usage = {k: v for k, v in _anthropic_usage(scenario.usage).items() if k != "output_tokens"}
    message_start: dict[str, object] = {
        "type": "message_start",
        "message": {
            "id": f"msg_{scenario.scenario_id}",
            "type": "message",
            "role": "assistant",
            "model": scenario.output.response_model or requested_model,
            "content": [],
            "stop_reason": None,
            **({"usage": input_usage} if emit_usage else {}),
        },
    }
    message_delta: dict[str, object] = {
        "type": "message_delta",
        "delta": {
            "stop_reason": "end_turn" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason
        },
        **({"usage": {"output_tokens": scenario.usage.output_tokens}} if emit_usage else {}),
    }
    return _sse(
        (
            ("message_start", message_start),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": scenario.output.text},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", message_delta),
            ("message_stop", {"type": "message_stop"}),
        )
    )


def _gemini_body(scenario: Scenario, requested_model: str) -> dict[str, object]:
    candidate: dict[str, object] = {
        "content": {"parts": [{"text": scenario.output.text}], "role": "model"},
        "finishReason": "STOP" if scenario.output.finish_reason == "stop" else scenario.output.finish_reason.upper(),
        "index": 0,
    }
    if scenario.usage.web_search_calls:
        candidate["groundingMetadata"] = {
            "webSearchQueries": [f"query {i}" for i in range(scenario.usage.web_search_calls)]
        }
    return {
        "candidates": [candidate],
        "usageMetadata": _gemini_usage(scenario.usage),
        "modelVersion": scenario.output.response_model or requested_model,
    }


def _gemini_sse(scenario: Scenario, requested_model: str) -> bytes:
    first = _gemini_body(scenario, requested_model)
    if scenario.stream_usage == "absent":
        first = {k: v for k, v in first.items() if k != "usageMetadata"}
    events: list[tuple[str | None, dict[str, object] | str]] = [(None, first)]
    if scenario.stream_usage == "final_chunk":
        events.append(
            (
                None,
                {
                    "candidates": [],
                    "usageMetadata": _gemini_usage(scenario.usage),
                    "modelVersion": scenario.output.response_model or requested_model,
                },
            )
        )
    return _sse(tuple(events))


def _responses_body(scenario: Scenario, requested_model: str) -> dict[str, object]:
    output: list[dict[str, object]] = [
        {"type": "web_search_call", "id": f"ws_{i}", "status": "completed"}
        for i in range(scenario.usage.web_search_calls)
    ]
    output.append(
        {
            "type": "message",
            "id": f"msg_{scenario.scenario_id}",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": scenario.output.text,
                    "annotations": [],
                }
            ],
        }
    )
    return {
        "id": f"resp_{scenario.scenario_id}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": scenario.output.response_model or requested_model,
        "output": output,
        "usage": _responses_usage(scenario.usage),
    }


def _responses_sse(scenario: Scenario, requested_model: str) -> bytes:
    completed = _responses_body(scenario, requested_model)
    if scenario.stream_usage == "absent":
        completed = {k: v for k, v in completed.items() if k != "usage"}
    created = {**completed, "status": "in_progress", "usage": None}
    return _sse(
        (
            ("response.created", {"type": "response.created", "response": created}),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": f"msg_{scenario.scenario_id}",
                    "output_index": scenario.usage.web_search_calls,
                    "content_index": 0,
                    "delta": scenario.output.text,
                },
            ),
            ("response.completed", {"type": "response.completed", "response": completed}),
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


def _request_body(body: bytes) -> dict[str, object]:
    try:
        return _REQUEST_BODY.validate_json(body)
    except ValueError:
        return {}


def _request_wants_stream(path_tail: str, body: bytes) -> bool:
    if ":streamGenerateContent" in path_tail:
        return True
    if not body:
        return False
    return _request_body(body).get("stream") is True


def _request_model(body: bytes) -> str:
    model = _request_body(body).get("model")
    return model if isinstance(model, str) else "unknown"


def handle_request(store: _ScenarioStore, method: str, raw_path: str, body: bytes) -> RenderedResponse:
    path = urlsplit(raw_path).path
    segments = [segment for segment in path.split("/") if segment]
    if method == "GET" and segments == ["health"]:
        return RenderedResponse(200, "application/json", _json_bytes({"status": "ok"}))
    if segments and segments[0] == "_scenarios":
        if method == "POST" and len(segments) == 1:
            try:
                scenario = Scenario.model_validate_json(body)
            except ValidationError as exc:
                return RenderedResponse(400, "application/json", _json_bytes({"error": str(exc)}))
            store.put(scenario)
            return RenderedResponse(200, "application/json", _json_bytes({"scenario_id": scenario.scenario_id}))
        if method == "DELETE" and len(segments) == 2:
            deleted = store.drop(segments[1])
            return RenderedResponse(
                200 if deleted else 404, "application/json", _json_bytes({"deleted": deleted})
            )
        return RenderedResponse(404, "application/json", _json_bytes({"error": "unknown control route"}))
    if len(segments) < 2 or method != "POST":
        return RenderedResponse(404, "application/json", _json_bytes({"error": f"no route for {method} {path}"}))
    scenario_id, mount = segments[0], segments[1]
    scenario = store.get(scenario_id)
    if scenario is None:
        return RenderedResponse(404, "application/json", _json_bytes({"error": f"unknown scenario {scenario_id}"}))
    if scenario.mount != mount:
        return RenderedResponse(
            400,
            "application/json",
            _json_bytes({"error": f"scenario {scenario_id} is wire {scenario.wire}, not mount {mount}"}),
        )
    tail = "/".join(segments[2:])
    return _render(scenario, stream=_request_wants_stream(tail, body), requested_model=_request_model(body))


class _ScriptedHandler(BaseHTTPRequestHandler):
    store: Final[_ScenarioStore] = _ScenarioStore()

    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        rendered = handle_request(self.store, method, self.path, body)
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
    server = ThreadingHTTPServer((bind_host, port), _ScriptedHandler)
    sys.stderr.write(f"scripted-provider listening on http://{bind_host}:{port}\n")
    server.serve_forever()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    serve(port=port_arg)
