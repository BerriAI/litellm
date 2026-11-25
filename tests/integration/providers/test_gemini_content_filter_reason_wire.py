import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm

_BACKEND: Final = "gemini-3.8-flash"
_OPENAI_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_ANSWER: Final = "Waves fold silver light into the shore."
_PARTIAL: Final = "Waves fold "
_OPENAI_ANSWER: Final = "Tides keep their own time."
_OPENAI_PROMPT: Final = "<scenario openai-plain>"
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_STREAM_SECONDS: Final = 60.0
_JSON_LIST: Final = TypeAdapter(list[JsonValue])

Provider: TypeAlias = Literal["gemini", "vertex_ai"]
_Chunks: TypeAlias = tuple[Mapping[str, JsonValue], ...]
_PROVIDERS: Final[tuple[Provider, ...]] = ("gemini", "vertex_ai")


@dataclass(frozen=True, slots=True)
class _Upstream:
    tag: str
    outcome: Mapping[str, JsonValue]
    usage: Mapping[str, int]
    frames: tuple[Mapping[str, JsonValue], ...]
    text: str
    finish_reason: str
    reason_fields: Mapping[str, str]
    note: str | None = None

    @property
    def prompt(self) -> str:
        return f"<scenario {self.tag}>"

    def body(self) -> bytes:
        return json.dumps({**self.outcome, "usageMetadata": dict(self.usage)}).encode()

    def events(self) -> tuple[bytes, ...]:
        finished: Final = {**self.frames[-1], "usageMetadata": dict(self.usage)}
        return tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in (*self.frames[:-1], finished))

    def expected_usage(self) -> Mapping[str, int]:
        return {
            "prompt_tokens": self.usage["promptTokenCount"],
            "completion_tokens": self.usage.get("candidatesTokenCount", 0),
            "total_tokens": self.usage["totalTokenCount"],
        }


def _text_frame(text: str) -> Mapping[str, JsonValue]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}]}


def _noted(key: str, note: str | None) -> Mapping[str, JsonValue]:
    return {} if note is None else {key: note}


def _with_note_tag(tag: str, note: str | None) -> str:
    return tag if note is None else f"{tag}-with-note"


def _flagged(reason: str, note: str | None = None) -> _Upstream:
    candidate: Final[Mapping[str, JsonValue]] = {"finishReason": reason, "index": 0, **_noted("finishMessage", note)}
    outcome: Final[Mapping[str, JsonValue]] = {"candidates": [dict(candidate)]}
    return _Upstream(
        tag=_with_note_tag(f"flagged-{reason}", note),
        outcome=outcome,
        usage={"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16},
        frames=(_text_frame(_PARTIAL), outcome),
        text=_PARTIAL,
        finish_reason="content_filter",
        reason_fields={"native_finish_reason": reason},
        note=note,
    )


def _blocked(reason: str, note: str | None = None) -> _Upstream:
    feedback: Final[Mapping[str, JsonValue]] = {"blockReason": reason, **_noted("blockReasonMessage", note)}
    outcome: Final[Mapping[str, JsonValue]] = {"promptFeedback": dict(feedback)}
    return _Upstream(
        tag=_with_note_tag(f"blocked-{reason}", note),
        outcome=outcome,
        usage={"promptTokenCount": 9, "totalTokenCount": 9},
        frames=(outcome,),
        text="",
        finish_reason="content_filter",
        reason_fields={"block_reason": reason},
        note=note,
    )


def _finished_after_text(native: str, finish_reason: str, note: str | None = None) -> _Upstream:
    finish: Final[Mapping[str, JsonValue]] = {"finishReason": native, "index": 0, **_noted("finishMessage", note)}
    return _Upstream(
        tag=_with_note_tag(f"answered-{native}", note),
        outcome={"candidates": [{"content": {"role": "model", "parts": [{"text": _ANSWER}]}, **finish}]},
        usage={"promptTokenCount": 7, "candidatesTokenCount": 8, "totalTokenCount": 15},
        frames=(
            _text_frame(_PARTIAL),
            _text_frame(_ANSWER.removeprefix(_PARTIAL)),
            {"candidates": [dict(finish)]},
        ),
        text=_ANSWER,
        finish_reason=finish_reason,
        reason_fields={"native_finish_reason": native},
        note=note,
    )


_FLAGGED_REASONS: Final = ("RECITATION", "SAFETY", "IMAGE_SAFETY")
_FLAGGED: Final = MappingProxyType({reason: _flagged(reason) for reason in _FLAGGED_REASONS})
_FLAGGED_WITH_NOTE: Final = _flagged("SAFETY", "The candidate was blocked for safety.")
_BLOCKED: Final = _blocked("SAFETY")
_BLOCKED_WITH_NOTE: Final = _blocked("PROHIBITED_CONTENT", "The prompt was blocked.")
_ANSWERED: Final = _finished_after_text("STOP", "stop")
_MALFORMED: Final = _finished_after_text("MALFORMED_FUNCTION_CALL", "stop")
_MALFORMED_WITH_NOTE: Final = _finished_after_text(
    "MALFORMED_FUNCTION_CALL", "stop", "Malformed function call: print(default_api.lookup(city='Paris'"
)
_UPSTREAMS: Final = (
    *_FLAGGED.values(),
    _FLAGGED_WITH_NOTE,
    _BLOCKED,
    _BLOCKED_WITH_NOTE,
    _ANSWERED,
    _MALFORMED,
    _MALFORMED_WITH_NOTE,
)
_FILTERED: Final = tuple(
    pytest.param(upstream, id=upstream.tag)
    for upstream in (*_FLAGGED.values(), _FLAGGED_WITH_NOTE, _BLOCKED, _BLOCKED_WITH_NOTE)
)
_WITH_NOTE: Final = tuple(
    pytest.param(upstream, id=upstream.tag)
    for upstream in (_FLAGGED_WITH_NOTE, _BLOCKED_WITH_NOTE, _MALFORMED_WITH_NOTE)
)
_FILTERED_ON_BRIDGES: Final = tuple(
    pytest.param(upstream, id=upstream.tag) for upstream in (_FLAGGED["RECITATION"], _BLOCKED)
)
_STREAMED: Final = tuple(
    pytest.param(upstream, id=upstream.tag)
    for upstream in (
        _FLAGGED["RECITATION"],
        _FLAGGED_WITH_NOTE,
        _BLOCKED,
        _BLOCKED_WITH_NOTE,
        _ANSWERED,
        _MALFORMED,
        _MALFORMED_WITH_NOTE,
    )
)


def _openai_events() -> tuple[bytes, ...]:
    def chunk(delta: Mapping[str, JsonValue], finish_reason: str | None) -> bytes:
        frame: Final = {
            "id": "chatcmpl-scripted",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _OPENAI_BACKEND,
            "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(frame)}\n\n".encode()

    return (
        chunk({"role": "assistant", "content": "Tides keep "}, None),
        chunk({"content": "their own time."}, None),
        chunk({}, "stop"),
        b"data: [DONE]\n\n",
    )


def _respond(request: Request) -> Reply:
    sent: Final = request.body.decode()
    if request.target.endswith("/chat/completions") and _OPENAI_PROMPT in sent:
        return Reply(content_type="text/event-stream", chunks=_openai_events())
    matched: Final = tuple(upstream for upstream in _UPSTREAMS if upstream.prompt in sent)
    if len(matched) != 1:
        return Reply(status=404, body=json.dumps({"error": f"no single scenario in {sent}"}).encode())
    if f"/models/{_BACKEND}:streamGenerateContent" in request.target:
        return Reply(content_type="text/event-stream", chunks=matched[0].events())
    if f"/models/{_BACKEND}:generateContent" in request.target:
        return Reply(body=matched[0].body())
    return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    models: Mapping[str, str]

    @property
    def model(self) -> str:
        return self.models["gemini"]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway) -> Iterator[_Rig]:
    with wire_server(_respond) as wire, session_gateway.scenario() as scenario:
        models: Final = MappingProxyType(
            {
                "gemini": scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY),
                "vertex_ai": scenario.model(
                    model=f"vertex_ai/{_BACKEND}",
                    api_base=f"{wire.url}{_MODEL_PATH}",
                    api_key=None,
                    vertex_project=_PROJECT,
                    vertex_location=_LOCATION,
                    vertex_credentials=service_account_json(_PROJECT, session_gateway.upstream_url.rstrip("/")),
                ),
                "openai": scenario.model(
                    model=f"openai/{_OPENAI_BACKEND}", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
                ),
            }
        )
        yield _Rig(session_gateway, models)


def _post(rig: _Rig, path: str, body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    response: Final = rig.gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def _chat(rig: _Rig, upstream: _Upstream, provider: Provider = "gemini") -> Mapping[str, JsonValue]:
    return _post(
        rig,
        "/v1/chat/completions",
        {"model": rig.models[provider], "messages": [{"role": "user", "content": upstream.prompt}], **_CACHE_BUST},
    )


def _stream(rig: _Rig, model: str, prompt: str) -> _Chunks:
    with rig.gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            **_CACHE_BUST,
        },
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
        timeout=_STREAM_SECONDS,
    ) as response:
        assert response.status_code == 200, response.read()
        events: Final = tuple(
            line.removeprefix("data: ") for line in response.iter_lines() if line.startswith("data: ")
        )
    assert events and events[-1] == "[DONE]", events
    return tuple(JSON_OBJECT.validate_json(event) for event in events[:-1])


def _streamed_choices(chunks: _Chunks) -> _Chunks:
    nested: Final = (_JSON_LIST.validate_python(chunk.get("choices", [])) for chunk in chunks)
    return tuple(JSON_OBJECT.validate_python(choice) for choice in chain.from_iterable(nested))


def _finish_choice(chunks: _Chunks) -> Mapping[str, JsonValue]:
    finished: Final = tuple(choice for choice in _streamed_choices(chunks) if choice.get("finish_reason"))
    assert len(finished) == 1, chunks
    return finished[0]


def _streamed_text(chunks: _Chunks) -> str:
    deltas: Final = (JSON_OBJECT.validate_python(choice.get("delta") or {}) for choice in _streamed_choices(chunks))
    return "".join(str(delta.get("content") or "") for delta in deltas)


def _assembled_choice_fields(chunks: _Chunks) -> object:
    assembled: Final = litellm.stream_chunk_builder(chunks=list(chunks))
    assert assembled is not None
    return getattr(assembled.choices[0], "provider_specific_fields", None)


def _first_choice(payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    choices: Final = payload["choices"]
    assert isinstance(choices, list) and len(choices) == 1, payload
    return JSON_OBJECT.validate_python(choices[0])


@pytest.mark.parametrize("reason", _FLAGGED_REASONS)
def test_regression_flagged_finish_reason_reaches_caller_as_native_finish_reason(rig: _Rig, reason: str) -> None:
    payload: Final = _chat(rig, _FLAGGED[reason])
    choice: Final = _first_choice(payload)
    assert choice.get("provider_specific_fields") == {"native_finish_reason": reason}, payload


def test_regression_blocked_prompt_reaches_caller_as_block_reason(rig: _Rig) -> None:
    payload: Final = _chat(rig, _BLOCKED)
    choice: Final = _first_choice(payload)
    assert choice.get("provider_specific_fields") == {"block_reason": "SAFETY"}, payload


@pytest.mark.parametrize("upstream", _FILTERED)
def test_guard_filtered_answer_keeps_content_filter_null_content_and_usage(rig: _Rig, upstream: _Upstream) -> None:
    payload: Final = _chat(rig, upstream)
    choice: Final = _first_choice(payload)
    assert choice["finish_reason"] == "content_filter", payload
    assert JSON_OBJECT.validate_python(choice["message"]).get("content") is None, payload
    usage: Final = JSON_OBJECT.validate_python(payload["usage"])
    assert {key: usage[key] for key in upstream.expected_usage()} == upstream.expected_usage(), payload


def test_guard_stop_answer_keeps_native_finish_reason_and_text(rig: _Rig) -> None:
    payload: Final = _chat(rig, _ANSWERED)
    choice: Final = _first_choice(payload)
    assert choice["finish_reason"] == "stop", payload
    assert JSON_OBJECT.validate_python(choice["message"])["content"] == _ANSWER, payload
    assert choice.get("provider_specific_fields") == {"native_finish_reason": "STOP"}, payload


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_non_streamed_choice_carries_only_google_reason_code(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    payload: Final = _chat(rig, upstream, provider)
    choice: Final = _first_choice(payload)
    assert choice["finish_reason"] == upstream.finish_reason, payload
    assert choice.get("provider_specific_fields") == upstream.reason_fields, payload


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_regression_streamed_finish_chunk_and_assembled_choice_carry_google_reason(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    chunks: Final = _stream(rig, rig.models[provider], upstream.prompt)
    assert _finish_choice(chunks).get("provider_specific_fields") == upstream.reason_fields, chunks
    assert _assembled_choice_fields(chunks) == upstream.reason_fields, chunks


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_regression_streamed_and_non_streamed_google_reason_agree(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    non_streamed: Final = _first_choice(_chat(rig, upstream, provider)).get("provider_specific_fields")
    streamed: Final = _finish_choice(_stream(rig, rig.models[provider], upstream.prompt)).get(
        "provider_specific_fields"
    )
    assert non_streamed is not None
    assert streamed == non_streamed


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_guard_streamed_text_finish_reason_and_usage_unchanged(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    chunks: Final = _stream(rig, rig.models[provider], upstream.prompt)
    assert _streamed_text(chunks) == upstream.text, chunks
    assert _finish_choice(chunks)["finish_reason"] == upstream.finish_reason, chunks
    usages: Final = tuple(JSON_OBJECT.validate_python(chunk["usage"]) for chunk in chunks if chunk.get("usage"))
    assert usages, chunks
    assert {key: usages[-1][key] for key in upstream.expected_usage()} == upstream.expected_usage(), chunks


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _WITH_NOTE)
def test_guard_google_free_text_note_never_reaches_the_non_streamed_response(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    assert upstream.note is not None
    payload: Final = _chat(rig, upstream, provider)
    assert upstream.note not in json.dumps(payload), payload


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _WITH_NOTE)
def test_guard_google_free_text_note_never_reaches_streamed_chunks_or_the_assembled_response(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    assert upstream.note is not None
    chunks: Final = _stream(rig, rig.models[provider], upstream.prompt)
    assert [chunk for chunk in chunks if upstream.note in json.dumps(chunk)] == [], chunks
    assembled: Final = litellm.stream_chunk_builder(chunks=list(chunks))
    assert assembled is not None
    assert upstream.note not in assembled.model_dump_json(), chunks


def test_guard_openai_stream_without_choice_fields_gains_none(rig: _Rig) -> None:
    chunks: Final = _stream(rig, rig.models["openai"], _OPENAI_PROMPT)
    assert _streamed_text(chunks) == _OPENAI_ANSWER, chunks
    assert _finish_choice(chunks)["finish_reason"] == "stop", chunks
    assert all(choice.get("provider_specific_fields") is None for choice in _streamed_choices(chunks)), chunks
    assert _assembled_choice_fields(chunks) is None, chunks


@pytest.mark.parametrize("upstream", _FILTERED_ON_BRIDGES)
def test_guard_messages_endpoint_still_reports_refusal(rig: _Rig, upstream: _Upstream) -> None:
    payload: Final = _post(
        rig,
        "/v1/messages",
        {"model": rig.model, "max_tokens": 64, "messages": [{"role": "user", "content": upstream.prompt}]},
    )
    assert payload["stop_reason"] == "refusal", payload


@pytest.mark.parametrize("upstream", _FILTERED_ON_BRIDGES)
def test_guard_responses_endpoint_still_reports_content_filter_incomplete(rig: _Rig, upstream: _Upstream) -> None:
    payload: Final = _post(rig, "/v1/responses", {"model": rig.model, "input": upstream.prompt})
    assert payload["status"] == "incomplete", payload
    assert JSON_OBJECT.validate_python(payload["incomplete_details"])["reason"] == "content_filter", payload
