import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_BACKEND: Final = "gemini-3.8-flash"
_API_KEY: Final = "synthetic-gemini-key"
_ANSWER: Final = "Waves fold silver light into the shore."


@dataclass(frozen=True, slots=True)
class _Upstream:
    tag: str
    outcome: Mapping[str, JsonValue]
    usage: Mapping[str, int]

    @property
    def prompt(self) -> str:
        return f"<scenario {self.tag}>"

    def body(self) -> bytes:
        return json.dumps({**self.outcome, "usageMetadata": dict(self.usage)}).encode()

    def expected_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.usage["promptTokenCount"],
            "completion_tokens": self.usage.get("candidatesTokenCount", 0),
            "total_tokens": self.usage["totalTokenCount"],
        }


def _flagged(reason: str) -> _Upstream:
    return _Upstream(
        tag=f"flagged-{reason}",
        outcome={"candidates": [{"finishReason": reason, "index": 0}]},
        usage={"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16},
    )


_FLAGGED_REASONS: Final = ("RECITATION", "SAFETY", "IMAGE_SAFETY")
_FLAGGED: Final = MappingProxyType({reason: _flagged(reason) for reason in _FLAGGED_REASONS})
_BLOCKED: Final = _Upstream(
    tag="blocked-SAFETY",
    outcome={"promptFeedback": {"blockReason": "SAFETY"}},
    usage={"promptTokenCount": 9, "totalTokenCount": 9},
)
_ANSWERED: Final = _Upstream(
    tag="answered-STOP",
    outcome={
        "candidates": [{"content": {"role": "model", "parts": [{"text": _ANSWER}]}, "finishReason": "STOP", "index": 0}]
    },
    usage={"promptTokenCount": 7, "candidatesTokenCount": 8, "totalTokenCount": 15},
)
_UPSTREAMS: Final = (*_FLAGGED.values(), _BLOCKED, _ANSWERED)
_FILTERED: Final = tuple(pytest.param(upstream, id=upstream.tag) for upstream in (*_FLAGGED.values(), _BLOCKED))
_FILTERED_ON_BRIDGES: Final = tuple(
    pytest.param(upstream, id=upstream.tag) for upstream in (_FLAGGED["RECITATION"], _BLOCKED)
)


def _respond(request: Request) -> Reply:
    if f"/models/{_BACKEND}:generateContent" not in request.target:
        return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())
    sent: Final = request.body.decode()
    matched: Final = tuple(upstream for upstream in _UPSTREAMS if upstream.prompt in sent)
    if len(matched) != 1:
        return Reply(status=404, body=json.dumps({"error": f"no single scenario in {sent}"}).encode())
    return Reply(body=matched[0].body())


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    model: str


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway) -> Iterator[_Rig]:
    with wire_server(_respond) as wire, session_gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        yield _Rig(session_gateway, model)


def _post(rig: _Rig, path: str, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    response: Final = rig.gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def _chat(rig: _Rig, upstream: _Upstream) -> dict[str, JsonValue]:
    return _post(
        rig,
        "/v1/chat/completions",
        {"model": rig.model, "messages": [{"role": "user", "content": upstream.prompt}]},
    )


def _first_choice(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
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
