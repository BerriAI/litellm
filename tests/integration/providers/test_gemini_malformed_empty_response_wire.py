import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gemini-4-carbon"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_ANSWER: Final = "Paris is sunny today."
_THOUGHT: Final = "The user wants the weather, so I should call the lookup tool."
_TOOL_NAME: Final = "lookup_weather"
_TOOL_ARGUMENTS: Final[Mapping[str, JsonValue]] = MappingProxyType({"city": "Paris"})
_TAG: Final = re.compile(r"<scenario ([a-z0-9-]+)>")
_SPEND_QUERY: Final = (
    'SELECT status, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
)
_SPEND_ROW_SECONDS: Final = 70
_LIST: Final = TypeAdapter(list[JsonValue])

Provider: TypeAlias = Literal["gemini", "vertex_ai"]
Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
_PROVIDERS: Final[tuple[Provider, ...]] = ("gemini", "vertex_ai")
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_PATHS: Final = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)
_THINKING: Final[Mapping[Endpoint, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "chat": {"reasoning_effort": "low"},
        "messages": {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        "responses": {"reasoning": {"effort": "low"}},
    }
)


@dataclass(frozen=True, slots=True)
class _Upstream:
    tag: str
    candidates: tuple[Mapping[str, JsonValue], ...]
    usage: Mapping[str, int]

    @property
    def native_finish_reason(self) -> str:
        return str(self.candidates[0]["finishReason"])

    def body(self) -> bytes:
        return json.dumps(
            {"candidates": [dict(c) for c in self.candidates], "usageMetadata": dict(self.usage)}
        ).encode()

    def prompt(self) -> str:
        return f"<scenario {self.tag}> nonce={uuid.uuid4().hex}"


def _candidate(reason: str, parts: tuple[Mapping[str, JsonValue], ...] | None, index: int = 0) -> dict[str, JsonValue]:
    content: Final[dict[str, JsonValue]] = (
        {"role": "model"} if parts is None else {"role": "model", "parts": [dict(part) for part in parts]}
    )
    return {"content": content, "finishReason": reason, "index": index}


def _single(
    tag: str, reason: str, parts: tuple[Mapping[str, JsonValue], ...] | None, usage: Mapping[str, int]
) -> _Upstream:
    return _Upstream(tag=tag, candidates=(_candidate(reason, parts),), usage=usage)


_BILLED_USAGE: Final[Mapping[str, int]] = MappingProxyType(
    {"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16}
)
_THINKING_USAGE: Final[Mapping[str, int]] = MappingProxyType(
    {"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 12, "totalTokenCount": 28}
)
_MALFORMED_REASONS: Final = ("MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE")
_EMPTY_SHAPES: Final[Mapping[str, tuple[Mapping[str, JsonValue], ...] | None]] = MappingProxyType(
    {"no-parts": None, "empty-text": ({"text": ""},)}
)


def _reason_slug(reason: str) -> str:
    return reason.lower().replace("_", "-")


_EMPTY_MALFORMED: Final = tuple(
    _single(f"empty-{_reason_slug(reason)}-{shape}", reason, parts, _BILLED_USAGE)
    for reason, (shape, parts) in product(_MALFORMED_REASONS, _EMPTY_SHAPES.items())
)
_THOUGHT_ONLY: Final = _single(
    "thought-only-malformed-function-call",
    "MALFORMED_FUNCTION_CALL",
    ({"text": _THOUGHT, "thought": True},),
    _THINKING_USAGE,
)
_MALFORMED_WITH_TEXT: Final = _single(
    "malformed-function-call-with-text", "MALFORMED_FUNCTION_CALL", ({"text": _ANSWER},), _BILLED_USAGE
)
_MALFORMED_WITH_CALL: Final = _single(
    "malformed-function-call-with-valid-call",
    "MALFORMED_FUNCTION_CALL",
    ({"functionCall": {"name": _TOOL_NAME, "args": dict(_TOOL_ARGUMENTS)}},),
    _BILLED_USAGE,
)
_STOP_EMPTY: Final = _single("stop-no-parts", "STOP", None, _BILLED_USAGE)
_ONE_GOOD_OF_TWO: Final = _Upstream(
    tag="two-candidates-one-malformed-empty",
    candidates=(
        _candidate("MALFORMED_FUNCTION_CALL", None, 0),
        _candidate("STOP", ({"text": _ANSWER},), 1),
    ),
    usage=_BILLED_USAGE,
)
_UPSTREAMS: Final = MappingProxyType(
    {
        upstream.tag: upstream
        for upstream in (
            *_EMPTY_MALFORMED,
            _THOUGHT_ONLY,
            _MALFORMED_WITH_TEXT,
            _MALFORMED_WITH_CALL,
            _STOP_EMPTY,
            _ONE_GOOD_OF_TWO,
        )
    }
)


def _respond(request: Request) -> Reply:
    found: Final = _TAG.search(request.body.decode())
    upstream: Final = _UPSTREAMS.get(found.group(1)) if found is not None else None
    if upstream is None or ":generateContent" not in request.target:
        return Reply(status=404, body=json.dumps({"error": f"no scenario for {request.target}"}).encode())
    return Reply(body=upstream.body())


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    models: Mapping[Provider, str]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway) -> Iterator[_Rig]:
    with wire_server(_respond) as wire, session_gateway.scenario() as scenario:
        models: Final[Mapping[Provider, str]] = MappingProxyType(
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
            }
        )
        yield _Rig(session_gateway, models)


def _body(endpoint: Endpoint, model: str, prompt: str, *, thinking: bool = False) -> dict[str, JsonValue]:
    extra: Final = dict(_THINKING[endpoint]) if thinking else {}
    match endpoint:
        case "chat":
            return {"model": model, "messages": [{"role": "user", "content": prompt}], "num_retries": 0, **extra}
        case "messages":
            return {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}], **extra}
        case "responses":
            return {"model": model, "input": prompt, **extra}


def _send(
    rig: _Rig,
    endpoint: Endpoint,
    provider: Provider,
    upstream: _Upstream,
    *,
    thinking: bool = False,
    **extra: JsonValue,
) -> httpx.Response:
    return rig.gateway.request(
        "POST",
        _PATHS[endpoint],
        {**_body(endpoint, rig.models[provider], upstream.prompt(), thinking=thinking), **extra},
    )


def _choices(response: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    return tuple(JSON_OBJECT.validate_python(choice) for choice in _LIST.validate_python(payload["choices"]))


def _message(choice: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_python(choice["message"])


def _assert_server_error_naming(response: httpx.Response, native_finish_reason: str) -> None:
    assert response.status_code == 500, response.text
    assert native_finish_reason in response.text, response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize(
    ("upstream", "endpoint"),
    (
        *(
            pytest.param(_EMPTY_MALFORMED[0], endpoint, id=f"{_EMPTY_MALFORMED[0].tag}-{endpoint}")
            for endpoint in _ENDPOINTS
        ),
        *(pytest.param(upstream, "chat", id=f"{upstream.tag}-chat") for upstream in _EMPTY_MALFORMED[1:]),
    ),
)
def test_regression_malformed_empty_candidate_is_a_server_error_naming_the_native_finish_reason(
    rig: _Rig, upstream: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    _assert_server_error_naming(_send(rig, endpoint, provider, upstream), upstream.native_finish_reason)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_regression_malformed_candidate_with_only_thoughts_is_a_server_error(
    rig: _Rig, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _send(rig, endpoint, provider, _THOUGHT_ONLY, thinking=True)
    _assert_server_error_naming(response, _THOUGHT_ONLY.native_finish_reason)


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_regression_malformed_empty_failure_is_logged_with_the_billed_completion_tokens(
    rig: _Rig, provider: Provider
) -> None:
    upstream: Final = _EMPTY_MALFORMED[0]
    response: Final = _send(rig, "chat", provider, upstream)
    _assert_server_error_naming(response, upstream.native_finish_reason)
    call_id: Final = response.headers["x-litellm-call-id"]
    rows: Final = eventually(
        lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=_SPEND_ROW_SECONDS
    )
    row: Final = rows[0]
    assert row["status"] == "failure", row
    assert row["completion_tokens"] == upstream.usage["candidatesTokenCount"], row
    assert float(str(row["spend"])) > 0, row


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_guard_malformed_function_call_with_text_keeps_the_text(rig: _Rig, provider: Provider) -> None:
    choices: Final = _choices(_send(rig, "chat", provider, _MALFORMED_WITH_TEXT))
    assert len(choices) == 1, choices
    assert _message(choices[0])["content"] == _ANSWER, choices


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_guard_malformed_function_call_with_a_valid_call_keeps_the_tool_call(rig: _Rig, provider: Provider) -> None:
    response: Final = _send(
        rig,
        "chat",
        provider,
        _MALFORMED_WITH_CALL,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
    )
    choices: Final = _choices(response)
    assert len(choices) == 1, choices
    calls: Final = _LIST.validate_python(_message(choices[0])["tool_calls"])
    assert len(calls) == 1, choices
    function: Final = JSON_OBJECT.validate_python(JSON_OBJECT.validate_python(calls[0])["function"])
    assert function["name"] == _TOOL_NAME, choices
    assert json.loads(str(function["arguments"])) == _TOOL_ARGUMENTS, choices


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_guard_stop_with_no_parts_stays_a_successful_empty_answer(rig: _Rig, provider: Provider) -> None:
    choices: Final = _choices(_send(rig, "chat", provider, _STOP_EMPTY))
    assert len(choices) == 1, choices
    assert choices[0]["finish_reason"] == "stop", choices


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_guard_one_malformed_empty_candidate_among_two_keeps_the_good_candidate(rig: _Rig, provider: Provider) -> None:
    choices: Final = _choices(_send(rig, "chat", provider, _ONE_GOOD_OF_TWO, n=2))
    assert _ANSWER in tuple(_message(choice).get("content") for choice in choices), choices
