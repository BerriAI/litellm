import json
import re
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import reduce
from hashlib import sha256
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import SseEvent, delta_text, event_type, event_types, parse_sse
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    held,
    list_value,
    object_value,
)
from integration._support.openai_wire import answering_model_discovery, posted_targets, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.prometheus_series import scrape
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing.test_payment_required_cooldown import (
    NO_DEPLOYMENTS,
    cooldown_config,
    cooled,
    router_settings,
)
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.constants import DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS
from litellm.types.router import RouterErrors

_BACKEND: Final = "gemini-3.8-flash"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_ANSWER: Final = "Waves fold silver light into the shore."
_PARTIAL: Final = "Waves fold "
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_STREAM_SECONDS: Final = 60.0
_JSON_LIST: Final = TypeAdapter(list[JsonValue])

Provider: TypeAlias = Literal["gemini", "vertex_ai"]
_Chunks: TypeAlias = tuple[Mapping[str, JsonValue], ...]
_PROVIDERS: Final[tuple[Provider, ...]] = ("gemini", "vertex_ai")


@dataclass(frozen=True, slots=True)
class _Script:
    tag: str
    native: str
    outcome: Mapping[str, JsonValue]
    usage: Mapping[str, int]
    frames: tuple[Mapping[str, JsonValue], ...]
    text: str
    note: str | None = None

    @property
    def prompt(self) -> str:
        return f"<scenario {self.tag}>"

    def body(self) -> bytes:
        return json.dumps({**self.outcome, "usageMetadata": dict(self.usage)}).encode()

    def events(self) -> tuple[bytes, ...]:
        finished: Final = {**self.frames[-1], "usageMetadata": dict(self.usage)}
        return tuple(f"data: {json.dumps(dict(frame))}\n\n".encode() for frame in (*self.frames[:-1], finished))


def _text_frame(text: str) -> Mapping[str, JsonValue]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}]}


def _with_note_tag(tag: str, note: str | None) -> str:
    return tag if note is None else f"{tag}-with-note"


def _finished_after_text(native: str, note: str | None = None) -> _Script:
    noted: Final[Mapping[str, JsonValue]] = {} if note is None else {"finishMessage": note}
    finish: Final[Mapping[str, JsonValue]] = {"finishReason": native, "index": 0, **noted}
    return _Script(
        tag=_with_note_tag(f"answered-{native}", note),
        native=native,
        outcome={"candidates": [{"content": {"role": "model", "parts": [{"text": _ANSWER}]}, **finish}]},
        usage={"promptTokenCount": 7, "candidatesTokenCount": 8, "totalTokenCount": 15},
        frames=(
            _text_frame(_PARTIAL),
            _text_frame(_ANSWER.removeprefix(_PARTIAL)),
            {"candidates": [dict(finish)]},
        ),
        text=_ANSWER,
        note=note,
    )


def _streamed_choices(chunks: _Chunks) -> _Chunks:
    nested: Final = (_JSON_LIST.validate_python(chunk.get("choices", [])) for chunk in chunks)
    return tuple(JSON_OBJECT.validate_python(choice) for choice in chain.from_iterable(nested))


def _streamed_text(chunks: _Chunks) -> str:
    deltas: Final = (JSON_OBJECT.validate_python(choice.get("delta") or {}) for choice in _streamed_choices(chunks))
    return "".join(str(delta.get("content") or "") for delta in deltas)


def _first_choice(payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    choices: Final = payload["choices"]
    assert isinstance(choices, list) and len(choices) == 1, payload
    return JSON_OBJECT.validate_python(choices[0])


Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
_Events: TypeAlias = tuple[SseEvent, ...]
_ToolCalls: TypeAlias = tuple[tuple[str, Mapping[str, JsonValue]], ...]
_BuildBody: TypeAlias = Callable[[Endpoint, str, str], Mapping[str, JsonValue]]
_AssertFailure: TypeAlias = Callable[[Endpoint, httpx.Response, _Script], None]

_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_ERROR_EVENT_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("messages", "responses")
_RESPONSES_API_RESCUED_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "responses")
_RESPONSES_STARTED_ONCE: Final[Mapping[Endpoint, int]] = MappingProxyType({"chat": 0, "messages": 0, "responses": 1})
_PATHS: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)
_ERROR_ENDING: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "error then done", "messages": "error", "responses": "response.failed"}
)
_NORMAL_ENDING: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "done", "messages": "message_stop", "responses": "response.completed"}
)
_STOP_FINISH: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "stop", "messages": "end_turn", "responses": "completed"}
)
_TOOL_FINISH: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "tool_calls", "messages": "tool_use", "responses": "completed"}
)
_NO_RETRY_NO_CACHE: Final[Mapping[str, JsonValue]] = MappingProxyType({**_CACHE_BUST, "num_retries": 0})
_TOOL_NAME: Final = "lookup_city"
_TOOL_ARGUMENTS: Final[Mapping[str, JsonValue]] = MappingProxyType({"city": "Paris"})
_TOOL_PARAMETERS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
)
_TOOL_CALL_CANDIDATE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "content": {"role": "model", "parts": [{"functionCall": {"name": _TOOL_NAME, "args": dict(_TOOL_ARGUMENTS)}}]},
        "index": 0,
    }
)
_SECOND_ANSWER: Final = "Gulls trace the tide line."
_HEALTHY_ANSWER: Final = "Harbor lights answer the tide."
_THOUGHT: Final = "Weighing which tool fits the request."
_OVERLOADED_MESSAGE: Final = "scripted upstream overload"
_PLAIN_PROMPT: Final = "<scenario no-tools-declared>"
_NOTE_MARKER: Final = "finish-message-marker-for-the-model-only"
_NOTE: Final = f"Malformed function call: {_NOTE_MARKER}"
_FAILED_TOOL_CALL_REASONS: Final = ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL")
_NORMAL_FINISH_REASONS: Final = ("STOP", "MALFORMED_RESPONSE", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE")
_INPUT_PRICE: Final = 0.001
_OUTPUT_PRICE: Final = 0.002
_ROW_SECONDS: Final = 70
_COOLDOWN_THRESHOLD: Final = DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS
_MAX_COOLDOWN_REQUESTS: Final = 60
_REQUESTS_AFTER_COOLDOWN: Final = 5
_ROW_SETTLE_SECONDS: Final = 3
_DEPLOYMENT_STATE: Final = "litellm_deployment_state"
_DEPLOYMENT_FAILURES: Final = "litellm_deployment_failure_responses_total"
_RESPONSES_API_BACKEND: Final = "gpt-5.3-codex"
_RESPONSES_API_USAGE: Final[Mapping[str, int]] = MappingProxyType(
    {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}
)
_PARTIAL_OUTAGE: Final = 1.0
_OWNED_PROXY: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)
_THOUGHT_FRAME: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"candidates": [{"content": {"role": "model", "parts": [{"text": _THOUGHT, "thought": True}]}, "index": 0}]}
)
_UPSTREAM_503_FRAME: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"error": {"code": 503, "message": _OVERLOADED_MESSAGE, "status": "UNAVAILABLE"}}
)
_UPSTREAM_500_FRAME: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"error": {"code": 500, "message": "scripted upstream failure", "status": "INTERNAL"}}
)
_COST_TOLERANCE: Final = 1e-9
_ROW_COLUMNS: Final = (
    "status, spend, prompt_tokens, completion_tokens, total_tokens, model_group, model_id, custom_llm_provider"
)
_GROUP_SPEND_ROWS: Final = f'SELECT {_ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE model_group IN (%s, %s)'
_SPEND_ROWS: Final = f'SELECT {_ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE model_group=%s'
_KEY_SPEND: Final = 'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s'
_HEALTHY_STATE: Final = 0.0
_DEPLOYMENT_SUCCESSES: Final = "litellm_deployment_success_responses_total"
_COSTLY_USAGE: Final[Mapping[str, int]] = MappingProxyType(
    {"promptTokenCount": 7, "candidatesTokenCount": 100_000, "totalTokenCount": 100_007}
)
_ZERO_USAGE: Final[Mapping[str, int]] = MappingProxyType(
    {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
)
_PROVIDER_BUDGET: Final = 1_000_000
_BUDGET_WINDOW: Final = "1d"
_OUTPUT_CAPS: Final[Mapping[Endpoint, Mapping[str, JsonValue]]] = MappingProxyType(
    {"chat": {"max_tokens": 16}, "messages": {"max_tokens": 16}, "responses": {"max_output_tokens": 16}}
)
_BUDGET_REFUSAL: Final = (422, "budget_exceeded")
_CROSSED_BUDGET: Final = RouterErrors.no_deployments_with_provider_budget_routing.value
_PERSON: Final = "John Smith"
_PHONE: Final = "+1 415-555-0134"
_PII_ENTITIES: Final[Mapping[str, str]] = MappingProxyType({_PERSON: "PERSON", _PHONE: "PHONE_NUMBER"})
_PII_OPENING: Final = f"Call {_PERSON} "
_PII_CLOSING: Final = f"at {_PHONE} today."
_MASKED_PII_TEXT: Final = "Call <PERSON> at <PHONE_NUMBER> today."
_PII_SCORE: Final = 0.85
_PRESIDIO_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages")


class _SpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_group: str | None
    model_id: str | None
    custom_llm_provider: str | None


class _KeySpend(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float


class _ProviderBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float


def _answer_candidate(text: str, index: int = 0) -> Mapping[str, JsonValue]:
    return {"content": {"role": "model", "parts": [{"text": text}]}, "index": index}


def _finish_candidate(native: str, index: int = 0) -> Mapping[str, JsonValue]:
    return {"finishReason": native, "index": index}


def _frame(*candidates: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {"candidates": [dict(candidate) for candidate in candidates]}


def _text_then_finish(native: str, note: str | None = None) -> _Script:
    return replace(_finished_after_text(native, note), tag=_with_note_tag(f"text-then-finish-{native}", note))


def _text_inside_final_chunk(native: str) -> _Script:
    final: Final = {**_answer_candidate(_ANSWER.removeprefix(_PARTIAL)), **_finish_candidate(native)}
    return replace(
        _finished_after_text(native),
        tag=f"text-inside-final-chunk-{native}",
        frames=(_text_frame(_PARTIAL), _frame(final)),
    )


def _whole_answer_in_one_chunk(native: str) -> _Script:
    answered: Final = _finished_after_text(native)
    return replace(answered, tag=f"whole-answer-in-one-chunk-{native}", frames=(answered.outcome,))


def _finish_without_text(native: str, note: str | None = None) -> _Script:
    answered: Final = _finished_after_text(native, note)
    return replace(
        answered,
        tag=_with_note_tag(f"no-text-{native}", note),
        outcome=answered.frames[-1],
        frames=answered.frames[-1:],
        text="",
    )


def _tool_call_then_finish(native: str) -> _Script:
    return replace(
        _finished_after_text(native),
        tag=f"tool-call-then-finish-{native}",
        outcome=_frame({**_TOOL_CALL_CANDIDATE, **_finish_candidate(native)}),
        frames=(_frame(_TOOL_CALL_CANDIDATE), _frame(_finish_candidate(native))),
        text="",
    )


def _tool_call_inside_final_chunk(native: str) -> _Script:
    return replace(
        _finished_after_text(native),
        tag=f"tool-call-inside-final-chunk-{native}",
        outcome=_frame({**_TOOL_CALL_CANDIDATE, **_finish_candidate(native)}),
        frames=(_frame({**_TOOL_CALL_CANDIDATE, **_finish_candidate(native)}),),
        text="",
    )


def _thought_then_finish(native: str) -> _Script:
    return replace(
        _finished_after_text(native),
        tag=f"thought-then-finish-{native}",
        frames=(_THOUGHT_FRAME, _frame(_finish_candidate(native))),
        text="",
    )


def _two_candidates(failing_index: int) -> _Script:
    finishes: Final = tuple(
        _finish_candidate("MALFORMED_FUNCTION_CALL" if index == failing_index else "STOP", index) for index in (0, 1)
    )
    return replace(
        _finished_after_text("STOP"),
        tag=f"two-candidates-index-{failing_index}-fails",
        outcome=_frame(
            {**_answer_candidate(_ANSWER, 0), **finishes[0]},
            {**_answer_candidate(_SECOND_ANSWER, 1), **finishes[1]},
        ),
        frames=(_frame(_answer_candidate(_ANSWER, 0), _answer_candidate(_SECOND_ANSWER, 1)), _frame(*finishes)),
    )


def _pii_then_finish(native: str) -> _Script:
    return replace(
        _finished_after_text(native),
        tag=f"pii-then-finish-{native}",
        outcome=_frame({**_answer_candidate(_PII_OPENING + _PII_CLOSING), **_finish_candidate(native)}),
        frames=(_text_frame(_PII_OPENING), _text_frame(_PII_CLOSING), _frame(_finish_candidate(native))),
        text=_MASKED_PII_TEXT,
    )


def _params(scripts: tuple[_Script, ...]) -> tuple[object, ...]:
    return tuple(pytest.param(script, id=script.tag) for script in scripts)


_TEXT_BEFORE_FAILURE: Final = (
    *map(_text_then_finish, _FAILED_TOOL_CALL_REASONS),
    *(_text_then_finish(native, _NOTE) for native in _FAILED_TOOL_CALL_REASONS),
    *map(_text_inside_final_chunk, _FAILED_TOOL_CALL_REASONS),
    *map(_whole_answer_in_one_chunk, _FAILED_TOOL_CALL_REASONS),
)
_THOUGHT_BEFORE_FAILURE: Final = tuple(map(_thought_then_finish, _FAILED_TOOL_CALL_REASONS))
_NO_TEXT_FAILURE: Final = (
    *map(_finish_without_text, _FAILED_TOOL_CALL_REASONS),
    *(_finish_without_text(native, _NOTE) for native in _FAILED_TOOL_CALL_REASONS),
)
_REAL_TOOL_CALLS: Final = (
    _tool_call_then_finish("MALFORMED_FUNCTION_CALL"),
    _tool_call_inside_final_chunk("MALFORMED_FUNCTION_CALL"),
)
_NORMAL_FINISHES: Final = tuple(map(_text_then_finish, _NORMAL_FINISH_REASONS))
_WITH_FINISH_MESSAGE: Final = tuple(
    script for script in (*_TEXT_BEFORE_FAILURE, *_NO_TEXT_FAILURE) if script.note is not None
)
_TWO_CANDIDATES: Final = (_two_candidates(0), _two_candidates(1))
_NON_STREAMED_FAILURES: Final = (
    *map(_text_then_finish, _FAILED_TOOL_CALL_REASONS),
    *(_text_then_finish(native, _NOTE) for native in _FAILED_TOOL_CALL_REASONS),
    *_NO_TEXT_FAILURE,
)
_FAILURE_WITH_AND_WITHOUT_TEXT: Final = (
    _text_then_finish("MALFORMED_FUNCTION_CALL"),
    _finish_without_text("MALFORMED_FUNCTION_CALL"),
)
_HEALTHY_SCRIPT: Final = replace(
    _finished_after_text("STOP"),
    tag="healthy-answer",
    usage={"promptTokenCount": 11, "candidatesTokenCount": 13, "totalTokenCount": 24},
    outcome=_frame({**_answer_candidate(_HEALTHY_ANSWER), **_finish_candidate("STOP")}),
    frames=(_text_frame(_HEALTHY_ANSWER), _frame(_finish_candidate("STOP"))),
    text=_HEALTHY_ANSWER,
)
_UPSTREAM_500_MID_STREAM: Final = replace(
    _finished_after_text("STOP"),
    tag="upstream-500-mid-stream",
    frames=(_text_frame(_PARTIAL), _UPSTREAM_500_FRAME),
    text=_PARTIAL,
)
_UPSTREAM_503_AFTER_THOUGHT: Final = replace(
    _finished_after_text("STOP"),
    tag="upstream-503-after-thought",
    frames=(_THOUGHT_FRAME, _UPSTREAM_503_FRAME),
    text="",
)
_COSTLY_FAILURE: Final = replace(
    _text_then_finish("MALFORMED_FUNCTION_CALL"),
    tag="costly-text-then-finish-MALFORMED_FUNCTION_CALL",
    usage=_COSTLY_USAGE,
)
_ZERO_COST_PROBE: Final = replace(_HEALTHY_SCRIPT, tag="zero-cost-probe", usage=_ZERO_USAGE)
_PII_BEFORE_FAILURE: Final = tuple(map(_pii_then_finish, _FAILED_TOOL_CALL_REASONS))
_SCRIPTS: Final = (
    *_TEXT_BEFORE_FAILURE,
    *_THOUGHT_BEFORE_FAILURE,
    _UPSTREAM_503_AFTER_THOUGHT,
    *_NO_TEXT_FAILURE,
    *_REAL_TOOL_CALLS,
    *_NORMAL_FINISHES,
    *_TWO_CANDIDATES,
    _COSTLY_FAILURE,
    _ZERO_COST_PROBE,
    *_PII_BEFORE_FAILURE,
)


def _scripted_reply(script: _Script, target: str) -> Reply:
    if f"/models/{_BACKEND}:streamGenerateContent" in target:
        return Reply(content_type="text/event-stream", chunks=script.events())
    if f"/models/{_BACKEND}:generateContent" in target:
        return Reply(body=script.body())
    return Reply(status=404, body=json.dumps({"error": f"unexpected target {target}"}).encode())


def _respond_with_script(request: Request) -> Reply:
    sent: Final = request.body.decode()
    matched: Final = tuple(script for script in _SCRIPTS if script.prompt in sent)
    if len(matched) != 1:
        return Reply(status=404, body=json.dumps({"error": f"no single script in {sent}"}).encode())
    return _scripted_reply(matched[0], request.target)


def _respond_with_healthy_answer(request: Request) -> Reply:
    return _scripted_reply(_HEALTHY_SCRIPT, request.target)


def _respond_with_upstream_500(request: Request) -> Reply:
    return _scripted_reply(_UPSTREAM_500_MID_STREAM, request.target)


def _respond_with_plain_500(_: Request) -> Reply:
    return Reply(status=500, body=json.dumps(dict(_UPSTREAM_500_FRAME)).encode())


def _respond_through_the_responses_api(request: Request) -> Reply:
    if request.target != "/v1/responses":
        return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())
    return responses_reply(f"resp-{uuid.uuid4().hex}", _RESPONSES_API_BACKEND, _HEALTHY_ANSWER, stream=True)


def _failing_only_requests_with_tools(script: _Script) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        declares_tools: Final = "tools" in JSON_OBJECT.validate_json(request.body)
        return _scripted_reply(script if declares_tools else _HEALTHY_SCRIPT, request.target)

    return respond


def _add_deployment(
    scenario: Scenario,
    provider: Provider,
    wire: Wire,
    *,
    group: str | None = None,
    identity: str | None = None,
    max_budget: float | None = None,
) -> str:
    model_info: Final[Mapping[str, JsonValue] | None] = None if identity is None else {"id": identity}
    limits: Final[Mapping[str, JsonValue]] = (
        {} if max_budget is None else {"max_budget": max_budget, "budget_duration": _BUDGET_WINDOW}
    )
    match provider:
        case "gemini":
            return scenario.model(
                name=group,
                model_info=model_info,
                model=f"gemini/{_BACKEND}",
                api_base=wire.url,
                api_key=_API_KEY,
                input_cost_per_token=_INPUT_PRICE,
                output_cost_per_token=_OUTPUT_PRICE,
                **limits,
            )
        case "vertex_ai":
            return scenario.model(
                name=group,
                model_info=model_info,
                model=f"vertex_ai/{_BACKEND}",
                api_base=f"{wire.url}{_MODEL_PATH}",
                api_key=None,
                vertex_project=_PROJECT,
                vertex_location=_LOCATION,
                vertex_credentials=service_account_json(_PROJECT, scenario.gateway.upstream_url.rstrip("/")),
                input_cost_per_token=_INPUT_PRICE,
                output_cost_per_token=_OUTPUT_PRICE,
                **limits,
            )


def _add_responses_api_deployment(scenario: Scenario, _: Provider, wire: Wire) -> str:
    return scenario.model(
        model=f"openai/{_RESPONSES_API_BACKEND}",
        api_base=f"{wire.url}/v1",
        api_key="synthetic-openai-key",
        input_cost_per_token=_INPUT_PRICE,
        output_cost_per_token=_OUTPUT_PRICE,
    )


@dataclass(frozen=True, slots=True)
class _FallbackPair:
    primary_wire: Wire
    fallback_wire: Wire
    primary: str
    fallback: str

    @property
    def groups(self) -> tuple[str, str]:
        return self.primary, self.fallback


@contextmanager
def _primary_with_fallback(
    gateway: Gateway,
    provider: Provider,
    respond_as_fallback: Callable[[Request], Reply] = _respond_with_healthy_answer,
    add_fallback: Callable[[Scenario, Provider, Wire], str] = _add_deployment,
) -> Generator[_FallbackPair]:
    with (
        wire_server(_respond_with_script) as primary_wire,
        wire_server(respond_as_fallback) as fallback_wire,
        gateway.scenario() as scenario,
    ):
        primary: Final = _add_deployment(scenario, provider, primary_wire)
        yield _FallbackPair(primary_wire, fallback_wire, primary, add_fallback(scenario, provider, fallback_wire))


@dataclass(frozen=True, slots=True)
class _ScriptedGemini:
    gateway: Gateway
    models: Mapping[Provider, str]
    wire: Wire


@pytest.fixture(scope="module")
def scripted_gemini(session_gateway: Gateway) -> Iterator[_ScriptedGemini]:
    with wire_server(_respond_with_script) as wire, session_gateway.scenario() as scenario:
        models: Final[Mapping[Provider, str]] = MappingProxyType(
            {provider: _add_deployment(scenario, provider, wire) for provider in _PROVIDERS}
        )
        yield _ScriptedGemini(session_gateway, models, wire)


def _tool_declaration(endpoint: Endpoint) -> JsonValue:
    parameters: Final = dict(_TOOL_PARAMETERS)
    match endpoint:
        case "chat":
            return {"type": "function", "function": {"name": _TOOL_NAME, "parameters": parameters}}
        case "messages":
            return {"name": _TOOL_NAME, "input_schema": parameters}
        case "responses":
            return {"type": "function", "name": _TOOL_NAME, "parameters": parameters}


def _stream_body(
    endpoint: Endpoint,
    model: str,
    prompt: str,
    *,
    fallback: str | None = None,
    candidate_count: int | None = None,
    declares_tools: bool = True,
) -> Mapping[str, JsonValue]:
    shared: Final[Mapping[str, JsonValue]] = {
        "model": model,
        "stream": True,
        **_NO_RETRY_NO_CACHE,
        **({} if fallback is None else {"fallbacks": [fallback]}),
        **({} if candidate_count is None else {"n": candidate_count}),
        **({"tools": [_tool_declaration(endpoint)]} if declares_tools else {}),
    }
    messages: Final[JsonValue] = [{"role": "user", "content": prompt}]
    match endpoint:
        case "chat":
            return {**shared, "messages": messages}
        case "messages":
            return {**shared, "max_tokens": 64, "messages": messages}
        case "responses":
            return {**shared, "input": prompt}


def _completion_body(
    endpoint: Endpoint,
    model: str,
    prompt: str,
    *,
    fallback: str | None = None,
    candidate_count: int | None = None,
    declares_tools: bool = True,
) -> Mapping[str, JsonValue]:
    streaming: Final = _stream_body(
        endpoint,
        model,
        prompt,
        fallback=fallback,
        candidate_count=candidate_count,
        declares_tools=declares_tools,
    )
    return {key: value for key, value in streaming.items() if key != "stream"}


def _send(
    gateway: Gateway, endpoint: Endpoint, body: Mapping[str, JsonValue], *, key: str | None = None
) -> httpx.Response:
    return gateway.client.post(
        _PATHS[endpoint],
        json=dict(body),
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}"},
        timeout=_STREAM_SECONDS,
    )


def _stream_script(
    scripted: _ScriptedGemini, endpoint: Endpoint, provider: Provider, script: _Script
) -> httpx.Response:
    return _send(scripted.gateway, endpoint, _stream_body(endpoint, scripted.models[provider], script.prompt))


def _payloads(events: _Events) -> _Chunks:
    return tuple(event.data for event in events)


def _chat_ending(body: str, events: _Events) -> str:
    finished_with_done: Final = body.rstrip().endswith("data: [DONE]")
    last_is_error: Final = bool(events) and "error" in events[-1].data
    match (last_is_error, finished_with_done):
        case (True, True):
            return "error then done"
        case (False, True):
            return "done"
        case _:
            return "cut off"


def _ending(endpoint: Endpoint, body: str, events: _Events) -> str:
    match endpoint:
        case "chat":
            return _chat_ending(body, events)
        case "messages" | "responses":
            return event_types(events)[-1] if events else "empty"


def _error_frames(endpoint: Endpoint, events: _Events) -> _Events:
    match endpoint:
        case "chat":
            return tuple(event for event in events if "error" in event.data)
        case "messages":
            return tuple(event for event in events if event_type(event) == "error")
        case "responses":
            return tuple(event for event in events if event_type(event) in ("error", "response.failed"))


def _finishes(endpoint: Endpoint, events: _Events) -> tuple[str, ...]:
    match endpoint:
        case "chat":
            choices: Final = _streamed_choices(_payloads(events))
            return tuple(str(choice["finish_reason"]) for choice in choices if choice.get("finish_reason"))
        case "messages":
            deltas: Final = (object_value(event.data["delta"]) for event in events if event.event == "message_delta")
            return tuple(str(delta.get("stop_reason")) for delta in deltas)
        case "responses":
            ended: Final = tuple(
                event for event in events if event_type(event) in ("response.completed", "response.incomplete")
            )
            return tuple(str(object_value(event.data["response"])["status"]) for event in ended)


def _delivered_text(endpoint: Endpoint, events: _Events) -> str:
    match endpoint:
        case "chat":
            return _streamed_text(_payloads(events))
        case "messages":
            return delta_text(events)
        case "responses":
            deltas: Final = (event for event in events if event_type(event) == "response.output_text.delta")
            return "".join(str(event.data["delta"]) for event in deltas)


def _delivered_reasoning(endpoint: Endpoint, events: _Events) -> str:
    match endpoint:
        case "chat":
            deltas: Final = (object_value(choice.get("delta") or {}) for choice in _streamed_choices(_payloads(events)))
            return "".join(str(delta.get("reasoning_content") or "") for delta in deltas)
        case "messages":
            blocks: Final = (
                object_value(event.data["delta"]) for event in events if event.event == "content_block_delta"
            )
            return "".join(str(delta.get("thinking", "")) for delta in blocks)
        case "responses":
            summaries: Final = (
                event for event in events if event_type(event) == "response.reasoning_summary_text.delta"
            )
            return "".join(str(event.data["delta"]) for event in summaries)


def _responses_started(events: _Events) -> int:
    return event_types(events).count("response.created")


def _chat_tool_calls(events: _Events) -> _ToolCalls:
    deltas: Final = (object_value(choice.get("delta") or {}) for choice in _streamed_choices(_payloads(events)))
    calls: Final = chain.from_iterable(_JSON_LIST.validate_python(delta.get("tool_calls") or []) for delta in deltas)
    functions: Final = tuple(object_value(object_value(call)["function"]) for call in calls)
    name: Final = "".join(str(function.get("name") or "") for function in functions)
    arguments: Final = "".join(str(function.get("arguments") or "") for function in functions)
    return ((name, JSON_OBJECT.validate_json(arguments)),)


def _messages_tool_calls(events: _Events) -> _ToolCalls:
    starts: Final = (event for event in events if event.event == "content_block_start")
    blocks: Final = tuple(object_value(event.data["content_block"]) for event in starts)
    name: Final = "".join(str(block["name"]) for block in blocks if block["type"] == "tool_use")
    deltas: Final = (object_value(event.data["delta"]) for event in events if event.event == "content_block_delta")
    arguments: Final = "".join(str(delta.get("partial_json", "")) for delta in deltas)
    return ((name, JSON_OBJECT.validate_json(arguments)),)


def _responses_tool_calls(events: _Events) -> _ToolCalls:
    completed: Final = tuple(event for event in events if event_type(event) == "response.completed")
    assert len(completed) == 1, events
    output: Final = tuple(map(object_value, list_value(object_value(completed[0].data["response"])["output"])))
    return tuple(
        (str(item["name"]), JSON_OBJECT.validate_json(str(item["arguments"])))
        for item in output
        if item["type"] == "function_call"
    )


def _delivered_tool_calls(endpoint: Endpoint, events: _Events) -> _ToolCalls:
    match endpoint:
        case "chat":
            return _chat_tool_calls(events)
        case "messages":
            return _messages_tool_calls(events)
        case "responses":
            return _responses_tool_calls(events)


def _terminal_error_message(endpoint: Endpoint, events: _Events) -> str:
    terminal: Final = events[-1].data
    match endpoint:
        case "chat" | "messages":
            return str(object_value(terminal["error"])["message"])
        case "responses":
            return str(object_value(object_value(terminal["response"])["error"])["message"])


def _assert_error_ending(endpoint: Endpoint, response: httpx.Response, script: _Script) -> None:
    events: Final = parse_sse(response.text)
    observed: Final = (
        response.status_code,
        _ending(endpoint, response.text, events),
        _finishes(endpoint, events),
        _delivered_text(endpoint, events),
    )
    assert observed == (200, _ERROR_ENDING[endpoint], (), script.text), response.text
    assert script.native in _terminal_error_message(endpoint, events), response.text
    assert _NOTE_MARKER not in response.text, response.text


def _assert_normal_ending(endpoint: Endpoint, response: httpx.Response, text: str, finish: str) -> None:
    events: Final = parse_sse(response.text)
    observed: Final = (
        response.status_code,
        _ending(endpoint, response.text, events),
        _error_frames(endpoint, events),
        _finishes(endpoint, events),
        _delivered_text(endpoint, events),
    )
    assert observed == (200, _NORMAL_ENDING[endpoint], (), (finish,), text), response.text


def _assert_500_naming_the_finish_reason(response: httpx.Response, script: _Script) -> None:
    assert response.status_code == 500, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert script.native in str(error["message"]), response.text
    assert _NOTE_MARKER not in response.text, response.text


def _completed_answer(endpoint: Endpoint, payload: Mapping[str, JsonValue]) -> tuple[str, str]:
    match endpoint:
        case "chat":
            choice: Final = _first_choice(payload)
            return str(object_value(choice["message"]).get("content") or ""), str(choice["finish_reason"])
        case "messages":
            blocks: Final = tuple(map(object_value, list_value(payload["content"])))
            text: Final = "".join(str(block["text"]) for block in blocks if block["type"] == "text")
            return text, str(payload["stop_reason"])
        case "responses":
            items: Final = tuple(map(object_value, list_value(payload["output"])))
            messages: Final = (item for item in items if item["type"] == "message")
            parts: Final = chain.from_iterable(list_value(item["content"]) for item in messages)
            return "".join(str(object_value(part)["text"]) for part in parts), str(payload["status"])


def _completed_tool_calls(endpoint: Endpoint, payload: Mapping[str, JsonValue]) -> _ToolCalls:
    match endpoint:
        case "chat":
            calls: Final = list_value(object_value(_first_choice(payload)["message"])["tool_calls"])
            functions: Final = tuple(object_value(object_value(call)["function"]) for call in calls)
            return tuple(
                (str(function["name"]), JSON_OBJECT.validate_json(str(function["arguments"]))) for function in functions
            )
        case "messages":
            blocks: Final = tuple(map(object_value, list_value(payload["content"])))
            return tuple(
                (str(block["name"]), object_value(block["input"])) for block in blocks if block["type"] == "tool_use"
            )
        case "responses":
            items: Final = tuple(map(object_value, list_value(payload["output"])))
            return tuple(
                (str(item["name"]), JSON_OBJECT.validate_json(str(item["arguments"])))
                for item in items
                if item["type"] == "function_call"
            )


def _assert_completed(endpoint: Endpoint, response: httpx.Response, text: str, finish: str) -> None:
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload.get("error") is None, response.text
    assert _completed_answer(endpoint, payload) == (text, finish), response.text


def _caller_outcome(endpoint: Endpoint, response: httpx.Response) -> tuple[int, str]:
    if response.status_code != 200:
        return response.status_code, "error body"
    return 200, _ending(endpoint, response.text, parse_sse(response.text))


def _outcomes_before_or_after_the_failure_is_surfaced(
    endpoint: Endpoint, script: _Script
) -> frozenset[tuple[int, str]]:
    streamed: Final = frozenset(((200, _NORMAL_ENDING[endpoint]), (200, _ERROR_ENDING[endpoint])))
    answered_before_streaming: Final = endpoint == "chat" and script.text == ""
    return streamed | frozenset(((500, "error body"),)) if answered_before_streaming else streamed


def _generate_calls(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target.split("?")[0].rsplit(":", 1)[-1] for request in wire.drain())


def _cost(usage: Mapping[str, int]) -> float:
    return usage["promptTokenCount"] * _INPUT_PRICE + usage["candidatesTokenCount"] * _OUTPUT_PRICE


def _recorded_row(gateway: Gateway, model: str) -> _SpendRow:
    rows: Final = held(
        lambda: gateway.rows(_SPEND_ROWS, (model,)),
        lambda found: len(found) == 1,
        holding=_ROW_SETTLE_SECONDS,
        seconds=_ROW_SECONDS,
    )
    return _SpendRow.model_validate(rows[0])


@dataclass(frozen=True, slots=True)
class _Priced:
    response: httpx.Response
    row: _SpendRow
    model: str
    identity: str


def _priced_row(
    scripted: _ScriptedGemini,
    endpoint: Endpoint,
    provider: Provider,
    script: _Script,
    build_body: _BuildBody,
) -> _Priced:
    identity: Final = f"priced-{uuid.uuid4().hex}"
    with scripted.gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, scripted.wire, identity=identity)
        response: Final = _send(scripted.gateway, endpoint, build_body(endpoint, model, script.prompt))
        return _Priced(response, _recorded_row(scripted.gateway, model), model, identity)


def _assert_recorded_once_as_a_gemini_success(priced: _Priced, provider: Provider, script: _Script) -> None:
    expected: Final = _SpendRow(
        status="success",
        spend=priced.row.spend,
        prompt_tokens=script.usage["promptTokenCount"],
        completion_tokens=script.usage["candidatesTokenCount"],
        total_tokens=script.usage["totalTokenCount"],
        model_group=priced.model,
        model_id=priced.identity,
        custom_llm_provider=provider,
    )
    assert priced.row == expected, priced.row
    assert priced.row.spend == pytest.approx(_cost(script.usage)), priced.row


def _group_rows(gateway: Gateway, groups: tuple[str, str]) -> tuple[_SpendRow, ...]:
    return tuple(map(_SpendRow.model_validate, gateway.rows(_GROUP_SPEND_ROWS, groups)))


def _bills_exactly(rows: tuple[_SpendRow, ...], usages: tuple[Mapping[str, int], ...]) -> bool:
    tokens: Final = (
        sum(row.prompt_tokens for row in rows),
        sum(row.completion_tokens for row in rows),
        sum(row.total_tokens for row in rows),
    )
    prompt: Final = sum(usage["promptTokenCount"] for usage in usages)
    completion: Final = sum(usage["candidatesTokenCount"] for usage in usages)
    total: Final = sum(usage["totalTokenCount"] for usage in usages)
    cost: Final = sum(map(_cost, usages))
    return tokens == (prompt, completion, total) and abs(sum(row.spend for row in rows) - cost) < _COST_TOLERANCE


def _group_billed_exactly(
    gateway: Gateway, groups: tuple[str, str], usages: tuple[Mapping[str, int], ...]
) -> tuple[_SpendRow, ...]:
    return held(
        lambda: _group_rows(gateway, groups),
        lambda found: _bills_exactly(found, usages),
        holding=_ROW_SETTLE_SECONDS,
        seconds=_ROW_SECONDS,
    )


def _assert_each_attempt_billed_once_and_the_fallback_succeeded(
    gateway: Gateway, pair: _FallbackPair, usages: tuple[Mapping[str, int], ...]
) -> None:
    rows: Final = _group_billed_exactly(gateway, pair.groups, usages)
    fallback_statuses: Final = tuple(row.status for row in rows if row.model_group == pair.fallback)
    assert (sorted(str(row.model_group) for row in rows), fallback_statuses) == (
        sorted(pair.groups),
        ("success",),
    ), rows


def _key_spend(gateway: Gateway, key: str) -> float:
    rows: Final = gateway.rows(_KEY_SPEND, (sha256(key.encode()).hexdigest(),))
    assert len(rows) == 1, rows
    return _KeySpend.model_validate(rows[0]).spend


def _refusal(response: httpx.Response) -> tuple[int, str]:
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    return response.status_code, str(error.get("type"))


def _assert_streamed_failure(endpoint: Endpoint, response: httpx.Response, script: _Script) -> None:
    _assert_error_ending(endpoint, response, script)


def _assert_non_streamed_failure(_: Endpoint, response: httpx.Response, script: _Script) -> None:
    _assert_500_naming_the_finish_reason(response, script)


_DELIVERY_MODES: Final = (
    pytest.param(_stream_body, _assert_streamed_failure, id="streamed"),
    pytest.param(_completion_body, _assert_non_streamed_failure, id="non-streamed"),
)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_TEXT_BEFORE_FAILURE))
def test_regression_failed_tool_call_after_text_delivers_the_text_and_ends_in_an_error(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_error_ending(endpoint, response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("script", _params(_NO_TEXT_FAILURE))
def test_regression_chat_failed_tool_call_without_text_answers_500_naming_the_finish_reason(
    scripted_gemini: _ScriptedGemini, script: _Script, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, "chat", provider, script)
    _assert_500_naming_the_finish_reason(response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ERROR_EVENT_ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NO_TEXT_FAILURE))
def test_regression_failed_tool_call_without_text_ends_in_the_error_event(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_error_ending(endpoint, response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_regression_unrescued_failed_tool_call_stream_is_recorded_once_as_a_success_with_gemini_usage_and_cost(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    script: Final = _text_then_finish("MALFORMED_FUNCTION_CALL")
    priced: Final = _priced_row(scripted_gemini, endpoint, provider, script, _stream_body)
    _assert_error_ending(endpoint, priced.response, script)
    _assert_recorded_once_as_a_gemini_success(priced, provider, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_regression_failed_tool_call_without_text_is_answered_by_the_fallback_deployment(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    script: Final = _finish_without_text("MALFORMED_FUNCTION_CALL")
    with _primary_with_fallback(gateway, provider) as pair:
        response: Final = _send(
            gateway, endpoint, _stream_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        )
        _assert_normal_ending(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        assert (_generate_calls(pair.primary_wire), _generate_calls(pair.fallback_wire)) == (
            ("streamGenerateContent",),
            ("streamGenerateContent",),
        ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_REAL_TOOL_CALLS))
def test_guard_real_tool_call_before_a_malformed_finish_still_ends_as_a_tool_call(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_normal_ending(endpoint, response, "", _TOOL_FINISH[endpoint])
    assert _delivered_tool_calls(endpoint, parse_sse(response.text)) == ((_TOOL_NAME, dict(_TOOL_ARGUMENTS)),), (
        response.text
    )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NORMAL_FINISHES))
def test_guard_finish_reasons_other_than_a_failed_tool_call_stay_a_normal_finish(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_normal_ending(endpoint, response, script.text, _STOP_FINISH[endpoint])


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_clean_stop_is_recorded_as_a_success_with_the_same_usage_and_cost(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    script: Final = _text_then_finish("STOP")
    priced: Final = _priced_row(scripted_gemini, endpoint, provider, script, _stream_body)
    _assert_normal_ending(endpoint, priced.response, script.text, _STOP_FINISH[endpoint])
    _assert_recorded_once_as_a_gemini_success(priced, provider, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_WITH_FINISH_MESSAGE))
def test_guard_gemini_finish_message_never_reaches_the_caller(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    allowed: Final = _outcomes_before_or_after_the_failure_is_surfaced(endpoint, script)
    assert _caller_outcome(endpoint, response) in allowed, (response.status_code, response.text)
    assert _delivered_text(endpoint, parse_sse(response.text)) == script.text, response.text
    assert _NOTE_MARKER not in response.text, response.text


def _complete_script(
    scripted: _ScriptedGemini, endpoint: Endpoint, provider: Provider, script: _Script
) -> httpx.Response:
    return _send(scripted.gateway, endpoint, _completion_body(endpoint, scripted.models[provider], script.prompt))


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NON_STREAMED_FAILURES))
def test_regression_non_streamed_failed_tool_call_answers_500_naming_the_finish_reason(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _complete_script(scripted_gemini, endpoint, provider, script)
    _assert_500_naming_the_finish_reason(response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_regression_non_streamed_failed_tool_call_is_answered_by_the_fallback_deployment(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    with _primary_with_fallback(gateway, provider) as pair:
        body: Final = _completion_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        response: Final = _send(gateway, endpoint, body)
        _assert_completed(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        assert (_generate_calls(pair.primary_wire), _generate_calls(pair.fallback_wire)) == (
            ("generateContent",),
            ("generateContent",),
        ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_regression_non_streamed_unrescued_failed_tool_call_is_recorded_once_as_a_success_with_gemini_usage_and_cost(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    priced: Final = _priced_row(scripted_gemini, endpoint, provider, script, _completion_body)
    _assert_500_naming_the_finish_reason(priced.response, script)
    _assert_recorded_once_as_a_gemini_success(priced, provider, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_THOUGHT_BEFORE_FAILURE))
def test_regression_failed_tool_call_after_reasoning_ends_in_an_error_without_calling_the_fallback(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    with _primary_with_fallback(gateway, provider) as pair:
        response: Final = _send(
            gateway, endpoint, _stream_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        )
        _assert_error_ending(endpoint, response, script)
        events: Final = parse_sse(response.text)
        assert _delivered_reasoning(endpoint, events) == _THOUGHT, response.text
        assert _responses_started(events) == _RESPONSES_STARTED_ONCE[endpoint], response.text
        assert (_generate_calls(pair.primary_wire), _generate_calls(pair.fallback_wire)) == (
            ("streamGenerateContent",),
            (),
        ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NO_TEXT_FAILURE[:2]))
def test_regression_streamed_failure_rescued_by_a_fallback_bills_both_attempts_exactly_once(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    with _primary_with_fallback(gateway, provider) as pair:
        response: Final = _send(
            gateway, endpoint, _stream_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        )
        _assert_normal_ending(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        _assert_each_attempt_billed_once_and_the_fallback_succeeded(
            gateway, pair, (script.usage, _HEALTHY_SCRIPT.usage)
        )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _RESPONSES_API_RESCUED_ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NO_TEXT_FAILURE[:2]))
def test_regression_streamed_failure_rescued_through_the_responses_api_bills_both_attempts_exactly_once(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    with _primary_with_fallback(
        gateway,
        provider,
        answering_model_discovery(_respond_through_the_responses_api),
        _add_responses_api_deployment,
    ) as pair:
        body: Final = _stream_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        response: Final = _send(gateway, endpoint, body)
        _assert_normal_ending(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        assert (_generate_calls(pair.primary_wire), posted_targets(pair.fallback_wire)) == (
            ("streamGenerateContent",),
            ("/v1/responses",),
        ), response.text
        _assert_each_attempt_billed_once_and_the_fallback_succeeded(gateway, pair, (script.usage, _RESPONSES_API_USAGE))


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_regression_non_streamed_failure_rescued_by_a_fallback_bills_both_attempts_exactly_once(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    with _primary_with_fallback(gateway, provider) as pair:
        body: Final = _completion_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        _assert_completed(endpoint, _send(gateway, endpoint, body), _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        _assert_each_attempt_billed_once_and_the_fallback_succeeded(
            gateway, pair, (script.usage, _HEALTHY_SCRIPT.usage)
        )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_upstream_503_after_reasoning_still_ends_in_an_error_without_the_fallback(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    script: Final = _UPSTREAM_503_AFTER_THOUGHT
    with _primary_with_fallback(gateway, provider) as pair:
        response: Final = _send(
            gateway, endpoint, _stream_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        )
        events: Final = parse_sse(response.text)
        observed: Final = (
            response.status_code,
            _ending(endpoint, response.text, events),
            _finishes(endpoint, events),
            _delivered_reasoning(endpoint, events),
        )
        assert observed == (200, _ERROR_ENDING[endpoint], (), _THOUGHT), response.text
        assert _OVERLOADED_MESSAGE in _terminal_error_message(endpoint, events), response.text
        assert (_generate_calls(pair.primary_wire), _generate_calls(pair.fallback_wire)) == (
            ("streamGenerateContent",),
            (),
        ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_healthy_stream_with_a_fallback_configured_bills_only_its_own_usage(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    script: Final = _text_then_finish("STOP")
    with _primary_with_fallback(gateway, provider) as pair:
        response: Final = _send(
            gateway, endpoint, _stream_body(endpoint, pair.primary, script.prompt, fallback=pair.fallback)
        )
        _assert_normal_ending(endpoint, response, script.text, _STOP_FINISH[endpoint])
        rows: Final = _group_billed_exactly(gateway, pair.groups, (script.usage,))
        assert [(row.model_group, row.status) for row in rows] == [(pair.primary, "success")], rows
        assert (_generate_calls(pair.primary_wire), _generate_calls(pair.fallback_wire)) == (
            ("streamGenerateContent",),
            (),
        ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_non_streamed_real_tool_call_with_a_malformed_finish_stays_a_tool_call(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _complete_script(scripted_gemini, endpoint, provider, _REAL_TOOL_CALLS[0])
    _assert_completed(endpoint, response, "", _TOOL_FINISH[endpoint])
    assert _completed_tool_calls(endpoint, JSON_OBJECT.validate_json(response.content)) == (
        (_TOOL_NAME, dict(_TOOL_ARGUMENTS)),
    ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NORMAL_FINISHES))
def test_guard_non_streamed_finish_reasons_other_than_a_failed_tool_call_stay_a_normal_answer(
    scripted_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _complete_script(scripted_gemini, endpoint, provider, script)
    _assert_completed(endpoint, response, script.text, _STOP_FINISH[endpoint])


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("script", _params(_TWO_CANDIDATES))
def test_guard_non_streamed_one_of_two_candidates_ending_malformed_stays_200(
    scripted_gemini: _ScriptedGemini, script: _Script, provider: Provider
) -> None:
    body: Final = _completion_body("chat", scripted_gemini.models[provider], script.prompt, candidate_count=2)
    response: Final = _send(scripted_gemini.gateway, "chat", body)
    assert response.status_code == 200, response.text
    choices: Final = tuple(map(object_value, list_value(JSON_OBJECT.validate_json(response.content)["choices"])))
    answers: Final = tuple(
        (choice["index"], object_value(choice["message"])["content"], choice["finish_reason"]) for choice in choices
    )
    assert sorted(answers, key=str) == [(0, _ANSWER, "stop"), (1, _SECOND_ANSWER, "stop")], response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("script", _params(_TWO_CANDIDATES))
def test_guard_one_of_two_candidates_ending_malformed_does_not_end_the_stream_in_an_error(
    scripted_gemini: _ScriptedGemini, script: _Script, provider: Provider
) -> None:
    response: Final = _send(
        scripted_gemini.gateway,
        "chat",
        _stream_body("chat", scripted_gemini.models[provider], script.prompt, candidate_count=2),
    )
    events: Final = parse_sse(response.text)
    choices: Final = _streamed_choices(_payloads(events))

    def text_of(index: int) -> str:
        chosen: Final = (choice for choice in choices if choice.get("index") == index)
        return "".join(str(object_value(choice.get("delta") or {}).get("content") or "") for choice in chosen)

    observed: Final = (
        response.status_code,
        _ending("chat", response.text, events),
        _error_frames("chat", events),
        frozenset(_finishes("chat", events)),
        text_of(0),
        text_of(1),
    )
    assert observed == (200, "done", (), frozenset(("stop",)), _ANSWER, _SECOND_ANSWER), response.text


@dataclass(frozen=True, slots=True)
class _CooldownRig:
    gateway: Gateway
    cache: OwnedRedis


@pytest.fixture(scope="module")
def cooldown_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CooldownRig]:
    directory: Final = tmp_path_factory.mktemp("gemini-failed-tool-call-cooldown")
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(directory))
        owned: Final = stack.enter_context(
            owned_proxy_process(
                upstream,
                directory,
                {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"},
                config=cooldown_config(directory, router_settings(cache)),
                remove_environment=("PROMETHEUS_MULTIPROC_DIR",),
            )
        )
        yield _CooldownRig(owned.gateway, cache)


def _new_group() -> str:
    return f"integration-gemini-cooldown-{uuid.uuid4().hex[:12]}"


def _requests_until_both_deployments_passed_the_threshold(
    gateway: Gateway,
    endpoint: Endpoint,
    body: Mapping[str, JsonValue],
    wires: tuple[Wire, Wire],
    *,
    key: str | None = None,
) -> tuple[httpx.Response, ...]:
    def send() -> Iterator[httpx.Response]:
        for response in (_send(gateway, endpoint, body, key=key) for _ in range(_MAX_COOLDOWN_REQUESTS)):
            yield response
            if NO_DEPLOYMENTS in response.text or _both_passed_the_threshold(wires):
                return

    return tuple(send())


def _both_passed_the_threshold(wires: tuple[Wire, Wire]) -> bool:
    return all(wire.received.qsize() > _COOLDOWN_THRESHOLD for wire in wires)


def _calls_and_cooldown_until_cooled(
    rig: _CooldownRig, endpoint: Endpoint, body: Mapping[str, JsonValue], wire: Wire, identity: str
) -> tuple[tuple[int, bool], ...]:
    def observe() -> tuple[int, bool]:
        _send(rig.gateway, endpoint, body)
        return wire.received.qsize(), cooled(rig.cache, identity)

    def send() -> Iterator[tuple[int, bool]]:
        for observed in (observe() for _ in range(_MAX_COOLDOWN_REQUESTS)):
            yield observed
            if observed[1]:
                return

    return tuple(send())


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_guard_failed_tool_calls_do_not_cool_the_group_for_a_request_without_tools(
    cooldown_rig: _CooldownRig, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = cooldown_rig.gateway
    group: Final = _new_group()
    respond: Final = _failing_only_requests_with_tools(script)
    with wire_server(respond) as first, wire_server(respond) as second, gateway.scenario() as scenario:
        _add_deployment(scenario, provider, first, group=group)
        _add_deployment(scenario, provider, second, group=group)
        burst: Final = _requests_until_both_deployments_passed_the_threshold(
            gateway, endpoint, _stream_body(endpoint, group, script.prompt), (first, second)
        )
        assert [response.text for response in burst if NO_DEPLOYMENTS in response.text] == []
        assert _both_passed_the_threshold((first, second)), (first.received.qsize(), second.received.qsize())
        answer: Final = _send(gateway, endpoint, _stream_body(endpoint, group, _PLAIN_PROMPT, declares_tools=False))
        _assert_normal_ending(endpoint, answer, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_guard_non_streamed_failed_tool_calls_do_not_cool_the_group_for_a_second_client(
    cooldown_rig: _CooldownRig, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = cooldown_rig.gateway
    group: Final = _new_group()
    respond: Final = _failing_only_requests_with_tools(script)
    with wire_server(respond) as first, wire_server(respond) as second, gateway.scenario() as scenario:
        _add_deployment(scenario, provider, first, group=group)
        _add_deployment(scenario, provider, second, group=group)
        failing_client: Final = scenario.key(models=[group])
        second_client: Final = scenario.key(models=[group])
        burst: Final = _requests_until_both_deployments_passed_the_threshold(
            gateway,
            endpoint,
            _completion_body(endpoint, group, script.prompt),
            (first, second),
            key=failing_client,
        )
        assert [response.text for response in burst if NO_DEPLOYMENTS in response.text] == []
        assert _both_passed_the_threshold((first, second)), (first.received.qsize(), second.received.qsize())
        body: Final = _completion_body(endpoint, group, _PLAIN_PROMPT, declares_tools=False)
        answer: Final = _send(gateway, endpoint, body, key=second_client)
        _assert_completed(endpoint, answer, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize(
    "respond",
    (
        pytest.param(_respond_with_upstream_500, id="mid-stream"),
        pytest.param(_respond_with_plain_500, id="plain"),
    ),
)
def test_guard_upstream_500_still_cools_the_deployment_at_the_failure_threshold(
    cooldown_rig: _CooldownRig, respond: Callable[[Request], Reply], endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = cooldown_rig.gateway
    group: Final = _new_group()
    identity: Final = f"failing-{uuid.uuid4().hex}"
    body: Final = _stream_body(endpoint, group, _PLAIN_PROMPT, declares_tools=False)
    with (
        wire_server(respond) as failing,
        wire_server(_respond_with_healthy_answer) as healthy,
        gateway.scenario() as scenario,
    ):
        _add_deployment(scenario, provider, failing, group=group, identity=identity)
        _add_deployment(scenario, provider, healthy, group=group)
        history: Final = _calls_and_cooldown_until_cooled(cooldown_rig, endpoint, body, failing, identity)
        assert [calls for calls, is_cooled in history if is_cooled and calls < _COOLDOWN_THRESHOLD] == [], history
        assert history[-1][1], history
        calls_when_cooled: Final = failing.received.qsize()
        after: Final = tuple(_send(gateway, endpoint, body) for _ in range(_REQUESTS_AFTER_COOLDOWN))
        for response in after:
            _assert_normal_ending(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        assert failing.received.qsize() == calls_when_cooled, history


def _deployment_health(gateway: Gateway, identity: str) -> tuple[float, float, tuple[float, ...]]:
    samples: Final = tuple(sample for sample in scrape(gateway) if sample.labels.get("model_id") == identity)
    return (
        sum(sample.value for sample in samples if sample.name == _DEPLOYMENT_SUCCESSES),
        sum(sample.value for sample in samples if sample.name == _DEPLOYMENT_FAILURES),
        tuple(sample.value for sample in samples if sample.name == _DEPLOYMENT_STATE),
    )


def _held_deployment_health(
    gateway: Gateway, identity: str, expected: tuple[float, float, tuple[float, ...]]
) -> tuple[float, float, tuple[float, ...]]:
    return held(
        lambda: _deployment_health(gateway, identity),
        lambda observed: observed == expected,
        holding=_ROW_SETTLE_SECONDS,
        seconds=_ROW_SECONDS,
    )


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_plain_upstream_500_counts_as_a_deployment_failure_and_marks_a_partial_outage(
    cooldown_rig: _CooldownRig, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = cooldown_rig.gateway
    identity: Final = f"plain-500-{uuid.uuid4().hex}"
    with wire_server(_respond_with_plain_500) as wire, gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, wire, identity=identity)
        response: Final = _send(gateway, endpoint, _completion_body(endpoint, model, _PLAIN_PROMPT))
        assert response.status_code == 500, response.text
        _held_deployment_health(gateway, identity, (0.0, 1.0, (_PARTIAL_OUTAGE,)))


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize(("build_body", "assert_failure"), _DELIVERY_MODES)
def test_regression_failed_tool_call_counts_as_a_deployment_success_without_a_partial_outage(
    cooldown_rig: _CooldownRig,
    build_body: _BuildBody,
    assert_failure: _AssertFailure,
    endpoint: Endpoint,
    provider: Provider,
) -> None:
    gateway: Final = cooldown_rig.gateway
    identity: Final = f"failed-tool-call-{uuid.uuid4().hex}"
    script: Final = _text_then_finish("MALFORMED_FUNCTION_CALL")
    with wire_server(_respond_with_script) as wire, gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, wire, identity=identity)
        assert_failure(endpoint, _send(gateway, endpoint, build_body(endpoint, model, script.prompt)), script)
        _held_deployment_health(gateway, identity, (1.0, 0.0, (_HEALTHY_STATE,)))


@dataclass(frozen=True, slots=True)
class _BudgetRig:
    gateway: Gateway
    wire: Wire


@pytest.fixture(scope="module")
def budget_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_BudgetRig]:
    directory: Final = tmp_path_factory.mktemp("gemini-failed-tool-call-budgets")
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(directory))
        wire: Final = stack.enter_context(wire_server(_respond_with_script))
        settings: Final = router_settings(
            cache,
            provider_budget_config={
                provider: {"budget_limit": _PROVIDER_BUDGET, "time_period": _BUDGET_WINDOW} for provider in _PROVIDERS
            },
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                upstream,
                directory,
                {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"},
                config=cooldown_config(directory, settings, prometheus=False),
            )
        )
        yield _BudgetRig(owned.gateway, wire)


def _provider_spend(gateway: Gateway, provider: Provider) -> float:
    providers: Final = object_value(gateway.get("/provider/budgets")["providers"])
    return _ProviderBudget.model_validate(providers[provider]).spend


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize(("build_body", "assert_failure"), _DELIVERY_MODES)
def test_regression_unrescued_failed_tool_call_debits_the_provider_budget_by_its_cost(
    budget_rig: _BudgetRig,
    build_body: _BuildBody,
    assert_failure: _AssertFailure,
    endpoint: Endpoint,
    provider: Provider,
) -> None:
    gateway: Final = budget_rig.gateway
    script: Final = _COSTLY_FAILURE
    debited: Final = _provider_spend(gateway, provider) + _cost(script.usage)
    with gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, budget_rig.wire)
        assert_failure(endpoint, _send(gateway, endpoint, build_body(endpoint, model, script.prompt)), script)
        held(
            lambda: _provider_spend(gateway, provider),
            lambda spend: abs(spend - debited) < _COST_TOLERANCE,
            holding=_ROW_SETTLE_SECONDS,
            seconds=_ROW_SECONDS,
        )


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize(("build_body", "assert_failure"), _DELIVERY_MODES)
def test_regression_unrescued_failed_tool_call_debits_the_deployment_budget_until_it_is_crossed(
    budget_rig: _BudgetRig,
    build_body: _BuildBody,
    assert_failure: _AssertFailure,
    endpoint: Endpoint,
    provider: Provider,
) -> None:
    gateway: Final = budget_rig.gateway
    script: Final = _COSTLY_FAILURE
    with gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, budget_rig.wire, max_budget=_cost(script.usage) / 2)
        assert_failure(endpoint, _send(gateway, endpoint, build_body(endpoint, model, script.prompt)), script)
        probe: Final = _completion_body("chat", model, _ZERO_COST_PROBE.prompt, declares_tools=False)
        refused: Final = eventually(
            lambda: _send(gateway, "chat", probe), lambda answer: _CROSSED_BUDGET in answer.text, seconds=_ROW_SECONDS
        )
        assert not refused.is_success, refused.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize(("build_body", "assert_failure"), _DELIVERY_MODES)
def test_regression_unrescued_failed_tool_call_charges_the_key_once_and_its_budget_refuses_the_next_request(
    scripted_gemini: _ScriptedGemini,
    build_body: _BuildBody,
    assert_failure: _AssertFailure,
    endpoint: Endpoint,
    provider: Provider,
) -> None:
    gateway: Final = scripted_gemini.gateway
    script: Final = _COSTLY_FAILURE
    cost: Final = _cost(script.usage)
    with gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, scripted_gemini.wire)
        key: Final = scenario.key(models=[model], max_budget=cost / 2)
        body: Final = {**build_body(endpoint, model, script.prompt), **_OUTPUT_CAPS[endpoint]}
        assert_failure(endpoint, _send(gateway, endpoint, body, key=key), script)
        held(
            lambda: _key_spend(gateway, key),
            lambda spend: abs(spend - cost) < _COST_TOLERANCE,
            holding=_ROW_SETTLE_SECONDS,
            seconds=_ROW_SECONDS,
        )
        probe: Final = {
            **_completion_body("chat", model, _ZERO_COST_PROBE.prompt, declares_tools=False),
            **_OUTPUT_CAPS["chat"],
        }
        after: Final = _send(gateway, "chat", probe, key=key)
        assert _refusal(after) == _BUDGET_REFUSAL, after.text


def _pii_spans(text: str, value: str, entity: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        {"entity_type": entity, "start": match.start(), "end": match.end(), "score": _PII_SCORE}
        for match in re.finditer(re.escape(value), text)
    )


def _analyze_pii(request: Request) -> Reply:
    if request.target != "/analyze":
        return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())
    text: Final = str(JSON_OBJECT.validate_json(request.body)["text"])
    spans: Final = chain.from_iterable(_pii_spans(text, value, entity) for value, entity in _PII_ENTITIES.items())
    return Reply(body=json.dumps([dict(span) for span in spans]).encode())


def _anonymized_item(result: JsonValue) -> Mapping[str, JsonValue]:
    span: Final = object_value(result)
    return {
        "entity_type": span["entity_type"],
        "start": span["start"],
        "end": span["end"],
        "operator": "replace",
        "text": f"<{span['entity_type']}>",
    }


def _anonymize_pii(request: Request) -> Reply:
    if request.target != "/anonymize":
        return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())
    body: Final = JSON_OBJECT.validate_json(request.body)
    masked: Final = reduce(
        lambda text, entry: text.replace(entry[0], f"<{entry[1]}>"), _PII_ENTITIES.items(), str(body["text"])
    )
    items: Final = [dict(_anonymized_item(result)) for result in list_value(body["analyzer_results"])]
    return Reply(body=json.dumps({"text": masked, "items": items}).encode())


def _presidio_config(directory: Path, analyzer: Wire, anonymizer: Wire) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    guardrail: Final = {
        "guardrail_name": "gemini-failed-tool-call-presidio-output",
        "litellm_params": {
            "guardrail": "presidio",
            "mode": "post_call",
            "default_on": True,
            "presidio_analyzer_api_base": analyzer.url,
            "presidio_anonymizer_api_base": anonymizer.url,
            "presidio_filter_scope": "output",
        },
    }
    path: Final = directory / f"presidio-output-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump({**base, "guardrails": [guardrail]}))
    return path


@pytest.fixture(scope="module")
def presidio_gemini(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ScriptedGemini]:
    directory: Final = tmp_path_factory.mktemp("gemini-failed-tool-call-presidio")
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        analyzer: Final = stack.enter_context(wire_server(_analyze_pii))
        anonymizer: Final = stack.enter_context(wire_server(_anonymize_pii))
        wire: Final = stack.enter_context(wire_server(_respond_with_script))
        owned: Final = stack.enter_context(
            owned_proxy_process(upstream, directory, {}, config=_presidio_config(directory, analyzer, anonymizer))
        )
        scenario: Final = stack.enter_context(owned.gateway.scenario())
        models: Final[Mapping[Provider, str]] = MappingProxyType(
            {provider: _add_deployment(scenario, provider, wire) for provider in _PROVIDERS}
        )
        yield _ScriptedGemini(owned.gateway, models, wire)


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _PRESIDIO_ENDPOINTS)
@pytest.mark.parametrize("script", _params(_PII_BEFORE_FAILURE))
def test_regression_presidio_output_masking_streams_masked_text_then_the_error_and_never_the_raw_pii(
    presidio_gemini: _ScriptedGemini, script: _Script, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(presidio_gemini, endpoint, provider, script)
    _assert_error_ending(endpoint, response, script)
    assert [value for value in _PII_ENTITIES if value in response.text] == [], response.text
