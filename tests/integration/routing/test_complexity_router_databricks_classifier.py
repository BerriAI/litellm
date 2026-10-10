import asyncio
import json
import os
import re
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with
from litellm.router_strategy.complexity_router.config import DEFAULT_JEV_INSTRUCTIONS

_MODEL: Final = "ai_decide"
_ROUTE: Final = "/api/2.0/ai-functions/ai-decide"
_API_KEY: Final = "synthetic-databricks-key"
_CLASSIFIER_MODEL: Final = f"databricks/{_MODEL}"
_TYPESAFE_MODEL: Final = "jev-1.13.0"
_TYPESAFE_KEY: Final = "synthetic-typesafe-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_ANTHROPIC_VERSION: Final = MappingProxyType({"anthropic-version": "2023-06-01"})
_SALT: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
_TIER_CRITERIA: Final[dict[str, JsonValue]] = {
    "SIMPLE": "Greetings, chitchat, or short factual lookups with known answers",
    "MEDIUM": "Everyday requests needing explanation, light reasoning, or minor technical work",
    "COMPLEX": "Non-trivial code, architecture, multi-step work, or specialized domain depth",
    "REASONING": "Open-ended analysis, proofs, tradeoffs, or tasks requiring careful thought",
}
_PROBABILITIES: Final[dict[str, JsonValue]] = {"SIMPLE": 0.03, "MEDIUM": 0.06, "COMPLEX": 0.91}
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 12, "output_tokens": 1}
_ANSWERS: Final[dict[str, JsonValue]] = {
    "tier": {"type": "choice", "choice": "COMPLEX", "confidence": 0.91, "probabilities": _PROBABILITIES}
}
_METADATA: Final[dict[str, JsonValue]] = {"version": "1.0"}
_ANSWER: Final[dict[str, JsonValue]] = {"response": {"answers": _ANSWERS}, "metadata": _METADATA}
_TYPESAFE_ANSWER: Final[dict[str, JsonValue]] = {"model": _TYPESAFE_MODEL, "answers": _ANSWERS, "usage": _USAGE}
_ROWS_QUERY: Final = (
    "SELECT request_id, status, model, call_type, cache_hit, custom_llm_provider, spend, "
    "metadata->>'internal_call_origin' AS origin "
    'FROM "LiteLLM_SpendLogs" WHERE model_group = %s ORDER BY "startTime"'
)
_SIMPLE_TEXT: Final = "simple answer"
_COMPLEX_TEXT: Final = "complex answer"
_CALL_TYPES: Final = MappingProxyType(
    {"/v1/chat/completions": "acompletion", "/v1/responses": "aresponses", "/v1/messages": "anthropic_messages"}
)


@dataclass(frozen=True, slots=True)
class _Routed:
    name: str
    identity: str
    simple: str
    complex_: str


@dataclass(frozen=True, slots=True)
class _Served:
    route: str
    stream: bool
    status: int
    identity: str
    text: str
    cause: str
    classifier_cost: str | None


@dataclass(frozen=True, slots=True)
class _Failure:
    label: str
    respond: Callable[[Request], Reply]
    overrides: Mapping[str, JsonValue]
    reason: str
    error_type: str
    classifier_rows: int


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _prompt() -> str:
    return f"design a distributed cache {uuid.uuid4().hex}"


def _short_prompt() -> str:
    return f"hi {uuid.uuid4().hex[:8]}"


def _answering(body: Mapping[str, JsonValue]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(body).encode())

    return respond


def _failing(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        return Reply(status=status, body=json.dumps({"error_code": "FAILED", "message": f"scripted {status}"}).encode())

    return respond


def _slow(seconds: float) -> Callable[[Request], Reply]:
    answer: Final = _answering(_ANSWER)

    def respond(request: Request) -> Reply:
        time.sleep(seconds)
        return answer(request)

    return respond


def _tier(prefix: str, text: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = _JSON_OBJECT.validate_json(request.body) if request.body else {}
        identity: Final = f"{prefix}-{uuid.uuid4().hex[:8]}"
        if request.target == "/chat/completions":
            return chat_reply(identity, "tier-model", text, stream=bool(body.get("stream")))
        if request.target == "/responses":
            return responses_reply(identity, "tier-model", text, stream=bool(body.get("stream")))
        return Reply(status=404, body=json.dumps({"error": {"message": f"no route {request.target}"}}).encode())

    return answering_model_discovery(respond)


@contextmanager
def _wires(classifier: Callable[[Request], Reply] | None = None) -> Iterator[tuple[Wire, Wire, Wire]]:
    with ExitStack() as stack:
        judge: Final = stack.enter_context(wire_server(classifier or _answering(_ANSWER)))
        simple: Final = stack.enter_context(wire_server(_tier("simple", _SIMPLE_TEXT)))
        complex_: Final = stack.enter_context(wire_server(_tier("complex", _COMPLEX_TEXT)))
        yield judge, simple, complex_


def _classifier_config(judge_url: str, **overrides: JsonValue) -> dict[str, JsonValue]:
    return {
        "provider": "databricks",
        "model": _MODEL,
        "api_base": judge_url,
        "api_key": _API_KEY,
        "timeout_ms": 20000,
        "circuit_breaker_enabled": False,
        **overrides,
    }


def _router_config(classifier: Mapping[str, JsonValue], simple: str, complex_: str) -> dict[str, JsonValue]:
    return {
        "classifier_type": "oss_classifier",
        "opensource_classifier_config": dict(classifier),
        "tiers": {"SIMPLE": simple, "MEDIUM": simple, "COMPLEX": complex_, "REASONING": complex_},
    }


def _deploy(
    scenario: Scenario, judge: Wire, simple: Wire, complex_: Wire, classifier: Mapping[str, JsonValue] | None = None
) -> _Routed:
    simple_model: Final = scenario.model(model="openai/simple-tier", api_base=simple.url, api_key="synthetic-tier-key")
    complex_model: Final = scenario.model(
        model="openai/complex-tier", api_base=complex_.url, api_key="synthetic-tier-key"
    )
    name: Final = f"router-{uuid.uuid4().hex[:10]}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_config": _router_config(
                    classifier if classifier is not None else _classifier_config(judge.url), simple_model, complex_model
                ),
            },
            "model_info": {},
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return _Routed(name=name, identity=identity, simple=simple_model, complex_=complex_model)


def _posts(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


def _state(prompt: str, system: str | None) -> str:
    if system is None:
        return f"\nClassify this message:\n{prompt}"
    return f"\nCaller system prompt, quoted as task context:\n{system}\n\nClassify this message:\n{prompt}"


def _assert_classifier_call(
    call: Request, *, prompt: str, system: str | None = None, key: str = _API_KEY, endpoint: str | None = None
) -> None:
    assert call.target == (_ROUTE if endpoint is None else f"/{endpoint}/invocations"), call.target
    assert call.headers.get("authorization") == f"Bearer {key}", call.headers
    assert json.loads(call.body) == {
        "state": _state(prompt, system),
        **({} if endpoint is None else {"model": endpoint}),
        "questions": {"tier": {"type": "choice", "instructions": DEFAULT_JEV_INSTRUCTIONS, "criteria": _TIER_CRITERIA}},
    }, call.body


def _assert_classified(headers: Mapping[str, str], *, router: str) -> None:
    assert (
        headers.get("x-litellm-model-group"),
        headers.get("x-litellm-model-name"),
        headers.get("x-litellm-complexity-router-tier"),
        headers.get("x-litellm-complexity-router-cause"),
        headers.get("x-litellm-classifier-cost"),
    ) == (router, "openai/complex-tier", "COMPLEX", "jev_classifier", None), dict(headers)


def _assert_fell_back(headers: Mapping[str, str], *, router: str) -> None:
    assert (
        headers.get("x-litellm-model-group"),
        headers.get("x-litellm-model-name"),
        headers.get("x-litellm-complexity-router-tier"),
        headers.get("x-litellm-complexity-router-cause"),
        headers.get("x-litellm-classifier-cost"),
    ) == (router, "openai/simple-tier", "SIMPLE", "heuristic_scorer", None), dict(headers)


def _rows(router: str, *, want: int) -> tuple[dict[str, JsonValue], ...]:
    return tuple(eventually(lambda: read_rows(_ROWS_QUERY, (router,)), lambda found: len(found) >= want, seconds=70))


def _classifier_rows(rows: Sequence[Mapping[str, JsonValue]]) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(row for row in rows if row["origin"] == "autorouter_classifier")


def _request_rows(rows: Sequence[Mapping[str, JsonValue]]) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(row for row in rows if row["origin"] != "autorouter_classifier")


def _assert_classifier_row(
    row: Mapping[str, JsonValue], *, model: str = _CLASSIFIER_MODEL, provider: str = "databricks", spend: float = 0.0
) -> None:
    assert (row["status"], row["call_type"], row["model"], row["custom_llm_provider"], row["cache_hit"]) == (
        "success",
        "pass_through_endpoint",
        model,
        provider,
        "False",
    ), row
    assert _number(row["spend"]) == pytest.approx(spend), row


def _issued_response_id(caller_id: str) -> str:
    decrypted: Final = decrypt_if_encrypted_with(caller_id.removeprefix("resp_"), _SALT)
    assert decrypted is not None, caller_id
    return decrypted.split(";")[0].split("response_id:")[-1]


def _logged_ids(route: str, caller_id: str) -> frozenset[str]:
    if route != "/v1/responses":
        return frozenset({caller_id})
    return frozenset({caller_id, _issued_response_id(caller_id)})


def _assert_logged_once(rows: Sequence[Mapping[str, JsonValue]], *, route: str, request_id: str) -> None:
    (classifier_row,) = _classifier_rows(rows)
    _assert_classifier_row(classifier_row)
    (request_row,) = _request_rows(rows)
    assert request_row["request_id"] in _logged_ids(route, request_id), (request_row, request_id)
    assert (request_row["call_type"], request_row["status"]) == (_CALL_TYPES[route], "success"), request_row


def _data_frames(lines: Sequence[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data:").strip())
        for line in lines
        if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]"
    )


def _delta_texts(frames: Sequence[Mapping[str, JsonValue]]) -> Iterator[str]:
    for frame in frames:
        choices: Final = frame["choices"]
        if not isinstance(choices, list):
            continue
        for choice in choices:
            content: Final = object_value(object_value(choice)["delta"]).get("content")
            if isinstance(content, str):
                yield content


def _chat_stream_identity_and_text(frames: Sequence[Mapping[str, JsonValue]]) -> tuple[str, str]:
    identities: Final = {string_value(frame["id"]) for frame in frames}
    assert len(identities) == 1, identities
    return identities.pop(), "".join(_delta_texts(frames))


def _responses_stream_identity_and_text(frames: Sequence[Mapping[str, JsonValue]]) -> tuple[str, str]:
    created: Final = [frame for frame in frames if frame["type"] == "response.created"]
    assert len(created) == 1, frames
    identity: Final = string_value(object_value(created[0]["response"])["id"])
    text: Final = "".join(
        string_value(frame["delta"]) for frame in frames if frame["type"] == "response.output_text.delta"
    )
    return identity, text


def _messages_stream_identity_and_text(frames: Sequence[Mapping[str, JsonValue]]) -> tuple[str, str]:
    started: Final = [frame for frame in frames if frame["type"] == "message_start"]
    assert len(started) == 1, frames
    identity: Final = string_value(object_value(started[0]["message"])["id"])
    deltas: Final = [object_value(frame["delta"]) for frame in frames if frame["type"] == "content_block_delta"]
    text: Final = "".join(string_value(delta["text"]) for delta in deltas if delta.get("type") == "text_delta")
    return identity, text


def _body(route: str, model: str, prompt: str, *, stream: bool) -> dict[str, JsonValue]:
    if route == "/v1/chat/completions":
        return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}
    if route == "/v1/responses":
        return {"model": model, "input": prompt, "stream": stream}
    return {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _identity_and_text(route: str, body: Mapping[str, JsonValue]) -> tuple[str, str]:
    if route == "/v1/chat/completions":
        (choice,) = body["choices"] if isinstance(body["choices"], list) else ()
        return string_value(body["id"]), string_value(object_value(object_value(choice)["message"])["content"])
    if route == "/v1/responses":
        (item,) = body["output"] if isinstance(body["output"], list) else ()
        (content,) = object_value(item)["content"] if isinstance(object_value(item)["content"], list) else ()
        return string_value(body["id"]), string_value(object_value(content)["text"])
    (block,) = body["content"] if isinstance(body["content"], list) else ()
    return string_value(body["id"]), string_value(object_value(block)["text"])


def _stream_identity_and_text(route: str, lines: Sequence[str]) -> tuple[str, str]:
    frames: Final = _data_frames(lines)
    if route == "/v1/chat/completions":
        return _chat_stream_identity_and_text(frames)
    if route == "/v1/responses":
        return _responses_stream_identity_and_text(frames)
    return _messages_stream_identity_and_text(frames)


async def _send(client: httpx.AsyncClient, route: str, model: str, prompt: str, *, stream: bool) -> _Served:
    body: Final = _body(route, model, prompt, stream=stream)
    headers: Final = dict(_ANTHROPIC_VERSION) if route == "/v1/messages" else {}
    if not stream:
        response: Final = await client.post(route, json=body, headers=headers)
        assert response.status_code == 200, response.text
        identity, text = _identity_and_text(route, _JSON_OBJECT.validate_json(response.content))
        return _Served(
            route=route,
            stream=False,
            status=response.status_code,
            identity=identity,
            text=text,
            cause=response.headers.get("x-litellm-complexity-router-cause", ""),
            classifier_cost=response.headers.get("x-litellm-classifier-cost"),
        )
    async with client.stream("POST", route, json=body, headers=headers) as streamed:
        lines: Final = [line async for line in streamed.aiter_lines()]
        assert streamed.status_code == 200, lines
        identity, text = _stream_identity_and_text(route, lines)
        return _Served(
            route=route,
            stream=True,
            status=streamed.status_code,
            identity=identity,
            text=text,
            cause=streamed.headers.get("x-litellm-complexity-router-cause", ""),
            classifier_cost=streamed.headers.get("x-litellm-classifier-cost"),
        )


def _mixed_calls(count: int) -> tuple[tuple[str, bool], ...]:
    routes: Final = tuple(_CALL_TYPES)
    return tuple((routes[index % len(routes)], index % 2 == 1) for index in range(count))


async def _burst(
    base_url: str,
    key: str,
    model: str,
    calls: Sequence[tuple[str, bool]],
    prompts: Sequence[str],
    *,
    tolerate_transport_errors: bool = False,
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(
        base_url=base_url, timeout=90, trust_env=False, headers={"Authorization": f"Bearer {key}"}
    ) as client:
        results: Final = await asyncio.gather(
            *(_send(client, route, model, prompt, stream=stream) for (route, stream), prompt in zip(calls, prompts)),
            return_exceptions=tolerate_transport_errors,
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _served_by_logged_id(served: Sequence[_Served]) -> Iterator[tuple[str, _Served]]:
    for item in served:
        for identity in _logged_ids(item.route, item.identity):
            yield identity, item


def _assert_each_logged_once(rows: Sequence[Mapping[str, JsonValue]], served: Sequence[_Served]) -> None:
    request_rows: Final = _request_rows(rows)
    accepted: Final = dict(_served_by_logged_id(served))
    assert all(string_value(row["request_id"]) in accepted for row in request_rows), (request_rows, accepted)
    matched: Final = tuple(accepted[string_value(row["request_id"])] for row in request_rows)
    assert sorted(item.identity for item in matched) == sorted(item.identity for item in served), (matched, served)
    for row, item in zip(request_rows, matched, strict=True):
        assert (row["call_type"], row["status"]) == (_CALL_TYPES[item.route], "success"), (item, row)


def _test_routing(gateway: Gateway, config: Mapping[str, JsonValue], prompt: str) -> dict[str, JsonValue]:
    response: Final = gateway.request(
        "POST", "/auto_router/test_routing", {"prompt": prompt, "complexity_router_config": dict(config)}
    )
    assert response.status_code == 200, response.text
    return object_value(object_value(response.json())["routing_decision"])


def _sdk_base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def test_chat_completion_over_httpx_is_classified_by_the_databricks_endpoint_with_the_system_prompt_quoted(
    gateway: Gateway,
) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": routed.name,
                "messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": prompt}],
            },
        )
        assert response.status_code == 200, response.text
        identity, text = _identity_and_text("/v1/chat/completions", _JSON_OBJECT.validate_json(response.content))
        assert text == _COMPLEX_TEXT, response.text
        _assert_classified(response.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt, system="be terse")
        (tier_call,) = _posts(complex_)
        assert tier_call.target == "/chat/completions", tier_call.target
        tier_body: Final = _JSON_OBJECT.validate_json(tier_call.body)
        assert tier_body["model"] == "complex-tier", tier_body
        assert tier_body["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": prompt},
        ], tier_body
        assert _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/chat/completions", request_id=identity)


def test_chat_completion_stream_over_httpx_is_classified_and_logged_under_the_chunk_id(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json=_body("/v1/chat/completions", routed.name, prompt, stream=True),
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as streamed:
            lines: Final = list(streamed.iter_lines())
            assert streamed.status_code == 200, lines
            _assert_classified(streamed.headers, router=routed.name)
        identity, text = _stream_identity_and_text("/v1/chat/completions", lines)
        assert text == _COMPLEX_TEXT, lines
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt)
        (tier_call,) = _posts(complex_)
        assert _JSON_OBJECT.validate_json(tier_call.body)["stream"] is True, tier_call.body
        assert _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/chat/completions", request_id=identity)


def test_chat_completion_through_the_openai_sdk_is_classified_and_logged_once(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        client: Final = openai.OpenAI(base_url=f"{_sdk_base(gateway)}/v1", api_key=gateway.key, max_retries=0)
        raw: Final = client.chat.completions.with_raw_response.create(
            model=routed.name, messages=[{"role": "user", "content": prompt}]
        )
        completion: Final = raw.parse()
        assert completion.choices[0].message.content == _COMPLEX_TEXT, completion
        _assert_classified(raw.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt)
        assert len(_posts(complex_)) == 1 and _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/chat/completions", request_id=completion.id)


async def test_chat_completion_stream_through_the_async_openai_sdk_is_classified_and_logged_once(
    gateway: Gateway,
) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        client: Final = openai.AsyncOpenAI(base_url=f"{_sdk_base(gateway)}/v1", api_key=gateway.key, max_retries=0)
        raw: Final = await client.chat.completions.with_raw_response.create(
            model=routed.name, messages=[{"role": "user", "content": prompt}], stream=True
        )
        chunks: Final = [chunk async for chunk in raw.parse()]
        identities: Final = {chunk.id for chunk in chunks}
        assert len(identities) == 1, identities
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == _COMPLEX_TEXT
        _assert_classified(raw.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt)
        assert len(_posts(complex_)) == 1 and _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/chat/completions", request_id=identities.pop())


def test_responses_request_over_httpx_is_classified_and_forwarded_to_the_tier_responses_route(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        response: Final = gateway.request("POST", "/v1/responses", {"model": routed.name, "input": prompt})
        assert response.status_code == 200, response.text
        identity, text = _identity_and_text("/v1/responses", _JSON_OBJECT.validate_json(response.content))
        assert text == _COMPLEX_TEXT, response.text
        _assert_classified(response.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt)
        (tier_call,) = _posts(complex_)
        assert tier_call.target == "/responses", tier_call.target
        tier_body: Final = _JSON_OBJECT.validate_json(tier_call.body)
        assert (tier_body["model"], tier_body["input"]) == ("complex-tier", prompt), tier_body
        assert _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/responses", request_id=identity)


async def test_responses_stream_through_the_async_openai_sdk_is_classified_and_logged_under_the_created_id(
    gateway: Gateway,
) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        client: Final = openai.AsyncOpenAI(base_url=f"{_sdk_base(gateway)}/v1", api_key=gateway.key, max_retries=0)
        raw: Final = await client.responses.with_raw_response.create(model=routed.name, input=prompt, stream=True)
        events: Final = [event async for event in raw.parse()]
        created: Final = [event for event in events if event.type == "response.created"]
        assert len(created) == 1, [event.type for event in events]
        text: Final = "".join(event.delta for event in events if event.type == "response.output_text.delta")
        assert text == _COMPLEX_TEXT, [event.type for event in events]
        _assert_classified(raw.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt)
        (tier_call,) = _posts(complex_)
        assert tier_call.target == "/responses", tier_call.target
        assert _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/responses", request_id=created[0].response.id)


def test_messages_request_through_the_anthropic_sdk_is_classified_and_bridged_to_the_tier_responses_route(
    gateway: Gateway,
) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        client: Final = anthropic.Anthropic(base_url=_sdk_base(gateway), api_key=gateway.key, max_retries=0)
        raw: Final = client.messages.with_raw_response.create(
            model=routed.name, max_tokens=64, messages=[{"role": "user", "content": prompt}]
        )
        message: Final = raw.parse()
        assert [block.text for block in message.content if block.type == "text"] == [_COMPLEX_TEXT], message
        _assert_classified(raw.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt)
        (tier_call,) = _posts(complex_)
        assert tier_call.target == "/responses", tier_call.target
        assert prompt in tier_call.body.decode(), tier_call.body
        assert _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/messages", request_id=message.id)


def test_messages_stream_over_httpx_is_classified_and_logged_under_the_message_start_id(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json=_body("/v1/messages", routed.name, prompt, stream=True),
            headers={"Authorization": f"Bearer {gateway.key}", **_ANTHROPIC_VERSION},
        ) as streamed:
            lines: Final = list(streamed.iter_lines())
            assert streamed.status_code == 200, lines
            _assert_classified(streamed.headers, router=routed.name)
        identity, text = _stream_identity_and_text("/v1/messages", lines)
        assert text == _COMPLEX_TEXT, lines
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt)
        (tier_call,) = _posts(complex_)
        assert tier_call.target == "/responses", tier_call.target
        assert _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/messages", request_id=identity)


def test_a_serving_endpoint_name_is_classified_through_its_invocations_route_and_logged_at_zero(
    gateway: Gateway,
) -> None:
    prompt: Final = _prompt()
    endpoint: Final = "my-jev-endpoint"
    with (
        _wires(_answering({"model": "/mosaicml/local_model", "answers": _ANSWERS, "usage": _USAGE})) as (
            judge,
            simple,
            complex_,
        ),
        gateway.scenario() as scenario,
    ):
        routed: Final = _deploy(scenario, judge, simple, complex_, _classifier_config(judge.url, model=endpoint))
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _body("/v1/chat/completions", routed.name, prompt, stream=False)
        )
        assert response.status_code == 200, response.text
        identity, text = _identity_and_text("/v1/chat/completions", _JSON_OBJECT.validate_json(response.content))
        assert text == _COMPLEX_TEXT, response.text
        _assert_classified(response.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt, endpoint=endpoint)
        assert len(_posts(complex_)) == 1 and _posts(simple) == ()
        rows: Final = _rows(routed.name, want=2)
        (classifier_row,) = _classifier_rows(rows)
        _assert_classifier_row(classifier_row, model=f"databricks/{endpoint}")
        (request_row,) = _request_rows(rows)
        assert (request_row["request_id"], request_row["status"]) == (identity, "success"), request_row


def test_a_response_cache_hit_still_runs_the_classifier_and_logs_the_hit_row_at_zero(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        body: Final = _body("/v1/chat/completions", routed.name, prompt, stream=False)
        first: Final = gateway.request("POST", "/v1/chat/completions", body)
        second: Final = gateway.request("POST", "/v1/chat/completions", body)
        assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
        first_identity, first_text = _identity_and_text(
            "/v1/chat/completions", _JSON_OBJECT.validate_json(first.content)
        )
        second_identity, second_text = _identity_and_text(
            "/v1/chat/completions", _JSON_OBJECT.validate_json(second.content)
        )
        assert (first_text, second_text) == (_COMPLEX_TEXT, _COMPLEX_TEXT)
        assert second_identity == first_identity, (first_identity, second_identity)
        assert "x-litellm-cache-key" not in first.headers, dict(first.headers)
        assert "x-litellm-cache-key" in second.headers, dict(second.headers)
        _assert_classified(first.headers, router=routed.name)
        _assert_classified(second.headers, router=routed.name)
        classifier_calls: Final = _posts(judge)
        assert len(classifier_calls) == 2, classifier_calls
        for call in classifier_calls:
            _assert_classifier_call(call, prompt=prompt)
        assert len(_posts(complex_)) == 1 and _posts(simple) == ()
        rows: Final = _rows(routed.name, want=4)
        classifier_rows: Final = _classifier_rows(rows)
        assert len(classifier_rows) == 2, rows
        for row in classifier_rows:
            _assert_classifier_row(row)
        request_rows: Final = _request_rows(rows)
        assert [row["request_id"] for row in request_rows if row["cache_hit"] != "True"] == [first_identity], rows
        (hit_row,) = [row for row in request_rows if row["cache_hit"] == "True"]
        assert string_value(hit_row["request_id"]).startswith(f"{first_identity}_cache_hit"), hit_row
        assert (hit_row["status"], _number(hit_row["spend"])) == ("success", 0.0), hit_row


_FAILURES: Final = (
    _Failure("500", _failing(500), {}, "classifier_error", "MaskedHTTPStatusError", 0),
    _Failure("401", _failing(401), {}, "classifier_error", "MaskedHTTPStatusError", 0),
    _Failure(
        "malformed",
        _answering({"response": {"answers": {"tier": {"type": "choice", "choice": "COMPLEX", "confidence": 0.91}}}}),
        {},
        "invalid_response",
        "ValidationError",
        1,
    ),
    _Failure(
        "unknown-tier",
        _answering(
            {
                "response": {
                    "answers": {
                        "tier": {
                            "type": "choice",
                            "choice": "GALAXY",
                            "confidence": 0.91,
                            "probabilities": {"GALAXY": 0.91},
                        }
                    }
                },
                "metadata": _METADATA,
            }
        ),
        {},
        "classifier_error",
        "ValueError",
        1,
    ),
    _Failure("timeout", _slow(3), {"timeout_ms": 1000}, "timeout", "TimeoutError", 0),
)


@pytest.mark.parametrize("failure", _FAILURES, ids=[failure.label for failure in _FAILURES])
def test_a_failing_databricks_classifier_falls_back_to_the_heuristic_and_names_the_failure(
    gateway: Gateway, failure: _Failure
) -> None:
    prompt: Final = _short_prompt()
    with _wires(failure.respond) as (judge, simple, complex_), gateway.scenario() as scenario:
        classifier: Final = _classifier_config(judge.url, **failure.overrides)
        routed: Final = _deploy(scenario, judge, simple, complex_, classifier)
        decision: Final = _test_routing(gateway, _router_config(classifier, routed.simple, routed.complex_), prompt)
        assert (
            decision["cause"],
            decision["tier"],
            decision["routed_model"],
            decision["classifier_failure_reason"],
            decision["classifier_error_type"],
        ) == ("heuristic_scorer", "SIMPLE", routed.simple, failure.reason, failure.error_type), decision
        assert "classifier_cost" not in decision, decision
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _body("/v1/chat/completions", routed.name, prompt, stream=False)
        )
        assert response.status_code == 200, response.text
        identity, text = _identity_and_text("/v1/chat/completions", _JSON_OBJECT.validate_json(response.content))
        assert text == _SIMPLE_TEXT, response.text
        _assert_fell_back(response.headers, router=routed.name)
        classifier_calls: Final = _posts(judge)
        assert len(classifier_calls) == 2, classifier_calls
        for call in classifier_calls:
            _assert_classifier_call(call, prompt=prompt)
        assert len(_posts(simple)) == 1 and _posts(complex_) == ()
        rows: Final = _rows(routed.name, want=1 + failure.classifier_rows)
        (request_row,) = _request_rows(rows)
        assert (request_row["request_id"], request_row["call_type"], request_row["status"]) == (
            identity,
            "acompletion",
            "success",
        ), request_row
        classifier_rows: Final = _classifier_rows(rows)
        assert len(classifier_rows) == failure.classifier_rows, rows
        for row in classifier_rows:
            _assert_classifier_row(row)


def test_a_five_kilobyte_prompt_with_a_system_message_reaches_the_classifier_whole(gateway: Gateway) -> None:
    prompt: Final = "p" * 5000 + uuid.uuid4().hex
    with _wires() as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": routed.name,
                "messages": [{"role": "system", "content": "answer in one line"}, {"role": "user", "content": prompt}],
            },
        )
        assert response.status_code == 200, response.text[:400]
        identity, text = _identity_and_text("/v1/chat/completions", _JSON_OBJECT.validate_json(response.content))
        assert text == _COMPLEX_TEXT, response.text[:400]
        _assert_classified(response.headers, router=routed.name)
        (classifier_call,) = _posts(judge)
        _assert_classifier_call(classifier_call, prompt=prompt, system="answer in one line")
        assert len(_posts(complex_)) == 1 and _posts(simple) == ()
        _assert_logged_once(_rows(routed.name, want=2), route="/v1/chat/completions", request_id=identity)


def _typesafe_cost() -> float:
    prices: Final = object_value(
        json.loads(Path("model_prices_and_context_window.json").read_text())[f"typesafe/{_TYPESAFE_MODEL}"]
    )
    return _number(_USAGE["input_tokens"]) * _number(prices["input_cost_per_token"]) + _number(
        _USAGE["output_tokens"]
    ) * _number(prices["output_cost_per_token"])


def test_the_default_jev_provider_still_posts_to_systemone_and_bills_from_the_cost_map(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    expected_cost: Final = _typesafe_cost()
    with (
        _wires(_answering(_TYPESAFE_ANSWER)) as (judge, simple, complex_),
        gateway.scenario() as scenario,
    ):
        classifier: Final[dict[str, JsonValue]] = {
            "model": _TYPESAFE_MODEL,
            "api_base": judge.url,
            "api_key": _TYPESAFE_KEY,
            "timeout_ms": 20000,
            "circuit_breaker_enabled": False,
        }
        routed: Final = _deploy(scenario, judge, simple, complex_, classifier)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _body("/v1/chat/completions", routed.name, prompt, stream=False)
        )
        assert response.status_code == 200, response.text
        identity, text = _identity_and_text("/v1/chat/completions", _JSON_OBJECT.validate_json(response.content))
        assert text == _COMPLEX_TEXT, response.text
        assert (
            response.headers["x-litellm-model-name"],
            response.headers["x-litellm-complexity-router-tier"],
            response.headers["x-litellm-complexity-router-cause"],
        ) == ("openai/complex-tier", "COMPLEX", "jev_classifier"), dict(response.headers)
        assert float(response.headers["x-litellm-classifier-cost"]) == pytest.approx(expected_cost)
        (classifier_call,) = _posts(judge)
        assert classifier_call.target == "/v1/systemone", classifier_call.target
        assert classifier_call.headers.get("authorization") == f"Bearer {_TYPESAFE_KEY}", classifier_call.headers
        assert prompt in classifier_call.body.decode(), classifier_call.body
        assert len(_posts(complex_)) == 1 and _posts(simple) == ()
        rows: Final = _rows(routed.name, want=2)
        (classifier_row,) = _classifier_rows(rows)
        _assert_classifier_row(
            classifier_row, model=f"typesafe/{_TYPESAFE_MODEL}", provider="typesafe", spend=expected_cost
        )
        (request_row,) = _request_rows(rows)
        assert (request_row["request_id"], request_row["status"]) == (identity, "success"), request_row


async def test_a_classifier_outage_mid_traffic_falls_back_and_recovers_with_every_request_logged_once(
    gateway: Gateway,
) -> None:
    base_url: Final = _sdk_base(gateway)
    calls: Final = _mixed_calls(12)
    with (
        wire_server(_tier("simple", _SIMPLE_TEXT)) as simple,
        wire_server(_tier("complex", _COMPLEX_TEXT)) as complex_,
        gateway.scenario() as scenario,
    ):
        with wire_server(_answering(_ANSWER)) as judge:
            routed: Final = _deploy(scenario, judge, simple, complex_)
            config: Final = _router_config(_classifier_config(judge.url), routed.simple, routed.complex_)
            port: Final = urlsplit(judge.url).port
            assert port is not None
            before_prompts: Final = tuple(_prompt() for _ in calls)
            before: Final = await _burst(base_url, gateway.key, routed.name, calls, before_prompts)
            assert [item.cause for item in before] == ["jev_classifier"] * 12, before
            assert [item.classifier_cost for item in before] == [None] * 12, before
            assert [item.text for item in before] == [_COMPLEX_TEXT] * 12, before
            assert sorted(json.loads(call.body)["state"] for call in _posts(judge)) == sorted(
                _state(prompt, None) for prompt in before_prompts
            )
        during_prompts: Final = tuple(_short_prompt() for _ in calls)
        during: Final = await _burst(base_url, gateway.key, routed.name, calls, during_prompts)
        assert [item.cause for item in during] == ["heuristic_scorer"] * 12, during
        assert [item.classifier_cost for item in during] == [None] * 12, during
        assert [item.text for item in during] == [_SIMPLE_TEXT] * 12, during
        decision: Final = _test_routing(gateway, config, _short_prompt())
        assert (decision["cause"], decision["classifier_failure_reason"]) == (
            "heuristic_scorer",
            "classifier_error",
        ), decision
        with wire_server(_answering(_ANSWER), port=port) as recovered:
            after_prompts: Final = tuple(_prompt() for _ in calls)
            after: Final = await _burst(base_url, gateway.key, routed.name, calls, after_prompts)
            assert [item.cause for item in after] == ["jev_classifier"] * 12, after
            assert [item.text for item in after] == [_COMPLEX_TEXT] * 12, after
            assert sorted(json.loads(call.body)["state"] for call in _posts(recovered)) == sorted(
                _state(prompt, None) for prompt in after_prompts
            )
        served: Final = (*before, *during, *after)
        assert len({item.identity for item in served}) == 36, served
        rows: Final = _rows(routed.name, want=60)
        _assert_each_logged_once(rows, served)
        classifier_rows: Final = _classifier_rows(rows)
        assert len(classifier_rows) == 24, rows
        for row in classifier_rows:
            _assert_classifier_row(row)


async def test_a_slow_classifier_under_a_burst_times_out_every_call_into_the_heuristic_without_classifier_rows(
    gateway: Gateway,
) -> None:
    base_url: Final = _sdk_base(gateway)
    calls: Final = tuple(("/v1/chat/completions", False) for _ in range(20))
    prompts: Final = tuple(_short_prompt() for _ in calls)
    with _wires(_slow(3)) as (judge, simple, complex_), gateway.scenario() as scenario:
        routed: Final = _deploy(scenario, judge, simple, complex_, _classifier_config(judge.url, timeout_ms=500))
        served: Final = await _burst(base_url, gateway.key, routed.name, calls, prompts)
        assert [item.cause for item in served] == ["heuristic_scorer"] * 20, served
        assert [item.text for item in served] == [_SIMPLE_TEXT] * 20, served
        assert len({item.identity for item in served}) == 20, served
        seen: Final[list[Request]] = []

        def drained() -> int:
            seen.extend(_posts(judge))
            return len(seen)

        eventually(drained, lambda count: count == 20, seconds=10)
        assert sorted(json.loads(call.body)["state"] for call in seen) == sorted(
            _state(prompt, None) for prompt in prompts
        )
        assert len(_posts(simple)) == 20 and _posts(complex_) == ()
        rows: Final = _rows(routed.name, want=20)
        _assert_each_logged_once(rows, served)
        assert _classifier_rows(rows) == (), rows


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    return tuple(int(pid) for pid in _STARTED_WORKER.findall(text)), text.count("Application startup complete.")


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _chaos_config(judge: Wire, simple: Wire, complex_: Wire, tmp_path: Path, router: str) -> Path:
    config: Final = {
        **_JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())),
        "model_list": [
            {
                "model_name": "chaos-simple",
                "litellm_params": {
                    "model": "openai/simple-tier",
                    "api_base": simple.url,
                    "api_key": "synthetic-tier-key",
                },
            },
            {
                "model_name": "chaos-complex",
                "litellm_params": {
                    "model": "openai/complex-tier",
                    "api_base": complex_.url,
                    "api_key": "synthetic-tier-key",
                },
            },
            {
                "model_name": router,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": _router_config(
                        _classifier_config(judge.url, timeout_ms=60000), "chaos-simple", "chaos-complex"
                    ),
                },
            },
        ],
    }
    path: Final = tmp_path / "databricks-classifier-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120 + 180))
async def test_worker_sigkill_mid_burst_leaves_the_sibling_classifying_through_the_databricks_endpoint(
    gateway: Gateway, tmp_path: Path
) -> None:
    router: Final = f"chaos-router-{uuid.uuid4().hex[:8]}"
    calls: Final = tuple(("/v1/chat/completions", False) for _ in range(20))
    prompts: Final = tuple(_prompt() for _ in calls)
    release: Final = threading.Event()
    held_states: Final[SimpleQueue[str]] = SimpleQueue()
    answer: Final = _answering(_ANSWER)

    def held(request: Request) -> Reply:
        held_states.put(string_value(_JSON_OBJECT.validate_json(request.body)["state"]))
        assert release.wait(timeout=120), "The burst was never released"
        return answer(request)

    with (
        wire_server(held) as judge,
        wire_server(_tier("simple", _SIMPLE_TEXT)) as simple,
        wire_server(_tier("complex", _COMPLEX_TEXT)) as complex_,
    ):
        path: Final = _chaos_config(judge, simple, complex_, tmp_path, router)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            base_url: Final = str(candidate.client.base_url)
            workers, _ = eventually(
                lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 2 and found[1] == 2, seconds=120
            )
            burst: Final = asyncio.create_task(
                _burst(base_url, candidate.key, router, calls, prompts, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held_states.qsize, lambda size: size == 20, 90)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, judge.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            assert [item.cause for item in served] == ["jev_classifier"] * len(served), served
            follow_up_prompt: Final = _prompt()
            (answered,) = await _burst(
                base_url, candidate.key, router, (("/v1/chat/completions", False),), (follow_up_prompt,)
            )
            assert (answered.cause, answered.text) == ("jev_classifier", _COMPLEX_TEXT), answered
            await asyncio.to_thread(
                eventually, lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 3 and found[1] == 3, 180
            )
            everything: Final = (*served, answered)
            assert len({item.identity for item in everything}) == len(everything), everything
            rows: Final = _rows(router, want=2 * len(everything))
            _assert_each_logged_once(rows, everything)
            classifier_rows: Final = _classifier_rows(rows)
            assert len(classifier_rows) == len(everything), rows
            for row in classifier_rows:
                _assert_classifier_row(row)
            states: Final[set[str]] = set()
            while not held_states.empty():
                states.add(held_states.get())
            assert states == {_state(prompt, None) for prompt in (*prompts, follow_up_prompt)}, states
