import asyncio
import json
import signal
import threading
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
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
from anthropic.types import RawContentBlockDeltaEvent, TextDelta, WebSearchTool20250305Param
from integration._support import interception_vendor as iv
from integration._support import responses_vendor as rv
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk, ChatCompletionFunctionToolParam, ChatCompletionUserMessageParam
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent, WebSearchToolParam
from pydantic import JsonValue

_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
pytestmark: Final = pytest.mark.timeout(_CELL_SECONDS)

_PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_NO_CACHE: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType({"include_usage": True})
_OBFUSCATION: Final[Mapping[str, JsonValue]] = MappingProxyType({"include_obfuscation": True})
_PARAMETERS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
)
_SEARCH_FUNCTION: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "function",
        "function": {"name": iv.SEARCH_TOOL, "description": "Search the web", "parameters": dict(_PARAMETERS)},
    }
)
_CODE_INTERPRETER: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "code_interpreter", "container": {"type": "auto"}}
)
_HEADROOM_RETRIEVE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "function",
        "function": {"name": "headroom_retrieve", "description": "Retrieve content", "parameters": dict(_PARAMETERS)},
    }
)
_RESPONSES_SEARCH: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": "web_search"})
_BURST_MODEL: Final = f"interception-burst-{uuid.uuid4().hex}"
_BURST_DEPLOYMENT: Final = iv.deployment("search")
_STARTED_WORKER: Final = "Started server process ["
_MALFORMED: Final = (
    pytest.param(5, id="int"),
    pytest.param([{"include_usage": True}], id="list"),
    pytest.param("", id="empty-string"),
    pytest.param("x" * 5120, id="5kb-string"),
)


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway

    @property
    def base_url(self) -> str:
        return str(self.proxy.gateway.client.base_url).rstrip("/")


@dataclass(frozen=True, slots=True)
class _Deployment:
    model: str
    name: str


@dataclass(frozen=True, slots=True)
class _Stream:
    status: int
    text: str

    @property
    def frames(self) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(
            iv.JSON_OBJECT.validate_json(line[6:]) for line in self.text.splitlines() if line.startswith("data: {")
        )

    @property
    def content(self) -> str:
        return "".join(_delta_text(frame) for frame in self.frames)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(string_value(frame["id"]) for frame in self.frames)

    @property
    def usages(self) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(object_value(frame["usage"]) for frame in self.frames if frame.get("usage") is not None)

    @property
    def deltas(self) -> str:
        return "".join(
            string_value(frame["delta"]) for frame in self.frames if frame.get("type") == "response.output_text.delta"
        )

    @property
    def completed(self) -> Mapping[str, JsonValue]:
        (frame,) = [frame for frame in self.frames if frame.get("type") == "response.completed"]
        return object_value(frame["response"])


def _delta_text(frame: Mapping[str, JsonValue]) -> str:
    choices: Final = rv.ITEMS.validate_python(frame.get("choices") or [])
    return "".join(_text_of(object_value(choice.get("delta") or {})) for choice in choices)


def _text_of(delta: Mapping[str, JsonValue]) -> str:
    content: Final = delta.get("content")
    return content if isinstance(content, str) else ""


def _config(
    directory: Path,
    wire: Wire,
    general_settings: Mapping[str, JsonValue],
    models: Sequence[Mapping[str, JsonValue]] = (),
) -> Path:
    base: Final = object_value(yaml.safe_load(_PROXY_CONFIG.read_text()))
    config: Final = {
        **base,
        "model_list": [*rv.ITEMS.validate_python(base["model_list"]), *(dict(model) for model in models)],
        "general_settings": {**object_value(base["general_settings"]), **general_settings},
        "litellm_settings": {
            **object_value(base["litellm_settings"]),
            "callbacks": ["websearch_interception", "code_interpreter_interception"],
            "websearch_interception_params": {"enabled_providers": ["azure"], "search_tool_name": "integration-search"},
            "code_interpreter_interception_params": {"enabled_providers": ["azure"]},
        },
        "search_tools": [
            {
                "search_tool_name": "integration-search",
                "litellm_params": {
                    "search_provider": "tavily",
                    "api_key": iv.TAVILY_KEY,
                    "api_base": f"{wire.url}/tavily",
                },
            }
        ],
        "guardrails": [
            {
                "guardrail_name": "integration-headroom",
                "litellm_params": {
                    "guardrail": "headroom",
                    "mode": "post_call",
                    "api_base": "http://127.0.0.1:9",
                    "unreachable_fallback": "fail_open",
                    "default_on": False,
                },
            }
        ],
    }
    path: Final = directory / f"interception-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _interception_proxy(
    directory: Path,
    respond: Callable[[Request], Reply],
    general_settings: Mapping[str, JsonValue],
    models: Callable[[Wire], Sequence[Mapping[str, JsonValue]]] = lambda _: (),
) -> Generator[_Rig, None, None]:
    with gateway_from_environment() as gateway, wire_server(respond) as wire:
        config: Final = _config(directory, wire, general_settings, models(wire))
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            yield _Rig(wire, owned)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    with _interception_proxy(tmp_path_factory.mktemp("interception-rig"), iv.respond, {}) as built:
        yield built


@pytest.fixture
def scenario(rig: _Rig) -> Iterator[Scenario]:
    rig.wire.drain()
    with rig.gateway.scenario() as opened:
        yield opened


def _deploy(rig: _Rig, scenario: Scenario, mode: iv.Mode, *, api_key: str = iv.AZURE_KEY) -> _Deployment:
    name: Final = iv.deployment(mode)
    model: Final = scenario.model(
        model=f"azure/{name}", api_base=rig.wire.url, api_key=api_key, api_version=iv.API_VERSION
    )
    return _Deployment(model, name)


def _chat_body(
    deployment: _Deployment,
    *,
    stream: bool = True,
    tool: Mapping[str, JsonValue] = _SEARCH_FUNCTION,
    extra: Mapping[str, JsonValue] = _NO_CACHE,
) -> Mapping[str, JsonValue]:
    return {
        "model": deployment.model,
        "messages": [{"role": "user", "content": f"Search for {deployment.name}"}],
        "tools": [dict(tool)],
        "stream": stream,
        **extra,
    }


def _responses_body(deployment: _Deployment, extra: Mapping[str, JsonValue] = _NO_CACHE) -> Mapping[str, JsonValue]:
    return {
        "model": deployment.model,
        "input": f"Search for {deployment.name}",
        "tools": [dict(_RESPONSES_SEARCH)],
        "stream": True,
        **extra,
    }


def _post(rig: _Rig, path: str, body: Mapping[str, JsonValue], *, key: str | None = None) -> _Stream:
    response: Final = rig.gateway.request("POST", path, body, key=key)
    return _Stream(response.status_code, response.text)


def _chat(rig: _Rig, body: Mapping[str, JsonValue]) -> _Stream:
    return _post(rig, "/v1/chat/completions", body)


def _responses(rig: _Rig, body: Mapping[str, JsonValue]) -> _Stream:
    return _post(rig, "/v1/responses", body)


def _upstream(rig: _Rig, deployment: _Deployment) -> iv.Received:
    return iv.received(rig.wire.drain(), deployment.name)


def _leaks(call: Mapping[str, JsonValue]) -> bool:
    return "stream_options" in call or call.get("stream") is True or any(key.startswith("_") for key in call)


def _assert_upstream(received: iv.Received, *, model_calls: int, searches: int) -> None:
    assert len(received.model_calls) == model_calls, received
    assert len(received.searches) == searches, received
    assert [call for call in received.model_calls if _leaks(call)] == [], received.model_calls


def _totals(usage: Mapping[str, JsonValue]) -> int:
    total: Final = usage["total_tokens"]
    assert isinstance(total, int), usage
    return total


def _assert_restreamed(stream: _Stream, text: str, *, usage: Mapping[str, int] | None) -> str:
    assert stream.status == 200, stream.text
    assert stream.content == text, stream.text
    assert [_totals(item) for item in stream.usages] == ([] if usage is None else [usage["total_tokens"]]), stream.text
    (identity,) = stream.ids
    return identity


def _assert_responses_restreamed(stream: _Stream, text: str) -> str:
    assert stream.status == 200, stream.text
    assert stream.deltas == text, stream.text
    completed: Final = stream.completed
    assert _totals(object_value(completed["usage"])) == iv.RESPONSES_FOLLOWUP_USAGE["total_tokens"], completed
    return string_value(completed["id"])


def _spend_rows(model: str, expected: int) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        eventually(
            lambda: read_rows(
                'SELECT request_id, status, cache_hit FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model,)
            ),
            lambda rows: sum(1 for row in rows if row["status"] == "success") >= expected,
            seconds=70,
        )
    )


def _assert_landed_once(model: str, identities: frozenset[str], *, rows: int) -> tuple[Mapping[str, JsonValue], ...]:
    landed: Final = _spend_rows(model, rows)
    assert len(landed) == rows, landed
    assert frozenset(string_value(row["request_id"]) for row in landed if row["status"] == "success") == identities, (
        landed
    )
    return landed


def _assert_response_landed_once(model: str, identity: str) -> None:
    (row,) = _spend_rows(model, 1)
    assert row["status"] == "success" and rv.same_response(string_value(row["request_id"]), identity), row


def _openai(rig: _Rig) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{rig.base_url}/v1", api_key=rig.gateway.key, max_retries=0)


def _async_openai(rig: _Rig) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=f"{rig.base_url}/v1", api_key=rig.gateway.key, max_retries=0)


def _sdk_tool() -> ChatCompletionFunctionToolParam:
    return ChatCompletionFunctionToolParam(
        type="function",
        function={"name": iv.SEARCH_TOOL, "description": "Search the web", "parameters": dict(_PARAMETERS)},
    )


def _sdk_message(deployment: _Deployment) -> ChatCompletionUserMessageParam:
    return ChatCompletionUserMessageParam(role="user", content=f"Search for {deployment.name}")


def _chunk_text(chunk: ChatCompletionChunk) -> str:
    return "".join(choice.delta.content or "" for choice in chunk.choices)


def _assert_sdk_chunks(chunks: Sequence[ChatCompletionChunk], deployment: _Deployment) -> str:
    assert "".join(map(_chunk_text, chunks)) == iv.answer(deployment.name), chunks
    usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
    assert [usage.total_tokens for usage in usages] == [iv.FOLLOWUP_USAGE["total_tokens"]], chunks
    (identity,) = {chunk.id for chunk in chunks}
    return identity


def _assert_search_turn(rig: _Rig, deployment: _Deployment, identity: str) -> None:
    received: Final = _upstream(rig, deployment)
    _assert_upstream(received, model_calls=2, searches=1)
    assert [search["query"] for search in received.searches] == [iv.query(deployment.name)], received.searches
    _assert_landed_once(deployment.model, frozenset({identity}), rows=1)


def test_openai_sdk_chat_stream_through_web_search_is_restreamed_with_its_usage_chunk(
    rig: _Rig, scenario: Scenario
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    chunks: Final = tuple(
        _openai(rig).chat.completions.create(
            model=deployment.model,
            messages=[_sdk_message(deployment)],
            tools=[_sdk_tool()],
            stream=True,
            stream_options={"include_usage": True},
            extra_body=dict(_NO_CACHE),
        )
    )
    _assert_search_turn(rig, deployment, _assert_sdk_chunks(chunks, deployment))


async def test_async_openai_sdk_chat_stream_through_web_search_is_restreamed_with_its_usage_chunk(
    rig: _Rig, scenario: Scenario
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    stream: Final = await _async_openai(rig).chat.completions.create(
        model=deployment.model,
        messages=[_sdk_message(deployment)],
        tools=[_sdk_tool()],
        stream=True,
        stream_options={"include_usage": True},
        extra_body=dict(_NO_CACHE),
    )
    chunks: Final = tuple([chunk async for chunk in stream])
    _assert_search_turn(rig, deployment, _assert_sdk_chunks(chunks, deployment))


def test_raw_chat_stream_without_stream_options_answers_without_a_usage_chunk(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    identity: Final = _assert_restreamed(_chat(rig, _chat_body(deployment)), iv.answer(deployment.name), usage=None)
    _assert_search_turn(rig, deployment, identity)


def test_openai_sdk_responses_stream_through_web_search_keeps_stream_options_off_azure(
    rig: _Rig, scenario: Scenario
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    events: Final = tuple(
        _openai(rig).responses.create(
            model=deployment.model,
            input=f"Search for {deployment.name}",
            tools=[WebSearchToolParam(type="web_search")],
            stream=True,
            stream_options={"include_obfuscation": True},
            extra_body=dict(_NO_CACHE),
        )
    )
    text: Final = "".join(event.delta for event in events if isinstance(event, ResponseTextDeltaEvent))
    (completed,) = [event for event in events if isinstance(event, ResponseCompletedEvent)]
    assert text == iv.answer(deployment.name), events
    usage: Final = completed.response.usage
    assert usage is not None and usage.total_tokens == iv.RESPONSES_FOLLOWUP_USAGE["total_tokens"], completed
    _assert_upstream(_upstream(rig, deployment), model_calls=2, searches=1)
    _assert_response_landed_once(deployment.model, completed.response.id)


async def test_async_openai_sdk_responses_stream_through_web_search_keeps_stream_options_off_azure(
    rig: _Rig, scenario: Scenario
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    stream: Final = await _async_openai(rig).responses.create(
        model=deployment.model,
        input=f"Search for {deployment.name}",
        tools=[WebSearchToolParam(type="web_search")],
        stream=True,
        stream_options={"include_obfuscation": True},
        extra_body=dict(_NO_CACHE),
    )
    events: Final = tuple([event async for event in stream])
    text: Final = "".join(event.delta for event in events if isinstance(event, ResponseTextDeltaEvent))
    (completed,) = [event for event in events if isinstance(event, ResponseCompletedEvent)]
    assert text == iv.answer(deployment.name), events
    _assert_upstream(_upstream(rig, deployment), model_calls=2, searches=1)
    _assert_response_landed_once(deployment.model, completed.response.id)


def test_raw_responses_stream_without_stream_options_answers(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    identity: Final = _assert_responses_restreamed(
        _responses(rig, _responses_body(deployment)), iv.answer(deployment.name)
    )
    _assert_upstream(_upstream(rig, deployment), model_calls=2, searches=1)
    _assert_response_landed_once(deployment.model, identity)


def test_openai_sdk_chat_without_streaming_through_web_search_answers_in_json(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    completion: Final = _openai(rig).chat.completions.create(
        model=deployment.model,
        messages=[_sdk_message(deployment)],
        tools=[_sdk_tool()],
        extra_body=dict(_NO_CACHE),
    )
    assert completion.choices[0].message.content == iv.answer(deployment.name), completion
    assert completion.usage is not None, completion
    _assert_search_turn(rig, deployment, completion.id)


@pytest.mark.parametrize(
    ("tool", "mode"),
    (
        pytest.param(_CODE_INTERPRETER, "plain", id="code-interpreter"),
        pytest.param(_HEADROOM_RETRIEVE, "plain", id="headroom-retrieve"),
        pytest.param(_SEARCH_FUNCTION, "plain", id="web-search-unused"),
    ),
)
def test_converted_chat_stream_with_usage_replays_the_single_call_usage(
    rig: _Rig, scenario: Scenario, tool: Mapping[str, JsonValue], mode: iv.Mode
) -> None:
    deployment: Final = _deploy(rig, scenario, mode)
    stream: Final = _chat(rig, _chat_body(deployment, tool=tool, extra={**_NO_CACHE, "stream_options": dict(_USAGE)}))
    identity: Final = _assert_restreamed(stream, iv.plain(deployment.name), usage=iv.FIRST_USAGE)
    _assert_upstream(_upstream(rig, deployment), model_calls=1, searches=0)
    _assert_landed_once(deployment.model, frozenset({identity}), rows=1)


def test_anthropic_sdk_messages_stream_with_only_web_search_on_azure_answers_from_the_search(
    rig: _Rig, scenario: Scenario
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    prompt: Final = f"Search for {deployment.name}"
    client: Final = anthropic.Anthropic(base_url=rig.base_url, api_key=rig.gateway.key, max_retries=0)
    events: Final = tuple(
        client.messages.create(
            model=deployment.model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
            tools=[WebSearchTool20250305Param(type="web_search_20250305", name="web_search")],
            stream=True,
            extra_body=dict(_NO_CACHE),
        )
    )
    deltas: Final = tuple(event.delta for event in events if isinstance(event, RawContentBlockDeltaEvent))
    text: Final = "".join(delta.text for delta in deltas if isinstance(delta, TextDelta))
    assert text == f"Title: title-{prompt}\nURL: https://example.test/result\nSnippet: snippet-{prompt}", events
    received: Final = _upstream(rig, deployment)
    _assert_upstream(received, model_calls=0, searches=1)
    assert [search.get("query") for search in received.searches] == [prompt], received


def test_per_request_fallback_from_a_missing_deployment_restreams_with_usage(rig: _Rig, scenario: Scenario) -> None:
    missing: Final = _deploy(rig, scenario, "missing")
    good: Final = _deploy(rig, scenario, "search")
    body: Final = _chat_body(missing, extra={**_NO_CACHE, "stream_options": dict(_USAGE), "fallbacks": [good.model]})
    stream: Final = _chat(rig, {**body, "messages": [{"role": "user", "content": "Search the web"}]})
    identity: Final = _assert_restreamed(stream, iv.answer(good.name), usage=iv.FOLLOWUP_USAGE)
    drained: Final = rig.wire.drain()
    _assert_upstream(iv.received(drained, missing.name), model_calls=1, searches=0)
    _assert_upstream(iv.received(drained, good.name), model_calls=2, searches=1)
    _assert_landed_once(good.model, frozenset({identity}), rows=1)


def test_chat_stream_cache_hit_replays_the_answer_without_calling_azure(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    body: Final = _chat_body(deployment, extra={"stream_options": dict(_USAGE)})
    first: Final = _assert_restreamed(_chat(rig, body), iv.answer(deployment.name), usage=iv.FOLLOWUP_USAGE)
    _assert_upstream(_upstream(rig, deployment), model_calls=2, searches=1)
    second: Final = _chat(rig, body)
    assert second.status == 200 and second.content == iv.answer(deployment.name), second.text
    _assert_upstream(_upstream(rig, deployment), model_calls=0, searches=0)
    rows: Final = _spend_rows(deployment.model, 2)
    served: Final = tuple(string_value(row["request_id"]) for row in rows if str(row["cache_hit"]) != "True")
    cached: Final = tuple(string_value(row["request_id"]) for row in rows if str(row["cache_hit"]) == "True")
    assert served == (first,) and len(cached) == 1 and cached[0].startswith(first), rows


def test_responses_stream_cache_hit_replays_the_answer_without_calling_azure(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    body: Final = _responses_body(deployment, {"stream_options": dict(_OBFUSCATION)})
    _assert_responses_restreamed(_responses(rig, body), iv.answer(deployment.name))
    _assert_upstream(_upstream(rig, deployment), model_calls=2, searches=1)
    second: Final = _responses(rig, body)
    assert second.status == 200 and second.deltas == iv.answer(deployment.name), second.text
    _assert_upstream(_upstream(rig, deployment), model_calls=0, searches=0)


def _with_duplicated_stream_options(body: Mapping[str, JsonValue]) -> bytes:
    return (json.dumps({**body, "stream_options": dict(_USAGE)})[:-1] + ', "stream_options": 5}').encode()


@pytest.mark.parametrize("stream_options", _MALFORMED)
def test_malformed_chat_stream_options_never_reach_azure(
    rig: _Rig, scenario: Scenario, stream_options: JsonValue
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    stream: Final = _chat(rig, _chat_body(deployment, extra={**_NO_CACHE, "stream_options": stream_options}))
    identity: Final = _assert_restreamed(stream, iv.answer(deployment.name), usage=None)
    _assert_search_turn(rig, deployment, identity)


def test_duplicated_chat_stream_options_key_never_reaches_azure(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    response: Final = rig.gateway.client.post(
        "/v1/chat/completions",
        content=_with_duplicated_stream_options(_chat_body(deployment)),
        headers={"Authorization": f"Bearer {rig.gateway.key}", "Content-Type": "application/json"},
    )
    identity: Final = _assert_restreamed(
        _Stream(response.status_code, response.text), iv.answer(deployment.name), usage=None
    )
    _assert_search_turn(rig, deployment, identity)


@pytest.mark.parametrize("stream_options", _MALFORMED)
def test_malformed_responses_stream_options_never_reach_azure(
    rig: _Rig, scenario: Scenario, stream_options: JsonValue
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    stream: Final = _responses(rig, _responses_body(deployment, {**_NO_CACHE, "stream_options": stream_options}))
    identity: Final = _assert_responses_restreamed(stream, iv.answer(deployment.name))
    _assert_upstream(_upstream(rig, deployment), model_calls=2, searches=1)
    _assert_response_landed_once(deployment.model, identity)


def test_forged_interception_keys_on_a_json_request_never_reach_azure(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    forged: Final = {
        **_NO_CACHE,
        "_websearch_interception_stream_options": dict(_USAGE),
        "_websearch_interception_converted_stream": True,
        "_headroom_interception_stream_options": dict(_USAGE),
        "_code_interpreter_interception_stream_options": dict(_USAGE),
        "max_agentic_loops": 50,
    }
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", _chat_body(deployment, stream=False, extra=forged)
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    body: Final = iv.JSON_OBJECT.validate_json(response.content)
    (choice,) = rv.ITEMS.validate_python(body["choices"])
    assert object_value(choice["message"])["content"] == iv.answer(deployment.name), body
    _assert_search_turn(rig, deployment, string_value(body["id"]))


def test_forged_garbage_stash_on_a_stream_is_replaced_by_the_real_stream_options(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "plain")
    extra: Final = {**_NO_CACHE, "stream_options": dict(_USAGE), "_code_interpreter_interception_stream_options": "x"}
    stream: Final = _chat(rig, _chat_body(deployment, tool=_CODE_INTERPRETER, extra=extra))
    identity: Final = _assert_restreamed(stream, iv.plain(deployment.name), usage=iv.FIRST_USAGE)
    _assert_upstream(_upstream(rig, deployment), model_calls=1, searches=0)
    _assert_landed_once(deployment.model, frozenset({identity}), rows=1)


def test_wrong_azure_key_surfaces_azures_401(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search", api_key="wrong-azure-key")
    stream: Final = _chat(rig, _chat_body(deployment, extra={**_NO_CACHE, "stream_options": dict(_USAGE)}))
    assert stream.status == 401, stream.text
    assert "Access denied due to invalid subscription key" in stream.text, stream.text
    assert len(_upstream(rig, deployment).model_calls) == 1


def test_failed_follow_up_call_reaches_the_caller_as_an_error(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "followup-fails")
    stream: Final = _chat(rig, _chat_body(deployment, extra={**_NO_CACHE, "stream_options": dict(_USAGE)}))
    assert stream.status == 200, stream.text
    assert iv.FOLLOWUP_FAILURE in stream.text and stream.content == "", stream.text
    received: Final = _upstream(rig, deployment)
    _assert_upstream(received, model_calls=2, searches=1)
    assert rig.gateway.chat(_deploy(rig, scenario, "plain").model)["object"] == "chat.completion"


def test_search_outage_still_restreams_the_follow_up_answer(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search-down")
    stream: Final = _chat(rig, _chat_body(deployment, extra={**_NO_CACHE, "stream_options": dict(_USAGE)}))
    identity: Final = _assert_restreamed(stream, iv.answer(deployment.name), usage=iv.FOLLOWUP_USAGE)
    _assert_search_turn(rig, deployment, identity)


def test_unknown_virtual_key_is_refused_before_azure(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    body: Final = _chat_body(deployment, extra={**_NO_CACHE, "stream_options": dict(_USAGE)})
    stream: Final = _post(rig, "/v1/chat/completions", body, key=f"sk-{uuid.uuid4().hex}")
    assert stream.status == 401, stream.text
    assert _upstream(rig, deployment).model_calls == ()


def test_json_request_with_its_own_stream_options_still_gets_azures_refusal(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    body: Final = _chat_body(deployment, stream=False, extra={**_NO_CACHE, "stream_options": dict(_USAGE)})
    response: Final = rig.gateway.request("POST", "/v1/chat/completions", body)
    assert response.status_code == 400, response.text
    assert iv.STREAM_OPTIONS_REFUSAL in response.text, response.text
    assert [call.get("stream_options") for call in _upstream(rig, deployment).model_calls] == [dict(_USAGE)]


@pytest.mark.parametrize(
    "stream_options",
    (
        pytest.param({"include_usage": False}, id="include-usage-false"),
        pytest.param(None, id="null"),
        pytest.param({}, id="empty"),
    ),
)
def test_chat_stream_without_a_usage_request_answers_without_a_usage_chunk(
    rig: _Rig, scenario: Scenario, stream_options: JsonValue
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    stream: Final = _chat(rig, _chat_body(deployment, extra={**_NO_CACHE, "stream_options": stream_options}))
    identity: Final = _assert_restreamed(stream, iv.answer(deployment.name), usage=None)
    _assert_search_turn(rig, deployment, identity)


@pytest.mark.parametrize("stream_options", (pytest.param(None, id="null"), pytest.param({}, id="empty")))
def test_responses_stream_with_empty_stream_options_answers(
    rig: _Rig, scenario: Scenario, stream_options: JsonValue
) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    stream: Final = _responses(rig, _responses_body(deployment, {**_NO_CACHE, "stream_options": stream_options}))
    identity: Final = _assert_responses_restreamed(stream, iv.answer(deployment.name))
    _assert_upstream(_upstream(rig, deployment), model_calls=2, searches=1)
    _assert_response_landed_once(deployment.model, identity)


def test_repeated_identical_streams_each_land_their_own_row(rig: _Rig, scenario: Scenario) -> None:
    deployment: Final = _deploy(rig, scenario, "search")
    body: Final = _chat_body(deployment, extra={**_NO_CACHE, "stream_options": dict(_USAGE)})
    identities: Final = frozenset(
        _assert_restreamed(_chat(rig, body), iv.answer(deployment.name), usage=iv.FOLLOWUP_USAGE) for _ in range(3)
    )
    assert len(identities) == 3, identities
    _assert_upstream(_upstream(rig, deployment), model_calls=6, searches=3)
    _assert_landed_once(deployment.model, identities, rows=3)


@dataclass(frozen=True, slots=True)
class _Served:
    path: str
    stream: _Stream


async def _send(client: httpx.AsyncClient, key: str, path: str, body: Mapping[str, JsonValue]) -> _Served:
    response: Final = await client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
    return _Served(path, _Stream(response.status_code, response.text))


async def _burst(
    base_url: str, key: str, requests: Sequence[tuple[str, Mapping[str, JsonValue]]], *, tolerate: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, path, body) for path, body in requests), return_exceptions=tolerate
        )
    assert [result for result in results if isinstance(result, BaseException)] == [] or tolerate, results
    assert all(isinstance(result, _Served | httpx.TransportError) for result in results), results
    return tuple(result for result in results if isinstance(result, _Served))


def _chat_request(deployment: _Deployment) -> tuple[str, Mapping[str, JsonValue]]:
    return "/v1/chat/completions", _chat_body(deployment, extra={**_NO_CACHE, "stream_options": dict(_USAGE)})


def _responses_request(deployment: _Deployment) -> tuple[str, Mapping[str, JsonValue]]:
    return "/v1/responses", _responses_body(deployment, {**_NO_CACHE, "stream_options": dict(_OBFUSCATION)})


async def test_concurrent_burst_across_chat_and_responses_answers_each_and_lands_each_once(
    rig: _Rig, scenario: Scenario
) -> None:
    chat: Final = _deploy(rig, scenario, "search")
    responses: Final = _deploy(rig, scenario, "search")
    outage: Final = _deploy(rig, scenario, "search-down")
    requests: Final = (
        *(_chat_request(chat) for _ in range(8)),
        *(_responses_request(responses) for _ in range(8)),
        *(_chat_request(outage) for _ in range(4)),
    )
    served: Final = await _burst(rig.base_url, rig.gateway.key, requests)
    chat_ids: Final = frozenset(
        _assert_restreamed(item.stream, iv.answer(chat.name), usage=iv.FOLLOWUP_USAGE) for item in served[:8]
    )
    response_ids: Final = tuple(
        _assert_responses_restreamed(item.stream, iv.answer(responses.name)) for item in served[8:16]
    )
    outage_ids: Final = frozenset(
        _assert_restreamed(item.stream, iv.answer(outage.name), usage=iv.FOLLOWUP_USAGE) for item in served[16:]
    )
    drained: Final = rig.wire.drain()
    _assert_upstream(iv.received(drained, chat.name), model_calls=16, searches=8)
    _assert_upstream(iv.received(drained, responses.name), model_calls=16, searches=8)
    _assert_upstream(iv.received(drained, outage.name), model_calls=8, searches=4)
    _assert_landed_once(chat.model, chat_ids, rows=8)
    _assert_landed_once(outage.model, outage_ids, rows=4)
    rows: Final = _spend_rows(responses.model, 8)
    assert len(rows) == 8 and len(set(response_ids)) == 8, rows
    assert all(
        any(rv.same_response(string_value(row["request_id"]), identity) for row in rows) for identity in response_ids
    )


def test_always_include_stream_usage_restreams_usage_on_chat_and_strips_it_from_azure_on_responses(
    tmp_path: Path,
) -> None:
    with _interception_proxy(tmp_path, iv.respond, {"always_include_stream_usage": True}) as owned:
        with owned.gateway.scenario() as opened:
            chat: Final = _deploy(owned, opened, "search")
            responses: Final = _deploy(owned, opened, "search")
            stream: Final = _chat(owned, _chat_body(chat))
            identity: Final = _assert_restreamed(stream, iv.answer(chat.name), usage=iv.FOLLOWUP_USAGE)
            replayed: Final = _assert_responses_restreamed(
                _responses(owned, _responses_body(responses)), iv.answer(responses.name)
            )
            drained: Final = owned.wire.drain()
            _assert_upstream(iv.received(drained, chat.name), model_calls=2, searches=1)
            _assert_upstream(iv.received(drained, responses.name), model_calls=2, searches=1)
            _assert_landed_once(chat.model, frozenset({identity}), rows=1)
            _assert_response_landed_once(responses.model, replayed)


def _holding(release: threading.Event, held: SimpleQueue[str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "POST" and urlsplit(request.target).path.endswith("/chat/completions"):
            held.put(request.target)
            assert release.wait(timeout=90), "The burst was never released"
        return iv.respond(request)

    return respond


@contextmanager
def _releasing(release: threading.Event) -> Generator[None, None, None]:
    try:
        yield
    finally:
        release.set()


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _worker_pids(log: str) -> tuple[int, ...]:
    return tuple(
        int(line.split(_STARTED_WORKER, 1)[1].split("]", 1)[0]) for line in log.splitlines() if _STARTED_WORKER in line
    )


def _burst_models(wire: Wire) -> Sequence[Mapping[str, JsonValue]]:
    return (
        {
            "model_name": _BURST_MODEL,
            "litellm_params": {
                "model": f"azure/{_BURST_DEPLOYMENT}",
                "api_base": wire.url,
                "api_key": iv.AZURE_KEY,
                "api_version": iv.API_VERSION,
            },
        },
    )


async def test_worker_sigkill_mid_burst_leaves_the_sibling_restreaming_with_usage(tmp_path: Path) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    deployment: Final = _Deployment(_BURST_MODEL, _BURST_DEPLOYMENT)
    with _interception_proxy(tmp_path, _holding(release, held), {}, _burst_models) as owned, _releasing(release):
        workers: Final = eventually(
            lambda: _worker_pids(owned.proxy.log.read_text()),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
        burst: Final = asyncio.create_task(
            _burst(
                owned.base_url, owned.gateway.key, tuple(_chat_request(deployment) for _ in range(20)), tolerate=True
            )
        )
        await asyncio.to_thread(eventually, held.qsize, lambda size: size == 20, 60)
        held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, owned.wire.url) for pid in workers})
        assert sum(held_by.values()) == 20, held_by
        victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
        victim: Final = psutil.Process(victim_pid)
        victim.suspend()
        victim.send_signal(signal.SIGKILL)
        release.set()
        served: Final = await burst
        assert held_by[survivor_pid] >= 10, held_by
        assert len(served) == held_by[survivor_pid], (held_by, len(served))
        identities: Final = frozenset(
            _assert_restreamed(item.stream, iv.answer(_BURST_DEPLOYMENT), usage=iv.FOLLOWUP_USAGE) for item in served
        )
        (follow_up,) = await _burst(owned.base_url, owned.gateway.key, (_chat_request(deployment),))
        recovered: Final = _assert_restreamed(follow_up.stream, iv.answer(_BURST_DEPLOYMENT), usage=iv.FOLLOWUP_USAGE)
        rows: Final = _spend_rows(_BURST_MODEL, len(identities) + 1)
        landed: Final = tuple(string_value(row["request_id"]) for row in rows if row["status"] == "success")
        assert sorted(landed) == sorted((*identities, recovered)), rows
