import json
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, held
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROJECT: Final = "scripted-project"
_PROMPT_TOKENS: Final = 13
_COMPLETION_TOKENS: Final = 4
_CACHED_TOKENS: Final = 5
_REASONING_TOKENS: Final = 2
_USAGE: Final = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_DETAILED_USAGE: Final = {
    **_USAGE,
    "prompt_tokens_details": {"cached_tokens": _CACHED_TOKENS},
    "completion_tokens_details": {"reasoning_tokens": _REASONING_TOKENS},
}
_REPLY_TEXT: Final = "iterator reply"
_SPEND_SELECT: Final = (
    'SELECT request_id, prompt_tokens, completion_tokens, spend, status FROM "LiteLLM_SpendLogs" WHERE api_key=%s'
)
_CACHE_SELECT: Final = (
    'SELECT cache_hit, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"'
)
_BURST_SELECT: Final = "SELECT api_key, prompt_tokens, completion_tokens FROM \"LiteLLM_SpendLogs\" WHERE api_key = ANY(string_to_array(%s, ','))"


def _vertex(model: str, location: str) -> Callable[[Gateway, Scenario, Wire], str]:
    def create(gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
        return scenario.model(
            model=f"vertex_ai/{model}",
            api_base=wire.url,
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=location,
            vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        )

    return create


def _keyed(model: str, **extra: JsonValue) -> Callable[[Gateway, Scenario, Wire], str]:
    def create(_gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
        return scenario.model(model=model, api_base=wire.url, api_key="synthetic-provider-key", **extra)

    return create


_MEMBERS: Final = {
    "vertex_model_garden": _vertex("xai/grok-4.20-non-reasoning", "global"),
    "vertex_mistral_partner": _vertex("mistral-small-2503", "us-central1"),
    "watsonx": _keyed("watsonx/ibm/granite-3-8b-instruct", zen_api_key="synthetic-zen-key", project_id=_PROJECT),
}


def _chunk(choices: list[JsonValue], **extra: JsonValue) -> bytes:
    body: Final = {
        "id": "chatcmpl-iterator",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "scripted",
        "choices": choices,
        **extra,
    }
    return f"data: {json.dumps(body)}\n\n".encode()


_DELTA: Final = _chunk([{"index": 0, "delta": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": None}])
_FINISH: Final = _chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}])
_USAGE_ONLY: Final = _chunk([], usage=_DETAILED_USAGE)
_DONE: Final = b"data: [DONE]\n\n"
_FINISH_THEN_USAGE: Final = (_DELTA, _FINISH, _USAGE_ONLY, _DONE)


def _reply_json() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "scripted",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": "stop"}
            ],
            "usage": _USAGE,
        }
    ).encode()


def _peer(frames: tuple[bytes, ...]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=frames)
        return Reply(body=_reply_json())

    return respond


def _sse_payloads(response: httpx.Response) -> list[dict[str, JsonValue]]:
    lines: Final = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    return [_JSON_OBJECT.validate_json(line) for line in lines if line != "[DONE]"]


def _chat_text(chunks: list[dict[str, JsonValue]]) -> str:
    return "".join(
        str(choice.get("delta", {}).get("content") or "")
        for chunk in chunks
        for choice in chunk.get("choices") or []  # comprehension-ok: chunk then choice
        if isinstance(choice, dict) and isinstance(choice.get("delta"), dict)
    )


def _assert_each_send_billed(key: str, sends: int) -> None:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: read_rows(_CACHE_SELECT, (digest,)), lambda rows: len(rows) >= sends, seconds=70)
    rows: Final = held(lambda: read_rows(_CACHE_SELECT, (digest,)), lambda found: len(found) == sends, holding=3)
    assert [row["cache_hit"] == "True" for row in rows] == [index > 0 for index in range(sends)], rows
    assert {(row["prompt_tokens"], row["completion_tokens"]) for row in rows} == {
        (_PROMPT_TOKENS, _COMPLETION_TOKENS)
    }, rows


def _spend_row(key: str) -> dict[str, JsonValue]:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda rows: len(rows) >= 1, seconds=70)
    rows: Final = held(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda found: len(found) == 1, holding=3)
    return rows[0]


def _stream_chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return _stream_chat_about(gateway, model, key, uuid.uuid4().hex)


def _stream_chat_about(gateway: Gateway, model: str, key: str, marker: str) -> httpx.Response:
    return gateway.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": f"iterator probe {marker}"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"Authorization": f"Bearer {key}"},
    )


@pytest.mark.parametrize("member", tuple(_MEMBERS))
def test_chat_stream_keeps_the_usage_chunk_sent_after_the_finish_chunk(gateway: Gateway, member: str) -> None:
    with wire_server(_peer(_FINISH_THEN_USAGE)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_chat(gateway, model, key)
        assert response.status_code == 200, response.text
        chunks: Final = _sse_payloads(response)
        assert _chat_text(chunks) == _REPLY_TEXT, response.text
        usage: Final = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
        assert len(usage) == 1, response.text
        assert {name: usage[0][name] for name in _USAGE} == _USAGE, response.text
        row: Final = _spend_row(key)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), row
        assert len(wire.drain()) == 1


@pytest.mark.parametrize("member", tuple(_MEMBERS))
def test_chat_stream_usage_carries_the_upstream_token_details(gateway: Gateway, member: str) -> None:
    with wire_server(_peer(_FINISH_THEN_USAGE)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_chat(gateway, model, key)
        assert response.status_code == 200, response.text
        usage: Final = [chunk["usage"] for chunk in _sse_payloads(response) if chunk.get("usage")]
        assert len(usage) == 1, response.text
        details: Final = usage[0]
        assert isinstance(details, dict)
        assert details["prompt_tokens_details"]["cached_tokens"] == _CACHED_TOKENS, response.text
        assert details["completion_tokens_details"]["reasoning_tokens"] == _REASONING_TOKENS, response.text
        assert len(wire.drain()) == 1


@pytest.mark.parametrize("member", tuple(_MEMBERS))
def test_chat_stream_cache_hit_replays_the_usage_chunk(gateway: Gateway, member: str) -> None:
    with wire_server(_peer(_FINISH_THEN_USAGE)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        marker: Final = uuid.uuid4().hex
        responses: Final = tuple(_stream_chat_about(gateway, model, key, marker) for _ in range(2))
        for response in responses:
            assert response.status_code == 200, response.text
            chunks = _sse_payloads(response)
            assert _chat_text(chunks) == _REPLY_TEXT, response.text
            usage = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
            assert len(usage) == 1, response.text
            assert {name: usage[0][name] for name in _USAGE} == _USAGE, response.text
        assert len(wire.drain()) == 1
        _assert_each_send_billed(key, 2)


@pytest.mark.parametrize("member", tuple(_MEMBERS))
def test_chat_non_stream_usage_is_unchanged(gateway: Gateway, member: str) -> None:
    with wire_server(_peer(_FINISH_THEN_USAGE)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]}, key=key
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert {name: body["usage"][name] for name in _USAGE} == _USAGE, response.text
        row: Final = _spend_row(key)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), row
        assert len(wire.drain()) == 1


@pytest.mark.parametrize("sends", [1, 2], ids=["miss", "cache_hit"])
def test_messages_stream_bridge_keeps_usage_after_finish(gateway: Gateway, sends: int) -> None:
    with wire_server(_peer(_FINISH_THEN_USAGE)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["vertex_model_garden"](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        body: Final = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": f"bridge probe {uuid.uuid4().hex}"}],
            "stream": True,
        }
        responses: Final = tuple(
            gateway.client.post("/v1/messages", json=body, headers={"Authorization": f"Bearer {key}"})
            for _ in range(sends)
        )
        for response in responses:
            assert response.status_code == 200, response.text
            deltas = [event for event in _sse_payloads(response) if event.get("type") == "message_delta"]
            assert len(deltas) == 1, response.text
            assert deltas[0]["usage"]["output_tokens"] == _COMPLETION_TOKENS, response.text
        _assert_each_send_billed(key, sends)
        assert len(wire.drain()) == 1


@pytest.mark.parametrize("sends", [1, 2], ids=["miss", "cache_hit"])
def test_responses_stream_bridge_keeps_usage_after_finish(gateway: Gateway, sends: int) -> None:
    with wire_server(_peer(_FINISH_THEN_USAGE)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["vertex_model_garden"](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        body: Final = {"model": model, "input": f"bridge probe {uuid.uuid4().hex}", "stream": True}
        responses: Final = tuple(
            gateway.client.post("/v1/responses", json=body, headers={"Authorization": f"Bearer {key}"})
            for _ in range(sends)
        )
        for response in responses:
            assert response.status_code == 200, response.text
            completed = [event for event in _sse_payloads(response) if event.get("type") == "response.completed"]
            assert len(completed) == 1, response.text
            usage = completed[0]["response"]["usage"]
            assert (usage["input_tokens"], usage["output_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), (
                response.text
            )
        _assert_each_send_billed(key, sends)
        assert len(wire.drain()) == 1


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param((_DELTA, b"data: {not json\n\n", _FINISH, _USAGE_ONLY, _DONE), id="malformed_data_line"),
        pytest.param((_DELTA, b"data:    \n\n", _FINISH, _USAGE_ONLY, _DONE), id="blank_data_line"),
        pytest.param((_DELTA, _FINISH, b"\n" * 5000, _USAGE_ONLY, _DONE), id="five_kb_of_blank_lines"),
        pytest.param(
            (b"id: 1\nevent: message\n" + _DELTA, b"id: 2\nevent: message\n" + _FINISH, _USAGE_ONLY, _DONE),
            id="event_and_id_lines",
        ),
    ],
)
def test_chat_stream_tolerates_noise_between_chunks(gateway: Gateway, frames: tuple[bytes, ...]) -> None:
    with wire_server(_peer(frames)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["vertex_model_garden"](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_chat(gateway, model, key)
        assert response.status_code == 200, response.text
        chunks: Final = _sse_payloads(response)
        assert _chat_text(chunks) == _REPLY_TEXT, response.text
        usage: Final = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
        assert len(usage) == 1 and usage[0]["prompt_tokens"] == _PROMPT_TOKENS, response.text
        assert len(wire.drain()) == 1


def test_chat_stream_of_only_blank_lines_ends_cleanly(gateway: Gateway) -> None:
    with wire_server(_peer((b"\n\n", b"\n\n", b"\n\n"))) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["vertex_model_garden"](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_chat(gateway, model, key)
        assert response.status_code == 200, response.text
        assert _chat_text(_sse_payloads(response)) == "", response.text
        assert "data: [DONE]" in response.text, response.text
        assert len(wire.drain()) == 1
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_chat_stream_upstream_drop_after_finish_ends_the_stream(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        if _JSON_OBJECT.validate_json(request.body).get("stream") is not True:
            return Reply(body=_reply_json())
        return Reply(
            content_type="text/event-stream", chunks=_FINISH_THEN_USAGE, abort_after=2, pause_between_chunks=1.5
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["vertex_model_garden"](gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_chat(gateway, model, key)
        assert response.status_code == 200, response.text
        assert _chat_text(_sse_payloads(response)) == _REPLY_TEXT, response.text
        assert len(wire.drain()) == 1
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        other: Final = scenario.key(models=[model])
        assert (
            gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "x"}]},
                key=other,
            ).status_code
            == 200
        )


def _burst_rows(digests: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return read_rows(_BURST_SELECT, (",".join(digests),))


def test_chat_stream_burst_bills_every_request_once(gateway: Gateway) -> None:
    with wire_server(_peer(_FINISH_THEN_USAGE)) as wire, gateway.scenario() as scenario:
        models: Final = {name: create(gateway, scenario, wire) for name, create in _MEMBERS.items()}
        names: Final = tuple(_MEMBERS) * 10
        keys: Final = tuple(scenario.key(models=[models[name]]) for name in names)
        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            responses: Final = tuple(
                pool.map(lambda pair: _stream_chat(gateway, models[pair[0]], pair[1]), zip(names, keys, strict=True))
            )
        assert [response.status_code for response in responses] == [200] * len(keys)
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        eventually(lambda: _burst_rows(digests), lambda rows: len(rows) >= len(keys), seconds=90)
        rows: Final = held(lambda: _burst_rows(digests), lambda found: len(found) == len(keys), holding=3)
        assert sorted(str(row["api_key"]) for row in rows) == sorted(digests)
        assert {(row["prompt_tokens"], row["completion_tokens"]) for row in rows} == {
            (_PROMPT_TOKENS, _COMPLETION_TOKENS)
        }, rows
        assert len(wire.drain()) == len(keys)
