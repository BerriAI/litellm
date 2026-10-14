import asyncio
import itertools
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import httpx
import openai
import psutil
import pytest
from integration._support.anthropic_sse import user_prompt
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, held
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, openai_error, responses_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

_SERVED_MODEL: Final = "Qwen/Qwen2.5-0.5B-Instruct"
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_PROMPT_TOKENS: Final = 5
_COMPLETION_TOKENS: Final = 3
_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_FINGERPRINT: Final = "fp-vllm-relay"
_SSE_END: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_LF: Final = b"\n\n"
_CRLF: Final = b"\r\n\r\n"
_ROW_COLUMNS: Final = (
    "request_id, status, call_type, model_group, custom_llm_provider, prompt_tokens, completion_tokens, spend, team_id"
)


@dataclass(frozen=True, slots=True)
class _Call:
    marker: str
    identity: str


def _call() -> _Call:
    marker: Final = "vllm-relay-" + uuid.uuid4().hex
    return _Call(marker, f"cmpl-{marker}")


def _frame(payload: Mapping[str, JsonValue] | str, end: bytes = _LF) -> bytes:
    data: Final = payload if isinstance(payload, str) else json.dumps(payload)
    return b"data: " + data.encode() + end


def _chunk(identity: str, choices: Sequence[JsonValue], **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _SERVED_MODEL,
        "choices": list(choices),
        **extra,
    }


def _delta(index: int, content: str) -> dict[str, JsonValue]:
    return {"index": index, "delta": {"content": content}, "finish_reason": None}


def _finish(index: int) -> dict[str, JsonValue]:
    return {"index": index, "delta": {}, "finish_reason": "stop"}


def _usage_only(identity: str) -> dict[str, JsonValue]:
    return _chunk(identity, [], usage=_USAGE, system_fingerprint=_FINGERPRINT)


def _stamped(chunk: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**chunk, "system_fingerprint": _FINGERPRINT}


def _sse(chunks: Sequence[bytes]) -> Reply:
    return Reply(content_type="text/event-stream", chunks=tuple(chunks))


def _block_payload(block: bytes) -> bytes:
    return b"\n".join(line[len(b"data:") :].strip() for line in block.splitlines() if line.startswith(b"data:"))


def _frames(raw: bytes) -> tuple[JsonValue, ...]:
    payloads: Final = tuple(_block_payload(block) for block in _SSE_END.split(raw))
    return tuple("[DONE]" if payload == b"[DONE]" else json.loads(payload) for payload in payloads if payload)


def _asks_usage(body: Mapping[str, JsonValue]) -> bool:
    options: Final = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") in (True, "true")


def _plain_stream(call: _Call, body: Mapping[str, JsonValue]) -> Reply:
    return _sse(
        (
            _frame(_chunk(call.identity, [_delta(0, "Hello")])),
            _frame(_chunk(call.identity, [_finish(0)])),
            *((_frame(_usage_only(call.identity)),) if _asks_usage(body) else ()),
            _frame("[DONE]"),
        )
    )


def _chat_body(model: str, call: _Call, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": call.marker}], **extra}


def _scripted(call: _Call, reply: Callable[[Mapping[str, JsonValue]], Reply]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert call.marker in json.dumps(body), body
        return reply(body)

    return answering_model_discovery(respond)


def _deploy(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"hosted_vllm/{_SERVED_MODEL}",
        api_base=wire.url + "/v1",
        input_cost_per_token=_INPUT_COST,
        output_cost_per_token=_OUTPUT_COST,
    )


@contextmanager
def _deployment(gateway: Gateway, respond: Callable[[Request], Reply]) -> Iterator[tuple[Wire, str]]:
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        yield wire, _deploy(scenario, wire)


def _posted(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(request.body) for request in wire.drain() if request.method == "POST")


def _read_rows(model: str) -> list[dict[str, JsonValue]]:
    return read_rows(f'SELECT {_ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,))


def _rows(model: str, count: int = 1) -> list[dict[str, JsonValue]]:
    return held(lambda: _read_rows(model), lambda rows: len(rows) >= count, holding=2, seconds=70)


def _assert_reported_usage(row: Mapping[str, JsonValue]) -> None:
    assert row["status"] == "success", row
    assert row["custom_llm_provider"] == "hosted_vllm", row
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), row
    assert row["spend"] == pytest.approx(_PROMPT_TOKENS * _INPUT_COST + _COMPLETION_TOKENS * _OUTPUT_COST), row


def _assert_counted_usage(row: Mapping[str, JsonValue]) -> None:
    prompt: Final = row["prompt_tokens"]
    completion: Final = row["completion_tokens"]
    assert row["status"] == "success", row
    assert isinstance(prompt, int) and isinstance(completion, int) and prompt > 0 and completion > 0, row
    assert row["spend"] == pytest.approx(prompt * _INPUT_COST + completion * _OUTPUT_COST), row


def _one_row(model: str, identity: str | None = None) -> dict[str, JsonValue]:
    rows: Final = _rows(model)
    assert len(rows) == 1, rows
    assert identity is None or rows[0]["request_id"] == identity, rows
    return rows[0]


def _stream(gateway: Gateway, path: str, body: Mapping[str, JsonValue], *, key: str | None = None) -> httpx.Response:
    response: Final = gateway.request("POST", path, body, key=key)
    assert response.status_code == 200, response.text
    return response


def test_two_choice_stream_keeps_every_frame_in_order_and_stamps_only_the_last_finish(gateway: Gateway) -> None:
    call: Final = _call()
    chunks: Final = (
        _chunk(call.identity, [_delta(0, "A")]),
        _chunk(call.identity, [_delta(1, "B")]),
        _chunk(call.identity, [_finish(0)]),
        _chunk(call.identity, [_delta(1, "C")]),
        _chunk(call.identity, [_finish(1)]),
    )
    reply: Final = _sse((*map(_frame, chunks), _frame(_usage_only(call.identity)), _frame("[DONE]")))
    with _deployment(gateway, _scripted(call, lambda _: reply)) as (wire, model):
        response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True, n=2))
        assert _frames(response.content) == (*chunks[:4], _stamped(chunks[4]), "[DONE]"), response.text
        (posted,) = _posted(wire)
        assert posted["stream_options"] == {"include_usage": True}, posted
        _assert_reported_usage(_one_row(model, call.identity))


def test_stream_with_usage_on_every_choice_chunk_is_relayed_byte_for_byte(gateway: Gateway) -> None:
    call: Final = _call()
    running: Final = tuple(
        {"prompt_tokens": _PROMPT_TOKENS, "completion_tokens": done, "total_tokens": _PROMPT_TOKENS + done}
        for done in (1, 2, 3)
    )
    upstream_bytes: Final = (
        _frame(_chunk(call.identity, [_delta(0, "Hel")], usage=running[0])),
        _frame(_chunk(call.identity, [_delta(0, "lo")], usage=running[1])),
        _frame(_chunk(call.identity, [_finish(0)], usage=running[2])),
        _frame(_usage_only(call.identity)),
        _frame("[DONE]"),
    )
    with _deployment(gateway, _scripted(call, lambda _: _sse(upstream_bytes))) as (_, model):
        response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True))
        assert response.content == b"".join(upstream_bytes), response.text
        _assert_reported_usage(_one_row(model, call.identity))


def test_stream_split_into_single_bytes_is_reassembled_and_stripped(gateway: Gateway) -> None:
    call: Final = _call()
    content: Final = _chunk(call.identity, [_delta(0, "Hello")])
    finish: Final = _chunk(call.identity, [_finish(0)])
    whole: Final = b"".join((_frame(content), _frame(finish), _frame(_usage_only(call.identity)), _frame("[DONE]")))
    reply: Final = _sse(tuple(whole[index : index + 1] for index in range(len(whole))))
    with _deployment(gateway, _scripted(call, lambda _: reply)) as (_, model):
        response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True))
        assert _frames(response.content) == (content, _stamped(finish), "[DONE]"), response.text
        _assert_reported_usage(_one_row(model, call.identity))


def test_crlf_terminated_frames_keep_their_terminators_after_the_usage_frame_is_removed(gateway: Gateway) -> None:
    call: Final = _call()
    content: Final = _chunk(call.identity, [_delta(0, "Hello")])
    finish: Final = _chunk(call.identity, [_finish(0)])
    reply: Final = _sse(
        tuple(_frame(payload, _CRLF) for payload in (content, finish, _usage_only(call.identity), "[DONE]"))
    )
    with _deployment(gateway, _scripted(call, lambda _: reply)) as (_, model):
        response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True))
        assert _frames(response.content) == (content, _stamped(finish), "[DONE]"), response.text
        assert response.content.endswith(_CRLF) and response.content.count(_CRLF) == 3, response.content
        assert _LF not in response.content.replace(_CRLF, b""), response.content
        _assert_reported_usage(_one_row(model, call.identity))


def test_stream_whose_upstream_sends_no_usage_frame_is_relayed_whole_and_counted_locally(gateway: Gateway) -> None:
    call: Final = _call()
    upstream_bytes: Final = (
        _frame(_chunk(call.identity, [_delta(0, "Hello there")])),
        _frame(_chunk(call.identity, [_finish(0)])),
        _frame("[DONE]"),
    )
    with _deployment(gateway, _scripted(call, lambda _: _sse(upstream_bytes))) as (wire, model):
        response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True))
        assert response.content == b"".join(upstream_bytes), response.text
        (posted,) = _posted(wire)
        assert posted["stream_options"] == {"include_usage": True}, posted
        row: Final = _one_row(model, call.identity)
        _assert_counted_usage(row)


def test_upstream_error_on_a_stream_fails_the_call_and_the_model_keeps_serving(gateway: Gateway) -> None:
    call: Final = _call()
    replies: Final = iter((openai_error(400), _plain_stream(call, {"stream_options": {"include_usage": True}})))
    with _deployment(gateway, _scripted(call, lambda _: next(replies))) as (_, model):
        eventually(lambda: _all_workers_serve(gateway, model), bool, seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS)
        failed: Final = gateway.request(
            "POST", "/vllm/chat/completions", _chat_body(model, call, stream=True), headers={"Connection": "close"}
        )
        assert failed.status_code >= 400, failed.text
        assert call.identity not in failed.text, failed.text
        served: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True))
        assert _frames(served.content)[-1] == "[DONE]", served.text


_MALFORMED_STREAM_OPTIONS: Final[Mapping[str, JsonValue]] = {
    "string": "include_usage",
    "list": ["include_usage"],
    "integer": 1,
    "five_kb_string": "u" * 5000,
}


def _rejects_malformed_stream_options(call: _Call) -> Callable[[Mapping[str, JsonValue]], Reply]:
    def reply(body: Mapping[str, JsonValue]) -> Reply:
        if isinstance(body.get("stream_options"), dict):
            return _plain_stream(call, body)
        return openai_error(400)

    return reply


@pytest.mark.parametrize("shape", list(_MALFORMED_STREAM_OPTIONS))
def test_malformed_stream_options_reach_the_upstream_as_the_caller_sent_them(gateway: Gateway, shape: str) -> None:
    call: Final = _call()
    sent: Final = _MALFORMED_STREAM_OPTIONS[shape]
    with _deployment(gateway, _scripted(call, _rejects_malformed_stream_options(call))) as (wire, model):
        response: Final = gateway.request(
            "POST",
            "/vllm/chat/completions",
            _chat_body(model, call, stream=True, stream_options=sent),
            headers={"Connection": "close"},
        )
        (posted,) = _posted(wire)
        assert posted["stream_options"] == sent, posted
        assert response.status_code >= 400, response.text


def test_null_stream_options_get_usage_injected_and_the_usage_frame_removed(gateway: Gateway) -> None:
    call: Final = _call()
    with _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (wire, model):
        response: Final = _stream(
            gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True, stream_options=None)
        )
        (posted,) = _posted(wire)
        assert posted["stream_options"] == {"include_usage": True}, posted
        assert _frames(response.content) == (
            _chunk(call.identity, [_delta(0, "Hello")]),
            _stamped(_chunk(call.identity, [_finish(0)])),
            "[DONE]",
        ), response.text
        _assert_reported_usage(_one_row(model, call.identity))


@pytest.mark.parametrize(
    "sent",
    [pytest.param({}, id="empty"), pytest.param({"include_usage": False}, id="include_usage_false")],
)
def test_stream_options_without_include_usage_true_get_usage_injected_and_the_usage_frame_removed(
    gateway: Gateway, sent: Mapping[str, JsonValue]
) -> None:
    call: Final = _call()
    with _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (wire, model):
        response: Final = _stream(
            gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True, stream_options=dict(sent))
        )
        (posted,) = _posted(wire)
        assert posted["stream_options"] == {"include_usage": True}, posted
        assert _frames(response.content) == (
            _chunk(call.identity, [_delta(0, "Hello")]),
            _stamped(_chunk(call.identity, [_finish(0)])),
            "[DONE]",
        ), response.text
        _assert_reported_usage(_one_row(model, call.identity))


def test_null_include_usage_reaches_the_upstream_as_sent_and_is_counted_locally(gateway: Gateway) -> None:
    call: Final = _call()
    sent: Final[dict[str, JsonValue]] = {"include_usage": None}
    with _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (wire, model):
        response: Final = _stream(
            gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True, stream_options=sent)
        )
        (posted,) = _posted(wire)
        assert posted["stream_options"] == sent, posted
        assert response.content == b"".join(_plain_stream(call, posted).chunks or ()), response.text
        _assert_counted_usage(_one_row(model, call.identity))


def test_duplicated_stream_options_key_takes_the_last_value(gateway: Gateway) -> None:
    call: Final = _call()
    raw: Final = (
        '{"model": "%s", "messages": [{"role": "user", "content": "%s"}], "stream": true, '
        '"stream_options": {"include_usage": true}, "stream_options": {"continuous_usage_stats": true}}'
    )
    with _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (wire, model):
        response: Final = gateway.client.post(
            "/vllm/chat/completions",
            content=(raw % (model, call.marker)).encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert response.status_code == 200, response.text
        (posted,) = _posted(wire)
        assert posted["stream_options"] == {"include_usage": True}, posted
        assert '"usage"' not in response.text, response.text
        _assert_reported_usage(_one_row(model, call.identity))


def test_include_usage_sent_as_the_string_true_still_reaches_the_caller(gateway: Gateway) -> None:
    call: Final = _call()
    with _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (_, model):
        response: Final = _stream(
            gateway,
            "/vllm/chat/completions",
            _chat_body(model, call, stream=True, stream_options={"include_usage": "true"}),
        )
        assert _usage_only(call.identity) in _frames(response.content), response.text
        _assert_reported_usage(_one_row(model, call.identity))


def test_stream_flag_sent_as_a_string_streams_verbatim_and_is_counted_locally(gateway: Gateway) -> None:
    call: Final = _call()
    with _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (wire, model):
        response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream="true"))
        (posted,) = _posted(wire)
        assert "stream_options" not in posted, posted
        assert response.content == b"".join(_plain_stream(call, posted).chunks or ()), response.text
        _assert_counted_usage(_one_row(model, call.identity))


def _upstream_socket_states(wire: Wire) -> frozenset[str]:
    port: Final = httpx.URL(wire.url).port
    return frozenset(
        connection.status
        for connection in psutil.Process().net_connections(kind="tcp")
        if connection.laddr.port == port
    )


def test_caller_disconnect_mid_stream_closes_the_upstream_and_logs_what_was_relayed(gateway: Gateway) -> None:
    call: Final = _call()
    release: Final = threading.Event()
    reply: Final = Reply(
        content_type="text/event-stream",
        chunks=(
            _frame(_chunk(call.identity, [_delta(0, "Hello there")])),
            _frame(_chunk(call.identity, [_finish(0)])),
            _frame(_usage_only(call.identity)) + _frame("[DONE]"),
        ),
        gate_after_first=release,
    )
    with _deployment(gateway, _scripted(call, lambda _: reply)) as (wire, model):
        try:
            with gateway.client.stream(
                "POST",
                "/vllm/chat/completions",
                json=_chat_body(model, call, stream=True),
                headers={"Authorization": f"Bearer {gateway.key}"},
            ) as response:
                assert response.status_code == 200
                assert call.identity.encode() in next(response.iter_raw())
            eventually(
                lambda: _upstream_socket_states(wire),
                lambda states: psutil.CONN_CLOSE_WAIT in states and psutil.CONN_ESTABLISHED not in states,
                seconds=30,
            )
        finally:
            release.set()
        _assert_counted_usage(_one_row(model, call.identity))


def _embedding_reply(_: Mapping[str, JsonValue]) -> Reply:
    body: Final = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
        "model": _SERVED_MODEL,
        "usage": {"prompt_tokens": _PROMPT_TOKENS, "total_tokens": _PROMPT_TOKENS},
    }
    return Reply(body=json.dumps(body).encode())


def test_embeddings_write_one_row_with_the_upstream_prompt_tokens(gateway: Gateway) -> None:
    call: Final = _call()
    with _deployment(gateway, _scripted(call, _embedding_reply)) as (wire, model):
        response: Final = gateway.request("POST", "/vllm/embeddings", {"model": model, "input": call.marker})
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["data"] == [
            {"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}
        ], response.text
        assert tuple(request.target for request in wire.drain() if request.method == "POST") == ("/v1/embeddings",)
        row: Final = _one_row(model)
        assert row["status"] == "success", row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_TOKENS, 0), row
        assert row["spend"] == pytest.approx(_PROMPT_TOKENS * _INPUT_COST), row


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_responses_write_one_row_and_relay_the_body_untouched(gateway: Gateway, stream: bool) -> None:
    call: Final = _call()
    reply: Final = responses_reply(f"resp_{call.marker}", _SERVED_MODEL, "Hello", stream=stream)
    with _deployment(gateway, _scripted(call, lambda _: reply)) as (wire, model):
        sent: Final = {"model": model, "input": call.marker, **({"stream": True} if stream else {})}
        response: Final = gateway.request("POST", "/vllm/responses", sent)
        assert response.status_code == 200, response.text
        assert response.content == (b"".join(reply.chunks) if reply.chunks else reply.body), response.text
        (posted,) = _posted(wire)
        assert {name: value for name, value in posted.items() if name != "model"} == {
            name: value for name, value in sent.items() if name != "model"
        }, posted
        _assert_reported_usage(_one_row(model))


def _pinned(gateway: Gateway) -> httpx.Client:
    return httpx.Client(
        base_url=gateway.client.base_url,
        timeout=30,
        trust_env=False,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=300),
    )


def _chunk_text(chunk: ChatCompletionChunk) -> str:
    return "".join(choice.delta.content or "" for choice in chunk.choices)


def _listed(response: httpx.Response, model: str) -> bool:
    return response.status_code == 200 and model in response.text


def test_openai_sdk_stream_assembles_the_answer_without_a_usage_chunk(gateway: Gateway) -> None:
    call: Final = _call()
    with (
        _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (_, model),
        _pinned(gateway) as http,
    ):
        auth: Final = {"Authorization": f"Bearer {gateway.key}"}
        eventually(
            lambda: http.get("/v1/models", headers=auth),
            lambda listed: _listed(listed, model),
            seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
        )
        client: Final = openai.OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/vllm",
            api_key=gateway.key,
            http_client=http,
            max_retries=0,
        )
        chunks: Final = tuple(
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": call.marker}], stream=True
            )
        )
        assert "".join(map(_chunk_text, chunks)) == "Hello", chunks
        assert [chunk.usage for chunk in chunks if chunk.usage is not None] == [], chunks
        assert {chunk.id for chunk in chunks} == {call.identity}, chunks
        _assert_reported_usage(_one_row(model, call.identity))


async def _async_listed(http: httpx.AsyncClient, key: str, model: str) -> httpx.Response:
    return await http.get("/v1/models", headers={"Authorization": f"Bearer {key}"})


async def test_async_openai_sdk_stream_assembles_the_answer_without_a_usage_chunk(gateway: Gateway) -> None:
    call: Final = _call()
    loop: Final = asyncio.get_running_loop()
    with _deployment(gateway, _scripted(call, lambda body: _plain_stream(call, body))) as (_, model):
        async with httpx.AsyncClient(
            base_url=gateway.client.base_url,
            timeout=30,
            trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=300),
        ) as http:
            await asyncio.to_thread(
                eventually,
                lambda: asyncio.run_coroutine_threadsafe(_async_listed(http, gateway.key, model), loop).result(),
                lambda listed: _listed(listed, model),
                2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
            )
            client: Final = openai.AsyncOpenAI(
                base_url=f"{str(gateway.client.base_url).rstrip('/')}/vllm",
                api_key=gateway.key,
                http_client=http,
                max_retries=0,
            )
            stream: Final = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": call.marker}], stream=True
            )
            chunks: Final = tuple([chunk async for chunk in stream])
        assert "".join(map(_chunk_text, chunks)) == "Hello", chunks
        assert [chunk.usage for chunk in chunks if chunk.usage is not None] == [], chunks
        assert {chunk.id for chunk in chunks} == {call.identity}, chunks
        _assert_reported_usage(_one_row(model, call.identity))


def test_identical_requests_repeated_each_reach_the_upstream_and_each_write_one_row(gateway: Gateway) -> None:
    call: Final = _call()
    served: Final = itertools.count()

    def reply(body: Mapping[str, JsonValue]) -> Reply:
        repeat: Final = _Call(call.marker, f"{call.identity}-{next(served)}")
        if body.get("stream") is True:
            return _plain_stream(repeat, body)
        completion: Final = {
            "id": repeat.identity,
            "object": "chat.completion",
            "created": 1,
            "model": _SERVED_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": _USAGE,
        }
        return Reply(body=json.dumps(completion).encode())

    with _deployment(gateway, _scripted(call, reply)) as (wire, model):
        bodies: Final = (_chat_body(model, call), _chat_body(model, call), *(_chat_body(model, call, stream=True),) * 2)
        texts: Final = tuple(_stream(gateway, "/vllm/chat/completions", body).text for body in bodies)
        identities: Final = tuple(f"{call.identity}-{index}" for index in range(4))
        assert all(identity in text for identity, text in zip(identities, texts)), texts
        assert len(_posted(wire)) == 4
        rows: Final = _rows(model, 4)
        assert sorted(str(row["request_id"]) for row in rows) == sorted(identities), rows
        for row in rows:
            _assert_reported_usage(row)


def test_non_json_content_type_never_reaches_the_upstream(gateway: Gateway) -> None:
    call: Final = _call()
    with (
        wire_server(answering_model_discovery(lambda _: _plain_stream(call, {}))) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _deploy(scenario, wire)
        response: Final = gateway.client.post(
            "/vllm/chat/completions",
            content=json.dumps(_chat_body(model, call, stream=True)).encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "text/plain", "Connection": "close"},
        )
        posted: Final = tuple(request.body for request in wire.drain() if request.method == "POST")
        assert response.status_code >= 400, response.text
        assert posted == (), posted


def test_team_key_stream_writes_the_row_under_the_team(gateway: Gateway) -> None:
    call: Final = _call()
    with wire_server(_scripted(call, lambda body: _plain_stream(call, body))) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True), key=key)
        assert '"usage"' not in response.text, response.text
        row: Final = _one_row(model, call.identity)
        assert row["team_id"] == team, row
        _assert_reported_usage(row)


_LARGE_FRAME_BYTES: Final = 8 * 1024 * 1024
_PIECE_BYTES: Final = 8 * 1024
_HEALTH_CEILING_SECONDS: Final = 1.0
_RELAY_RATIO_CEILING: Final = 4
_RELAY_SLACK_SECONDS: Final = 1.0


@dataclass(frozen=True, slots=True)
class _Relay:
    response: httpx.Response
    seconds: float
    health: tuple[float, ...]


def _timed_health(client: httpx.Client) -> float:
    started: Final = time.perf_counter()
    response: Final = client.get("/health/liveliness", headers={"Connection": "close"})
    assert response.status_code == 200, response.text
    return time.perf_counter() - started


def _health_latencies(gateway: Gateway, done: threading.Event) -> tuple[float, ...]:
    with httpx.Client(base_url=gateway.client.base_url, timeout=60, trust_env=False) as fresh:
        return tuple(_timed_health(fresh) for _ in iter(done.is_set, True))


def _finish_and_done(call: _Call) -> bytes:
    return _frame(_chunk(call.identity, [_finish(0)])) + _frame("[DONE]")


def _content_then_usage(call: _Call, content: Sequence[bytes]) -> Callable[[Mapping[str, JsonValue]], Reply]:
    usage: Final = _frame(_chunk(call.identity, [], usage=_USAGE))
    finish: Final = _frame(_chunk(call.identity, [_finish(0)]))

    def reply(body: Mapping[str, JsonValue]) -> Reply:
        return _sse((*content, finish, *((usage,) if _asks_usage(body) else ()), _frame("[DONE]")))

    return reply


def _timed_relay(gateway: Gateway, call: _Call, content: Sequence[bytes]) -> _Relay:
    with _deployment(gateway, _scripted(call, _content_then_usage(call, content))) as (_, model):
        eventually(lambda: _all_workers_serve(gateway, model), bool, seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS)
        done: Final = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            polled: Final = pool.submit(_health_latencies, gateway, done)
            started: Final = time.perf_counter()
            try:
                response: Final = _stream(gateway, "/vllm/chat/completions", _chat_body(model, call, stream=True))
            finally:
                done.set()
            seconds: Final = time.perf_counter() - started
            return _Relay(response, seconds, polled.result())


def test_frame_terminated_only_at_its_end_relays_as_fast_as_the_same_bytes_in_small_frames(
    gateway: Gateway, record_property: Callable[[str, object], None]
) -> None:
    small_call, large_call = _call(), _call()
    small: Final = tuple(
        _frame(_chunk(small_call.identity, [_delta(0, "x" * _PIECE_BYTES)]))
        for _ in range(_LARGE_FRAME_BYTES // _PIECE_BYTES)
    )
    large: Final = _frame(_chunk(large_call.identity, [_delta(0, "x" * _LARGE_FRAME_BYTES)]))
    pieces: Final = tuple(large[start : start + _PIECE_BYTES] for start in range(0, len(large), _PIECE_BYTES))
    small_relay: Final = _timed_relay(gateway, small_call, small)
    large_relay: Final = _timed_relay(gateway, large_call, pieces)
    record_property("small_frames_seconds", round(small_relay.seconds, 3))
    record_property("large_frame_seconds", round(large_relay.seconds, 3))
    record_property("health_calls", len(large_relay.health))
    record_property("health_max_seconds", round(max(large_relay.health), 3))
    assert small_relay.response.content == b"".join(small) + _finish_and_done(small_call), len(
        small_relay.response.content
    )
    assert large_relay.response.content == large + _finish_and_done(large_call), len(large_relay.response.content)
    assert max(large_relay.health) < _HEALTH_CEILING_SECONDS, sorted(large_relay.health)[-5:]
    assert large_relay.seconds < _RELAY_RATIO_CEILING * small_relay.seconds + _RELAY_SLACK_SECONDS, (
        large_relay.seconds,
        small_relay.seconds,
    )


@dataclass(frozen=True, slots=True)
class _Outcome:
    identity: str
    status: int
    text: str


def _burst_reply(dropped: Callable[[], bool]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = JSON_OBJECT.validate_json(request.body)
        marker: Final = user_prompt(body) if "messages" in body else str(body.get("prompt"))
        call: Final = _Call(marker, f"cmpl-{marker}")
        if dropped():
            return Reply(drop_connection=True)
        if body.get("stream") is not True:
            completion: Final = {
                "id": call.identity,
                "object": "chat.completion",
                "created": 1,
                "model": _SERVED_MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
                "usage": _USAGE,
            }
            return Reply(body=json.dumps(completion).encode())
        stream: Final = _plain_stream(call, body)
        return Reply(content_type=stream.content_type, chunks=stream.chunks, pause_between_chunks=0.05)

    return answering_model_discovery(respond)


async def _send(client: httpx.AsyncClient, key: str, body: Mapping[str, JsonValue], identity: str) -> _Outcome:
    try:
        response: Final = await client.post(
            "/vllm/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
        )
    except httpx.TransportError as error:
        return _Outcome(identity, 0, repr(error))
    return _Outcome(identity, response.status_code, response.text)


def _all_workers_serve(gateway: Gateway, model: str) -> bool:
    auth: Final = {"Authorization": f"Bearer {gateway.key}"}
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as fresh:
        return all(_listed(fresh.get("/v1/models", headers={**auth, "Connection": "close"}), model) for _ in range(16))


async def test_burst_with_dropped_upstream_connections_lands_every_completed_call_once(gateway: Gateway) -> None:
    arrivals: Final = itertools.count(1)
    with wire_server(_burst_reply(lambda: next(arrivals) % 4 == 0)) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, wire)
        await asyncio.to_thread(
            eventually, lambda: _all_workers_serve(gateway, model), bool, 2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
        )
        calls: Final = tuple(_call() for _ in range(24))
        bodies: Final = tuple(
            _chat_body(model, call, **({"stream": True} if index % 3 else {})) for index, call in enumerate(calls)
        )
        async with httpx.AsyncClient(base_url=gateway.client.base_url, timeout=90, trust_env=False) as client:
            outcomes: Final = await asyncio.gather(
                *(_send(client, gateway.key, body, call.identity) for body, call in zip(bodies, calls))
            )
        completed: Final = frozenset(
            outcome.identity for outcome in outcomes if outcome.status == 200 and outcome.identity in outcome.text
        )
        failed: Final = tuple(outcome for outcome in outcomes if outcome.identity not in completed)
        assert len(completed) >= 12, outcomes
        assert all(outcome.status != 200 for outcome in failed), failed
        rows: Final = _rows(model, len(completed))
        assert sorted(str(row["request_id"]) for row in rows) == sorted(completed), rows
        health: Final = gateway.request("GET", "/health/liveliness")
        assert health.status_code == 200, health.text
