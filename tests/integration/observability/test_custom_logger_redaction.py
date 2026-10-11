import asyncio
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.redaction_probe import RedactionProbe, SyncRedactionProbe
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

INPUT: Final = "private-input-redaction-canary"
REASONING: Final = "private-reasoning-redaction-canary"
OUTPUT: Final = "private-output-redaction-canary"
pytestmark = pytest.mark.timeout(240)


def _provider(request: Request) -> Reply:
    if request.method == "GET" and request.target == "/v1/models":
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = json.loads(request.body)
    if request.target == "/v1/embeddings":
        assert body["input"] == INPUT, body
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "model": body["model"],
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                }
            ).encode()
        )
    if request.target == "/v1/completions":
        assert body["prompt"] == INPUT, body
        return Reply(
            body=json.dumps(
                {
                    "id": "cmpl-redaction-probe",
                    "object": "text_completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"text": OUTPUT, "index": 0, "finish_reason": "stop", "logprobs": None}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )
    if request.target == "/v1/responses":
        assert body["input"] == INPUT, body
        return Reply(
            body=json.dumps(
                {
                    "id": "resp-redaction-probe",
                    "object": "response",
                    "created_at": 1,
                    "model": body["model"],
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg-redaction-probe",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": OUTPUT, "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )
    if request.target == "/v1/messages":
        assert body["messages"] == [{"role": "user", "content": INPUT}], body
        return Reply(
            body=json.dumps(
                {
                    "id": "msg-redaction-probe",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [
                        {"type": "thinking", "thinking": REASONING, "signature": "synthetic-signature"},
                        {"type": "text", "text": OUTPUT},
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ).encode()
        )
    assert body["messages"] == [{"role": "user", "content": INPUT}], body
    payload: Final = {
        "id": "chatcmpl-redaction-" + body.get("user", "probe"),
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": OUTPUT,
                    "provider_specific_fields": {
                        "reasoning_content": REASONING,
                        "thinking_blocks": [{"thinking": REASONING}],
                        "reasoningContentBlocks": [{"reasoningText": {"text": REASONING}}],
                    },
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    if body.get("stream"):
        choice: Final = payload["choices"][0]
        chunk: Final = {
            **payload,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": choice["message"], "finish_reason": None}],
        }
        finish: Final = {
            **payload,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"data: {json.dumps(event)}\n\n".encode() for event in (chunk, finish))
            + (b"data: [DONE]\n\n",),
        )
    return Reply(body=json.dumps(payload).encode())


@pytest.fixture(scope="module")
def redaction_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire, Wire]]:
    tmp_path: Final = tmp_path_factory.mktemp("custom-logger-redaction")
    with (
        wire_server(_provider) as provider,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = tmp_path / "redaction.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": name,
                            "litellm_params": {
                                "model": model,
                                "api_base": provider.url if model.startswith("anthropic/") else provider.url + "/v1",
                                "api_key": "synthetic-provider-key",
                            },
                        }
                        for name, model in (
                            ("redaction-probe", "openai/gpt-4o-mini"),
                            ("redaction-text", "openai/gpt-3.5-turbo-instruct"),
                            ("redaction-embedding", "openai/text-embedding-3-large"),
                            ("redaction-responses", "openai/gpt-5.6-sol"),
                            ("redaction-messages", "anthropic/claude-sonnet-5-5"),
                        )
                    ],
                    "litellm_settings": {
                        "callbacks": ["integration._support.redaction_probe.probe"],
                        "enable_redis_auth_cache": True,
                        "cache": True,
                        "cache_params": {
                            "type": "redis",
                            "host": "os.environ/REDIS_HOST",
                            "port": "os.environ/REDIS_PORT",
                        },
                    },
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "database_url": "os.environ/DATABASE_URL",
                        "disable_model_info_refresh": True,
                    },
                }
            )
        )
        with owned_proxy(
            rig, tmp_path, {"REDACTION_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config, workers=2
        ) as proxy:
            yield proxy, provider, sink


@pytest.mark.parametrize("redact", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_custom_logger_redacts_nested_input_and_provider_reasoning(
    redaction_proxy: tuple[Gateway, Wire, Wire], redact: bool, stream: bool
) -> None:
    proxy, provider, sink = redaction_proxy
    response: Final = proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "redaction-probe",
            "messages": [{"role": "user", "content": INPUT}],
            "stream": stream,
            "cache": {"no-cache": True},
        },
        headers={"x-litellm-enable-message-redaction": "true"} if redact else {},
    )
    assert response.status_code == 200, response.text
    first: Final = (
        json.loads(next(line[6:] for line in response.text.splitlines() if line.startswith("data: ")))
        if stream
        else response.json()
    )
    message: Final = first["choices"][0]["delta" if stream else "message"]
    assert message["content"] == OUTPUT, message
    assert REASONING in json.dumps(message["provider_specific_fields"]), message
    received: Final = eventually(sink.drain, lambda values: len(values) == 1, seconds=20)
    callback: Final = json.loads(received[0].body)
    body: Final = callback["request"]["body"]
    assert "proxy_server_request" not in body, body
    assert len(tuple(request for request in provider.drain() if request.method == "POST")) == 1
    logged: Final = json.dumps(callback)
    if redact:
        assert INPUT not in logged and REASONING not in logged and OUTPUT not in logged, callback
    else:
        assert INPUT in logged and REASONING in logged and OUTPUT in logged, callback


@pytest.mark.parametrize("redact", [False, True])
@pytest.mark.parametrize(
    ("route", "field", "model"),
    [("/v1/completions", "prompt", "redaction-text"), ("/v1/embeddings", "input", "redaction-embedding")],
)
def test_custom_logger_redacts_prompt_and_embedding_input(
    redaction_proxy: tuple[Gateway, Wire, Wire], route: str, field: str, model: str, redact: bool
) -> None:
    proxy, provider, sink = redaction_proxy
    response: Final = proxy.request(
        "POST",
        route,
        {"model": model, field: INPUT, "cache": {"no-cache": True}},
        headers={"x-litellm-enable-message-redaction": "true"} if redact else {},
    )
    assert response.status_code == 200, response.text
    if field == "input":
        assert response.json()["data"][0]["embedding"] == [0.25, 0.5]
    else:
        assert response.json()["choices"][0]["text"] == OUTPUT
    received: Final = eventually(sink.drain, bool, seconds=20)
    assert len(received) == 1, received
    callback: Final = json.loads(received[0].body)
    assert "proxy_server_request" not in callback["request"]["body"]
    assert len(tuple(request for request in provider.drain() if request.method == "POST")) == 1
    assert (INPUT in json.dumps(callback)) is not redact, callback


@pytest.mark.parametrize("redact", [False, True])
@pytest.mark.parametrize("native_messages", [False, True])
def test_native_messages_and_responses_logging_redaction(
    redaction_proxy: tuple[Gateway, Wire, Wire], native_messages: bool, redact: bool
) -> None:
    proxy, provider, sink = redaction_proxy
    body: Final = (
        {"model": "redaction-messages", "messages": [{"role": "user", "content": INPUT}], "max_tokens": 64}
        if native_messages
        else {"model": "redaction-responses", "input": INPUT}
    )
    response: Final = proxy.request(
        "POST",
        "/v1/messages" if native_messages else "/v1/responses",
        {**body, "cache": {"no-cache": True}},
        headers={"x-litellm-enable-message-redaction": "true"} if redact else {},
    )
    assert response.status_code == 200, response.text
    if native_messages:
        assert response.json()["content"][1]["text"] == OUTPUT
        assert response.json()["content"][0]["thinking"] == REASONING
    else:
        assert response.json()["output"][0]["content"][0]["text"] == OUTPUT
    received: Final = eventually(sink.drain, bool, seconds=20)
    assert len(received) == 1, received
    logged: Final = received[0].body.decode()
    assert len(tuple(request for request in provider.drain() if request.method == "POST")) == 1
    assert (INPUT in logged) is not redact, logged
    assert (OUTPUT in logged) is not redact, logged
    if native_messages:
        assert (REASONING in logged) is not redact, logged


def test_concurrent_redacted_and_unredacted_callbacks_remain_isolated(
    redaction_proxy: tuple[Gateway, Wire, Wire],
) -> None:
    proxy, provider, sink = redaction_proxy
    users: Final = tuple(f"redaction-chaos-{index}" for index in range(4))

    def send(index: int):
        return proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "redaction-probe",
                "messages": [{"role": "user", "content": INPUT}],
                "user": users[index],
                "cache": {"no-cache": True},
            },
            headers={"x-litellm-enable-message-redaction": "true"} if index % 2 else {},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses: Final = tuple(pool.map(send, range(4)))
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == OUTPUT
        assert REASONING in json.dumps(response.json()["choices"][0]["message"])
    batches: Final[list[Request]] = []  # mutable-ok: drain consumes the sink queue, so polls retain previous events

    def observed() -> tuple[Request, ...]:
        batches.extend(sink.drain())
        return tuple(batches)

    events: Final = eventually(observed, lambda values: len(values) == 4, seconds=20)
    callbacks: Final = tuple(json.loads(event.body) for event in events)
    assert {callback["request"]["body"]["user"] for callback in callbacks} == set(users)
    for callback in callbacks:
        index: Final = users.index(callback["request"]["body"]["user"])
        logged: Final = json.dumps(callback)
        assert (INPUT in logged) is (index % 2 == 0), callback
        assert (REASONING in logged) is (index % 2 == 0), callback
        assert (OUTPUT in logged) is (index % 2 == 0), callback
    assert len(tuple(request for request in provider.drain() if request.method == "POST")) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("redact", [False, True])
async def test_sdk_success_callback_redaction_over_http(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, redact: bool
) -> None:
    with wire_server(_provider) as provider, wire_server(lambda request: Reply()) as sink:
        monkeypatch.setenv("REDACTION_PROBE_URL", sink.url)
        messages: Final = [{"role": "user", "content": INPUT}]
        request: Final = {"body": {"messages": messages, "model": "gpt-4o-mini"}}
        callback: Final = RedactionProbe().async_log_success_event if asynchronous else SyncRedactionProbe()
        parameters: Final = {
            "model": "openai/gpt-4o-mini",
            "messages": messages,
            "api_base": provider.url + "/v1",
            "api_key": "synthetic-sdk-provider-key",
            "proxy_server_request": request,
            "turn_off_message_logging": redact,
            "success_callback": [callback],
        }
        response: Final = (
            await litellm.acompletion(**parameters)
            if asynchronous
            else await asyncio.to_thread(litellm.completion, **parameters)
        )
        assert response.choices[0].message.content == OUTPUT
        assert REASONING in json.dumps(response.choices[0].message.provider_specific_fields)
        await GLOBAL_LOGGING_WORKER.flush()
        delivered: Final = await asyncio.to_thread(eventually, sink.drain, bool, seconds=20)
        assert len(delivered) == 1, delivered
        logged: Final = delivered[0].body.decode()
        assert request["body"]["messages"] == messages
        if redact:
            assert INPUT not in logged and REASONING not in logged and OUTPUT not in logged, logged
        else:
            assert INPUT in logged and REASONING in logged and OUTPUT in logged, logged
