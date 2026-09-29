import asyncio
import json
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, JsonValue, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    sink: Wire
    batches: list[Request]

    def failure_events(self, model: str) -> tuple[dict[str, JsonValue], ...]:
        self.batches.extend(self.sink.drain())
        return tuple(
            object_value(event)
            for batch in self.batches
            for event in json.loads(batch.body)
            if model in json.dumps(event)
        )


def _prompt_text(body: dict[str, JsonValue]) -> str:
    messages: Final = body.get("messages")
    if isinstance(messages, list) and messages:
        last: Final = messages[-1]
        if isinstance(last, dict):
            content: Final = last.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(part["text"])
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
    input_value: Final = body.get("input")
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        return json.dumps(input_value)
    return json.dumps(body)[:200]


def _success_reply(target: str, text: str) -> Reply:
    if target.endswith("/messages"):
        return Reply(
            body=json.dumps(
                {
                    "id": "msg_" + uuid.uuid4().hex,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                }
            ).encode()
        )
    if target.endswith("/responses"):
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_" + uuid.uuid4().hex,
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "id": "msg_" + uuid.uuid4().hex,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                }
            ).encode()
        )
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    try:
        body: Final = json.loads(request.body)
    except json.JSONDecodeError:
        return Reply(status=404, body=b"{}")
    text: Final = _prompt_text(body)
    if text.startswith("ok "):
        return _success_reply(request.target, text)
    status: Final = 401 if text.startswith("auth401 ") else 400
    error: Final = {"type": "invalid_request_error", "message": f"Unsupported content: {text}"}
    if request.target.endswith("/messages"):
        return Reply(status=status, body=json.dumps({"type": "error", "error": error}).encode())
    return Reply(status=status, body=json.dumps({"error": error}).encode())


_SINK_OUTAGE: Final = threading.Event()
_SINK_SLOW: Final = threading.Event()


def _sink(request: Request) -> Reply:
    if _SINK_OUTAGE.is_set():
        return Reply(status=503, body=b'{"error":"sink down"}')
    if _SINK_SLOW.is_set():
        time.sleep(2)
    return Reply()


@contextmanager
def _booted_rig(
    root: Path,
    provider: Wire,
    sink: Wire,
    settings: Mapping[str, JsonValue],
) -> Iterator[Rig]:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["allow_client_side_credentials"] = True
    config["litellm_settings"].update({"DEFAULT_FLUSH_INTERVAL_SECONDS": 1, **settings})
    path: Final = root / "failure_redaction.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        gateway_from_environment() as gateway,
        owned_proxy(gateway, root, {"GENERIC_LOGGER_ENDPOINT": sink.url}, config=path, workers=2) as proxy,
    ):
        yield Rig(proxy, provider, sink, [])  # mutable-ok: sink drain consumes batches, later polls keep earlier ones


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_provider) as wire:
        yield wire


@pytest.fixture(scope="module")
def sink() -> Iterator[Wire]:
    with wire_server(_sink) as wire:
        yield wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire) -> Iterator[Rig]:
    with _booted_rig(
        tmp_path_factory.mktemp("failure_redaction"),
        provider,
        sink,
        {
            "callbacks": ["generic_api"],
            "turn_off_message_logging": True,
            "standard_logging_payload_excluded_fields": ["hidden_params"],
        },
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def rig_off(tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire) -> Iterator[Rig]:
    with _booted_rig(
        tmp_path_factory.mktemp("failure_redaction_off"), provider, sink, {"callbacks": ["generic_api"]}
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def rig_failure_callback(tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire) -> Iterator[Rig]:
    with _booted_rig(
        tmp_path_factory.mktemp("failure_redaction_failure_cb"),
        provider,
        sink,
        {"failure_callback": ["generic_api"], "turn_off_message_logging": True},
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def rig_success_callback(tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire) -> Iterator[Rig]:
    with _booted_rig(
        tmp_path_factory.mktemp("failure_redaction_success_cb"),
        provider,
        sink,
        {"success_callback": ["generic_api"], "turn_off_message_logging": True},
    ) as booted:
        yield booted


def _secret_prompt() -> str:
    return "confidential-prompt-" + uuid.uuid4().hex


def _single_failure_event(rig: Rig, model: str) -> dict[str, JsonValue]:
    events: Final = eventually(
        lambda: tuple(event for event in rig.failure_events(model) if event.get("status") == "failure"),
        lambda values: len(values) >= 1,
        seconds=30,
    )
    assert len(events) == 1, events
    return events[0]


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT status, messages, response, proxy_server_request, metadata FROM "LiteLLM_SpendLogs" '
            "WHERE request_id=%s",
            (call_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _chat(rig: Rig, model: str, messages: list[JsonValue], key: str | None = None) -> httpx.Response:
    return rig.proxy.request("POST", "/v1/chat/completions", {"model": model, "messages": messages}, key=key)


def _error_information(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(event["error_information"])


def test_provider_error_echoing_the_prompt_is_redacted_in_callbacks_and_spend_logs(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(
            api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key", num_retries=0
        )
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code == 400, response.text
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        event: Final = _single_failure_event(rig, model)
        assert secret not in json.dumps(event), json.dumps(event)
        error_information: Final = _error_information(event)
        assert error_information["error_class"] == "BadRequestError", error_information
        assert error_information["error_code"] == "400", error_information
        assert error_information["llm_provider"] == "openai", error_information
        assert "hidden_params" not in event, sorted(event)
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert row["status"] == "failure", row
        assert secret not in json.dumps(row, default=str), row
        persisted: Final = object_value(
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        )
        assert object_value(persisted["error_information"])["error_class"] == "BadRequestError", persisted


def test_transformation_error_quoting_the_prompt_is_redacted_in_callbacks(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5", api_base=rig.provider.url, api_key="synthetic-anthropic-key"
        )
        response: Final = _chat(
            rig,
            model,
            [
                {"role": "user", "content": secret},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": 12345, "type": "function", "function": {"name": None, "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": 12345, "content": secret},
            ],
        )
        assert response.status_code == 400, response.text
        assert rig.provider.drain() == ()
        event: Final = _single_failure_event(rig, model)
        assert secret not in json.dumps(event), json.dumps(event)
        error_information: Final = _error_information(event)
        assert error_information["error_code"] == "400", error_information
        assert error_information["error_class"], error_information


def test_proxy_only_rejection_does_not_leak_the_prompt_to_callbacks(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        allowed: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        denied: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[allowed])
        rig.provider.drain()
        response: Final = _chat(rig, denied, [{"role": "user", "content": secret}], key=key)
        assert response.status_code in (401, 403), response.text
        assert rig.provider.drain() == ()
        event: Final = _single_failure_event(rig, denied)
        assert secret not in json.dumps(event), json.dumps(event)
        assert "hidden_params" not in event, sorted(event)
        assert _error_information(event)["error_code"] == str(response.status_code), event


def _openai(rig: Rig, key: str | None = None) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=str(rig.proxy.client.base_url) + "/v1",
        api_key=key or rig.proxy.key,
        max_retries=0,
    )


def _async_openai(rig: Rig, key: str | None = None) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=str(rig.proxy.client.base_url) + "/v1",
        api_key=key or rig.proxy.key,
        max_retries=0,
    )


def _anthropic(rig: Rig, key: str | None = None) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(rig.proxy.client.base_url), api_key=key or rig.proxy.key, max_retries=0)


def _async_anthropic(rig: Rig, key: str | None = None) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(rig.proxy.client.base_url), api_key=key or rig.proxy.key, max_retries=0
    )


def _call_id(response: httpx.Response) -> str:
    call_id: Final = response.headers.get("x-litellm-call-id")
    assert call_id, dict(response.headers)
    return call_id


def _error_call_id(error: openai.APIStatusError | anthropic.APIStatusError) -> str:
    call_id: Final = error.response.headers.get("x-litellm-call-id")
    assert call_id, dict(error.response.headers)
    return call_id


def _assert_failure_redacted(rig: Rig, model: str, secret: str, call_id: str) -> dict[str, JsonValue]:
    event: Final = _single_failure_event(rig, model)
    assert secret not in json.dumps(event), json.dumps(event)
    error_information: Final = _error_information(event)
    assert error_information["error_class"], error_information
    assert error_information["llm_provider"], error_information
    row: Final = _spend_row(call_id)
    assert row["status"] == "failure", row
    assert secret not in json.dumps(row, default=str), row
    return event


def _assert_failure_raw(rig: Rig, model: str, secret: str, call_id: str | None = None) -> dict[str, JsonValue]:
    event: Final = _single_failure_event(rig, model)
    assert secret in json.dumps(event), json.dumps(event)
    if call_id is not None:
        row: Final = _spend_row(call_id)
        assert secret in json.dumps(row, default=str), row
    return event


def _denied_body(path: str, model: str, secret: str) -> dict[str, JsonValue]:
    if path == "/v1/messages":
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": secret}]}
    if path == "/v1/responses":
        return {"model": model, "input": secret}
    return {"model": model, "messages": [{"role": "user", "content": secret}]}


# --- A. Endpoint x stream x client, provider 400 echoing the prompt -------------------

_SDK_CLIENTS: Final = ("openai_sync", "openai_async", "httpx")


def _chat_call(rig: Rig, client: str, model: str, secret: str, stream: bool) -> str:
    messages: Final = [{"role": "user", "content": secret}]
    if client == "httpx":
        response: Final = _chat(rig, model, messages)
        assert response.status_code == 400, response.text
        return _call_id(response)
    if client == "openai_sync":
        with pytest.raises(openai.BadRequestError) as caught:
            _openai(rig).chat.completions.create(model=model, messages=messages, stream=stream)
        assert caught.value.status_code == 400, caught.value
        return _error_call_id(caught.value)

    async def fire() -> str:
        with pytest.raises(openai.BadRequestError) as caught:
            await _async_openai(rig).chat.completions.create(model=model, messages=messages, stream=stream)
        return _error_call_id(caught.value)

    return asyncio.run(fire())


@pytest.mark.parametrize("client", _SDK_CLIENTS, ids=lambda value: value)
def test_a1_chat_provider_error_redacted(rig: Rig, client: str) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        call_id: Final = _chat_call(rig, client, model, secret, stream=False)
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        event: Final = _assert_failure_redacted(rig, model, secret, call_id)
        assert object_value(event["error_information"])["error_class"] == "BadRequestError", event


@pytest.mark.parametrize("client", _SDK_CLIENTS, ids=lambda value: value)
def test_a2_chat_stream_provider_error_redacted(rig: Rig, client: str) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        call_id: Final = _chat_call(rig, client, model, secret, stream=True)
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        _assert_failure_redacted(rig, model, secret, call_id)


def _consume_messages_stream(client: anthropic.Anthropic, model: str, messages: list[dict[str, str]]) -> None:
    with client.messages.stream(model=model, max_tokens=16, messages=messages) as events:
        for _ in events:
            pass


async def _consume_messages_stream_async(
    client: anthropic.AsyncAnthropic, model: str, messages: list[dict[str, str]]
) -> None:
    async with client.messages.stream(model=model, max_tokens=16, messages=messages) as events:
        async for _ in events:
            pass


def _messages_call(rig: Rig, client: str, model: str, secret: str, stream: bool) -> str:
    messages: Final = [{"role": "user", "content": secret}]
    if client == "httpx":
        response: Final = rig.proxy.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 16, "messages": messages}
        )
        assert response.status_code == 400, response.text
        return _call_id(response)
    if client == "openai_sync":
        anthropic_client: Final = _anthropic(rig)
        if stream:
            with pytest.raises(anthropic.BadRequestError) as caught:
                _consume_messages_stream(anthropic_client, model, messages)
            return _error_call_id(caught.value)
        with pytest.raises(anthropic.BadRequestError) as caught:
            anthropic_client.messages.create(model=model, max_tokens=16, messages=messages)
        return _error_call_id(caught.value)

    async def fire() -> str:
        client_async: Final = _async_anthropic(rig)
        if stream:
            with pytest.raises(anthropic.BadRequestError) as caught:
                await _consume_messages_stream_async(client_async, model, messages)
            return _error_call_id(caught.value)
        with pytest.raises(anthropic.BadRequestError) as caught:
            await client_async.messages.create(model=model, max_tokens=16, messages=messages)
        return _error_call_id(caught.value)

    return asyncio.run(fire())


@pytest.mark.parametrize("client", _SDK_CLIENTS, ids=lambda value: value)
def test_a3_messages_provider_error_redacted(rig: Rig, client: str) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5", api_base=rig.provider.url, api_key="synthetic-anthropic-key"
        )
        call_id: Final = _messages_call(rig, client, model, secret, stream=False)
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        _assert_failure_redacted(rig, model, secret, call_id)


@pytest.mark.parametrize("client", _SDK_CLIENTS, ids=lambda value: value)
def test_a4_messages_stream_provider_error_redacted(rig: Rig, client: str) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5", api_base=rig.provider.url, api_key="synthetic-anthropic-key"
        )
        call_id: Final = _messages_call(rig, client, model, secret, stream=True)
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        _assert_failure_redacted(rig, model, secret, call_id)


def _responses_call(rig: Rig, client: str, model: str, secret: str, stream: bool) -> str:
    if client == "httpx":
        response: Final = rig.proxy.request(
            "POST", "/v1/responses", {"model": model, "input": secret, "stream": stream}
        )
        assert response.status_code == 400, response.text
        return _call_id(response)
    if client == "openai_sync":
        with pytest.raises(openai.BadRequestError) as caught:
            _openai(rig).responses.create(model=model, input=secret, stream=stream)
        return _error_call_id(caught.value)

    async def fire() -> str:
        with pytest.raises(openai.BadRequestError) as caught:
            await _async_openai(rig).responses.create(model=model, input=secret, stream=stream)
        return _error_call_id(caught.value)

    return asyncio.run(fire())


@pytest.mark.parametrize("client", _SDK_CLIENTS, ids=lambda value: value)
def test_a5_responses_provider_error_redacted(rig: Rig, client: str) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        call_id: Final = _responses_call(rig, client, model, secret, stream=False)
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        _assert_failure_redacted(rig, model, secret, call_id)


@pytest.mark.parametrize("client", _SDK_CLIENTS, ids=lambda value: value)
def test_a6_responses_stream_provider_error_redacted(rig: Rig, client: str) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        call_id: Final = _responses_call(rig, client, model, secret, stream=True)
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        _assert_failure_redacted(rig, model, secret, call_id)


def test_a7_messages_transformation_error_redacted(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5", api_base=rig.provider.url, api_key="synthetic-anthropic-key"
        )
        response: Final = rig.proxy.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": secret},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": 12345, "type": "function", "function": {"name": None, "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": 12345, "content": secret},
                ],
            },
        )
        assert response.status_code == 400, response.text
        _assert_failure_redacted(rig, model, secret, _call_id(response))


@pytest.mark.parametrize(
    "path", ("/v1/chat/completions", "/v1/messages", "/v1/responses"), ids=lambda v: v.split("/")[-1]
)
def test_a8_proxy_only_rejection_redacted(rig: Rig, path: str) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        allowed: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        denied: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[allowed])
        rig.provider.drain()
        response: Final = rig.proxy.request("POST", path, _denied_body(path, denied, secret), key=key)
        assert response.status_code in (401, 403), response.text
        assert rig.provider.drain() == ()
        event: Final = _single_failure_event(rig, denied)
        assert secret not in json.dumps(event), json.dumps(event)
        assert _error_information(event)["error_code"] == str(response.status_code), event


def test_a9_unknown_model_failure_redacted(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    model: Final = "unknown-" + uuid.uuid4().hex
    rig.provider.drain()
    response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
    assert response.status_code == 400, response.text
    assert rig.provider.drain() == ()
    event: Final = eventually(lambda: rig.failure_events(model), lambda values: len(values) >= 1, seconds=20)[0]
    assert secret not in json.dumps(event), json.dumps(event)
    row: Final = _spend_row(_call_id(response))
    assert secret not in json.dumps(row, default=str), row


# --- B. Redaction source modes (YAML global off unless noted) -------------------------


@pytest.mark.parametrize(
    ("headers", "body"),
    (
        pytest.param(None, {}, id="b1_no_signal"),
        pytest.param({"x-litellm-enable-message-redaction": "true"}, {}, id="b2_enable_header"),
        pytest.param({"litellm-enable-message-redaction": "true"}, {}, id="b3_legacy_enable_header"),
        pytest.param(None, {"turn_off_message_logging": True}, id="b4_body_param"),
        pytest.param(None, {"metadata": {"turn_off_message_logging": True}}, id="b5_metadata_param"),
        pytest.param(None, {"litellm_metadata": {"turn_off_message_logging": True}}, id="b5b_litellm_metadata"),
    ),
)
def test_b_request_level_opt_in_redacts(
    rig_off: Rig, headers: Mapping[str, str] | None, body: Mapping[str, JsonValue]
) -> None:
    secret: Final = _secret_prompt()
    with rig_off.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_off.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = rig_off.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": secret}], **body},
            headers=headers,
        )
        assert response.status_code == 400, response.text
        assert any(secret.encode() in request.body for request in rig_off.provider.drain())
        if headers is None and not body:
            _assert_failure_raw(rig_off, model, secret, _call_id(response))
        else:
            _assert_failure_redacted(rig_off, model, secret, _call_id(response))


def test_b8_global_on_permitted_key_opt_out_keeps_raw(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model], metadata={"allow_client_message_redaction_opt_out": True})
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": secret}]},
            key=key,
            headers={"litellm-disable-message-redaction": "true"},
        )
        assert response.status_code == 400, response.text
        _assert_failure_raw(rig, model, secret, _call_id(response))


def test_b9_global_on_disable_header_without_permission_stays_redacted(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model])
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": secret}]},
            key=key,
            headers={"litellm-disable-message-redaction": "true"},
        )
        assert response.status_code == 400, response.text
        _assert_failure_redacted(rig, model, secret, _call_id(response))


def test_b10_global_on_team_permission_opt_out_keeps_raw(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        team: Final = scenario.team(metadata={"allow_client_message_redaction_opt_out": True})
        key: Final = scenario.key(models=[model], team_id=team)
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": secret}]},
            key=key,
            headers={"litellm-disable-message-redaction": "true"},
        )
        assert response.status_code == 400, response.text
        _assert_failure_raw(rig, model, secret, _call_id(response))


def _logging_callback_vars(flag: bool) -> dict[str, JsonValue]:
    return {"logging": [{"callback_name": "generic_api", "callback_vars": {"turn_off_message_logging": flag}}]}


def _event_for_call(rig: Rig, model: str, call_id: str) -> dict[str, JsonValue]:
    return eventually(
        lambda: tuple(event for event in rig.failure_events(model) if event.get("litellm_call_id") == call_id),
        lambda values: len(values) == 1,
        seconds=90,
    )[0]


def _assert_callback_vars_decision(rig: Rig, model: str, flag: bool, key: str) -> None:
    ok_secret: Final = f"ok {_secret_prompt()}"
    succeeded: Final = rig.proxy.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": ok_secret}]}, key=key
    )
    assert succeeded.status_code == 200, succeeded.text
    success_event: Final = _event_for_call(rig, model, _call_id(succeeded))
    assert (ok_secret not in json.dumps(success_event)) == flag, json.dumps(success_event)
    fail_secret: Final = _secret_prompt()
    failed: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": fail_secret}]},
        key=key,
    )
    assert failed.status_code == 400, failed.text
    if flag:
        _assert_failure_redacted(rig, model, fail_secret, _call_id(failed))
    else:
        _assert_failure_raw(rig, model, fail_secret, _call_id(failed))


@pytest.mark.timeout(280)
@pytest.mark.parametrize("global_flag", ["on", "off"])
@pytest.mark.parametrize("flag", [True, False], ids=["vars_true", "vars_false"])
def test_b6_key_logging_callback_vars_drive_the_failure_decision(
    request: pytest.FixtureRequest, global_flag: str, flag: bool
) -> None:
    rig: Final = request.getfixturevalue("rig" if global_flag == "on" else "rig_off")
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model], metadata=_logging_callback_vars(flag))
        _assert_callback_vars_decision(rig, model, flag, key)


@pytest.mark.timeout(280)
@pytest.mark.parametrize("global_flag", ["on", "off"])
@pytest.mark.parametrize("flag", [True, False], ids=["vars_true", "vars_false"])
def test_b7_team_logging_callback_vars_drive_the_failure_decision(
    request: pytest.FixtureRequest, global_flag: str, flag: bool
) -> None:
    rig: Final = request.getfixturevalue("rig" if global_flag == "on" else "rig_off")
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        team: Final = scenario.team(metadata=_logging_callback_vars(flag))
        key: Final = scenario.key(models=[model], team_id=team)
        _assert_callback_vars_decision(rig, model, flag, key)


# --- C. Callback registration modes (YAML global on) ----------------------------------


def test_c2_failure_callback_registration_redacts(rig_failure_callback: Rig) -> None:
    rig: Final = rig_failure_callback
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code == 400, response.text
        _assert_failure_redacted(rig, model, secret, _call_id(response))


def test_c3_success_callback_only_emits_no_failure_event(rig_success_callback: Rig) -> None:
    rig: Final = rig_success_callback
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code == 400, response.text
        events: Final = eventually(
            lambda: rig.failure_events(model),
            lambda values: len(values) >= 1,
            seconds=8,
            return_last_on_timeout=True,
        )
        assert events == (), events


def test_c4_excluded_fields_stripped_on_failure(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code == 400, response.text
        event: Final = _assert_failure_redacted(rig, model, secret, _call_id(response))
        assert "hidden_params" not in event, sorted(event)


# --- D. Cache-hit twin -----------------------------------------------------------------


@pytest.mark.timeout(280)
def test_d1_cache_hit_success_then_failure_redacted(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        rig.provider.drain()
        ok_text: Final = "ok " + uuid.uuid4().hex
        first: Final = _chat(rig, model, [{"role": "user", "content": ok_text}])
        assert first.status_code == 200, first.text
        second: Final = _chat(rig, model, [{"role": "user", "content": ok_text}])
        assert second.status_code == 200, second.text
        assert second.json()["id"] == first.json()["id"], "second call did not hit the response cache"
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code == 400, response.text
        _assert_failure_redacted(rig, model, secret, _call_id(response))


# --- E. Sad paths -----------------------------------------------------------------------

_HOSTILE_HEADERS: Final = ("", "0", "false", "1", "a,b", "h" * 5000)


@pytest.mark.parametrize("value", _HOSTILE_HEADERS, ids=lambda v: v[:8] or "empty")
@pytest.mark.timeout(280)
def test_e1_hostile_enable_header_unauthenticated_and_authenticated(rig_off: Rig, value: str) -> None:
    secret: Final = _secret_prompt()
    with rig_off.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_off.provider.url + "/v1", api_key="synthetic-provider-key")
        rejected: Final = rig_off.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": secret}]},
            key="sk-bogus",
            headers={"x-litellm-enable-message-redaction": value},
        )
        assert rejected.status_code == 401, rejected.text
        call_ids: list[str] = []  # mutable-ok: collect per-request call ids across the loop
        for _ in range(2):
            response: Final = rig_off.proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": secret}], "cache": {"no-cache": True}},
                headers={"x-litellm-enable-message-redaction": value},
            )
            assert response.status_code == 400, response.text
            call_ids.append(_call_id(response))
        rig_off.provider.drain()
        for call_id in call_ids:
            events: Final = eventually(
                lambda cid=call_id: tuple(
                    event
                    for event in rig_off.failure_events(model)
                    if event.get("litellm_call_id") == cid
                    and object_value(event.get("error_information")).get("error_class") != "KeyNotFoundError"
                ),
                lambda values: len(values) == 1,
                seconds=90,
            )
            if value:
                assert secret not in json.dumps(events[0]), (value, json.dumps(events[0])[:400])
            else:
                assert secret in json.dumps(events[0]), (value, json.dumps(events[0])[:400])


@pytest.mark.parametrize("value", (1, [True], "", "v" * 5000, None), ids=lambda v: type(v).__name__)
def test_e2_odd_turn_off_message_logging_values_never_500(rig_off: Rig, value: JsonValue) -> None:
    secret: Final = _secret_prompt()
    with rig_off.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_off.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = rig_off.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": secret}],
                "turn_off_message_logging": value,
            },
        )
        assert response.status_code == 400, response.text
        event: Final = _single_failure_event(rig_off, model)
        assert event["status"] == "failure", event


def test_e3_sink_rejections_keep_proxy_serving_and_later_events_still_redacted(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    _SINK_OUTAGE.set()
    try:
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
            during: Final = _chat(rig, model, [{"role": "user", "content": "down " + secret}])
            assert during.status_code == 400, during.text
            ok: Final = _chat(rig, model, [{"role": "user", "content": "ok alive " + uuid.uuid4().hex}])
            assert ok.status_code == 200, ok.text
            rig.sink.drain()
    finally:
        _SINK_OUTAGE.clear()
    with rig.proxy.scenario() as scenario:
        model_two: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        after: Final = _chat(rig, model_two, [{"role": "user", "content": "after " + secret}])
        assert after.status_code == 400, after.text
        event: Final = _single_failure_event(rig, model_two)
        assert secret not in json.dumps(event), json.dumps(event)


def test_e4_header_only_unknown_model_spend_row_never_carries_marker(rig_off: Rig) -> None:
    secret: Final = _secret_prompt()
    response: Final = rig_off.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": "unknown-" + uuid.uuid4().hex, "messages": [{"role": "user", "content": secret}]},
        headers={"x-litellm-enable-message-redaction": "true"},
    )
    assert response.status_code == 400, response.text
    row: Final = _spend_row(_call_id(response))
    assert secret not in json.dumps(row, default=str), row


@pytest.mark.parametrize("callback_fixture", ("rig", "rig_failure_callback"), ids=("callbacks", "failure_callback"))
@pytest.mark.parametrize("text", ("auth401 ", ""), ids=("provider_401", "provider_400"))
def test_e5_provider_401_and_400_echo_redacted(
    request: pytest.FixtureRequest, callback_fixture: str, text: str
) -> None:
    rig: Final = request.getfixturevalue(callback_fixture)
    secret: Final = text + _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code in (400, 401), response.text
        _assert_failure_redacted(rig, model, secret, _call_id(response))


@pytest.mark.timeout(280)
def test_e6_malformed_callback_config_fails_boot_identically(
    tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire
) -> None:
    with pytest.raises(AssertionError, match="exited before readiness"):
        with _booted_rig(
            tmp_path_factory.mktemp("failure_redaction_bad_cb"),
            provider,
            sink,
            {
                "callbacks": ["generic_api", "not_a_callback"],
                "failure_callback": None,
                "turn_off_message_logging": True,
            },
        ):
            raise AssertionError("malformed callback config must not boot")


def test_e7_spend_logs_detail_endpoint_redacted(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code == 400, response.text
        call_id: Final = _call_id(response)
        _assert_failure_redacted(rig, model, secret, call_id)
        detail: Final = eventually(
            lambda: rig.proxy.get(f"/spend/logs/ui/{call_id}"),
            lambda value: bool(value),
            seconds=30,
        )
        assert secret not in json.dumps(detail), json.dumps(detail)[:2000]


# --- F. Edge ----------------------------------------------------------------------------

_F1_SLOTS: Final = ("top", "metadata", "litellm_metadata")


def _redaction_body(model: str, secret: str, slot: str, value: JsonValue) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": secret}],
    }
    if slot == "top":
        if value != "MISSING":
            body["turn_off_message_logging"] = value
    else:
        body[slot] = {} if value == "MISSING" else {"turn_off_message_logging": value}
    return body


@pytest.mark.parametrize("slot", _F1_SLOTS)
@pytest.mark.parametrize("value", ("MISSING", None, ""), ids=("missing", "null", "empty"))
def test_f1_turn_off_edge_values_global_on(rig: Rig, slot: str, value: JsonValue) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = rig.proxy.request("POST", "/v1/chat/completions", _redaction_body(model, secret, slot, value))
        assert response.status_code == 400, response.text
        _assert_failure_redacted(rig, model, secret, _call_id(response))


@pytest.mark.parametrize("slot", _F1_SLOTS)
@pytest.mark.parametrize("value", ("MISSING", None, ""), ids=("missing", "null", "empty"))
def test_f1_turn_off_edge_values_global_off(rig_off: Rig, slot: str, value: JsonValue) -> None:
    secret: Final = _secret_prompt()
    with rig_off.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_off.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = rig_off.proxy.request(
            "POST", "/v1/chat/completions", _redaction_body(model, secret, slot, value)
        )
        assert response.status_code == 400, response.text
        _assert_failure_raw(rig_off, model, secret, _call_id(response))


def test_f2_permitted_opt_out_top_level_false_beats_metadata_true(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model], metadata={"allow_client_message_redaction_opt_out": True})
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": secret}],
                "turn_off_message_logging": False,
                "metadata": {"turn_off_message_logging": True},
            },
            key=key,
        )
        assert response.status_code == 400, response.text
        _assert_failure_raw(rig, model, secret, _call_id(response))


def test_f3_team_opt_out_permission_flip_takes_effect(rig: Rig) -> None:
    first_secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        team: Final = scenario.team(metadata={"allow_client_message_redaction_opt_out": True})
        key: Final = scenario.key(models=[model], team_id=team)
        allowed: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": first_secret}]},
            key=key,
            headers={"litellm-disable-message-redaction": "true"},
        )
        assert allowed.status_code == 400, allowed.text
        _assert_failure_raw(rig, model, first_secret, _call_id(allowed))
        rig.proxy.post("/team/update", {"team_id": team, "metadata": {}})

        def now_redacted() -> tuple[bool, ...]:
            secret: Final = _secret_prompt()
            fired: Final = rig.proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": secret}]},
                key=key,
                headers={"litellm-disable-message-redaction": "true"},
            )
            assert fired.status_code == 400, fired.text
            events: Final = eventually(
                lambda: tuple(
                    event for event in rig.failure_events(model) if event.get("litellm_call_id") == _call_id(fired)
                ),
                lambda values: len(values) == 1,
                seconds=20,
            )
            return (secret not in json.dumps(events[0]),)

        converged: Final = eventually(now_redacted, lambda values: values[0], seconds=120)
        assert converged[0]


@pytest.mark.timeout(280)
def test_f3_key_logging_callback_vars_flip_takes_effect(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model], metadata=_logging_callback_vars(True))
        denied: Final = rig.proxy.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": secret}]}, key=key
        )
        assert denied.status_code == 400, denied.text
        _assert_failure_redacted(rig, model, secret, _call_id(denied))
        rig.proxy.post("/key/update", {"key": key, "metadata": _logging_callback_vars(False)})

        def now_raw() -> tuple[bool, ...]:
            probe: Final = _secret_prompt()
            fired: Final = rig.proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": probe}]},
                key=key,
            )
            assert fired.status_code == 400, fired.text
            events: Final = eventually(
                lambda: tuple(
                    event for event in rig.failure_events(model) if event.get("litellm_call_id") == _call_id(fired)
                ),
                lambda values: len(values) == 1,
                seconds=20,
            )
            return (probe in json.dumps(events[0]),)

        converged: Final = eventually(now_raw, lambda values: values[0], seconds=120)
        assert converged[0]


def test_f4_five_identical_failures_log_once_each(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        call_ids: Final = tuple(_call_id(_chat(rig, model, [{"role": "user", "content": secret}])) for _ in range(5))
        assert len(set(call_ids)) == 5
        for call_id in call_ids:
            eventually(
                lambda cid=call_id: tuple(
                    event for event in rig.failure_events(model) if event.get("litellm_call_id") == cid
                ),
                lambda values: len(values) == 1,
                seconds=30,
            )
            assert _spend_row(call_id)["status"] == "failure", call_id
