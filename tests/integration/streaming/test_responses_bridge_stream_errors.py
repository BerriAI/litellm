import json
import socket
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_stream import (
    AZURE_TARGET,
    OPENAI_TARGET,
    RATE_LIMIT_MESSAGE,
    azure_rate_limit,
    chat_content,
    created,
    delta,
    error_event,
    failed,
    frame,
    function_tools,
    healthy_stream,
    rate_limited_stream,
    serve,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_RATE_LIMIT_PREFIX: Final = "litellm.RateLimitError: "
_SENTINEL_PREFIX: Final = "litellm.MidStreamFallbackError: "
_PRIMARY: Final = "bridged-primary"
_SPARE: Final = "bridged-spare"
pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


def chat_body(model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": marker}],
        "tools": function_tools(),
        "num_retries": 0,
        "cache": {"no-cache": True},
        **extra,
    }


def messages_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": marker}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ],
        "num_retries": 0,
        "cache": {"no-cache": True},
    }


def error_body(response: httpx.Response) -> Mapping[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def data_frames(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def spend_row(call_id: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT litellm_call_id, status, model_group FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s',
            (call_id,),
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert len(rows) == 1, rows
    return rows[0]


def assert_provider_typed_rate_limit(error: Mapping[str, JsonValue]) -> None:
    assert error["type"] == "throttling_error", error
    assert str(error["code"]) == "429", error
    message: Final = string_value(error["message"])
    assert message.startswith(_RATE_LIMIT_PREFIX) and RATE_LIMIT_MESSAGE in message, message
    assert _SENTINEL_PREFIX not in message, message


def assert_failed_once(wire: Wire, call_id: str, model: str, attempts: int = 1) -> tuple[Request, ...]:
    received: Final = wire.drain()
    assert len(received) == attempts, [request.target for request in received]
    row: Final = spend_row(call_id)
    assert row["status"] == "failure" and row["model_group"] == model, row
    return received


def test_bridged_azure_in_stream_rate_limit_reaches_the_openai_sdk_as_a_throttling_error(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(identity), AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure/gpt-6", api_base=wire.url, api_key="synthetic-azure-key")
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        with pytest.raises(openai.RateLimitError) as raised:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": identity}],
                tools=function_tools(),
                stream=True,
                extra_body={"num_retries": 0, "cache": {"no-cache": True}},
            )
        assert raised.value.status_code == 429
        assert_provider_typed_rate_limit(object_value(raised.value.body))
        assert_failed_once(wire, raised.value.response.headers["x-litellm-call-id"], model)


async def test_bridged_openai_in_stream_rate_limit_reaches_the_async_openai_sdk_as_a_throttling_error(
    gateway: Gateway,
) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(identity), OPENAI_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-5.3-codex", api_base=wire.url, api_key="synthetic-openai-key")
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        with pytest.raises(openai.RateLimitError) as raised:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": identity}],
                stream=True,
                extra_body={"num_retries": 0, "cache": {"no-cache": True}},
            )
        assert raised.value.status_code == 429
        assert_provider_typed_rate_limit(object_value(raised.value.body))
        assert_failed_once(wire, raised.value.response.headers["x-litellm-call-id"], model)


def _chat(gateway: Gateway, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", body)


@dataclass(frozen=True, slots=True)
class FallbackProxy:
    gateway: Gateway
    primary_port: int
    spare_port: int


def _free_ports(count: int) -> tuple[int, ...]:
    with ExitStack() as reserved:
        sockets: Final = tuple(reserved.enter_context(socket.socket()) for _ in range(count))
        for reserve in sockets:
            reserve.bind(("127.0.0.1", 0))
        return tuple(reserve.getsockname()[1] for reserve in sockets)


def _fallback_config(directory: Path, primary_port: int, spare_port: int) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    deployments: Final = [
        {
            "model_name": name,
            "litellm_params": {
                "model": "azure/gpt-6",
                "api_base": f"http://127.0.0.1:{port}",
                "api_key": "synthetic-azure-key",
            },
        }
        for name, port in ((_PRIMARY, primary_port), (_SPARE, spare_port))
    ]
    router_settings: Final = {"num_retries": 0, "disable_cooldowns": True, "fallbacks": [{_PRIMARY: [_SPARE]}]}
    path: Final = directory / "bridged-fallbacks.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": deployments, "router_settings": router_settings}))
    return path


@pytest.fixture(scope="module")
def fallback_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FallbackProxy]:
    directory: Final = tmp_path_factory.mktemp("bridged-fallbacks")
    primary_port, spare_port = _free_ports(2)
    with (
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=_fallback_config(directory, primary_port, spare_port)) as owned,
    ):
        yield FallbackProxy(owned, primary_port, spare_port)


def test_bridged_in_stream_rate_limit_falls_back_to_the_healthy_deployment(fallback_proxy: FallbackProxy) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    with (
        wire_server(serve(rate_limited_stream(identity), AZURE_TARGET), port=fallback_proxy.primary_port) as primary,
        wire_server(
            serve(healthy_stream(identity, "fallback answer"), AZURE_TARGET), port=fallback_proxy.spare_port
        ) as spare,
    ):
        response: Final = _chat(fallback_proxy.gateway, chat_body(_PRIMARY, identity))
        assert response.status_code == 200, response.text
        content: Final = chat_content(data_frames(response.text))
        assert content == "fallback answer", response.text
        assert len(primary.drain()) == 1 and len(spare.drain()) == 1
        row: Final = spend_row(response.headers["x-litellm-call-id"])
        assert row["status"] == "success" and row["model_group"] == _SPARE, row


def test_bridged_in_stream_rate_limit_whose_fallback_is_also_rate_limited_answers_a_throttling_error(
    fallback_proxy: FallbackProxy,
) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    with (
        wire_server(serve(rate_limited_stream(identity), AZURE_TARGET), port=fallback_proxy.primary_port) as primary,
        wire_server(serve(rate_limited_stream(identity), AZURE_TARGET), port=fallback_proxy.spare_port) as spare,
    ):
        response: Final = _chat(fallback_proxy.gateway, chat_body(_PRIMARY, identity))
        assert response.status_code == 429, response.text
        assert_provider_typed_rate_limit(error_body(response))
        assert len(primary.drain()) == 1 and len(spare.drain()) == 1
        assert spend_row(response.headers["x-litellm-call-id"])["status"] == "failure"


def _in_stream_error_status(gateway: Gateway, stream: tuple[bytes, ...]) -> tuple[httpx.Response, str]:
    with wire_server(serve(stream, AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure/gpt-6", api_base=wire.url, api_key="synthetic-azure-key")
        response: Final = _chat(gateway, chat_body(model, uuid.uuid4().hex))
        assert_failed_once(wire, response.headers["x-litellm-call-id"], model)
        return response, model


def test_bridged_in_stream_server_error_reaches_the_client_as_the_provider_error(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    stream: Final = (
        frame(created(identity)),
        frame(error_event({"type": "server_error", "code": "server_error", "message": "The server had an error"})),
        frame(failed(identity, "server_error", "The server had an error")),
    )
    response, _ = _in_stream_error_status(gateway, stream)
    assert response.status_code == 500, response.text
    error: Final = error_body(response)
    message: Final = string_value(error["message"])
    assert str(error["code"]) == "500", error
    assert message.startswith("litellm.APIError: ") and "The server had an error" in message, message
    assert _SENTINEL_PREFIX not in message, message


def test_bridged_in_stream_invalid_prompt_is_a_bad_request_on_both_legs(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    stream: Final = (
        frame(created(identity)),
        frame(error_event({"type": "invalid_request_error", "code": "invalid_prompt", "message": "Invalid prompt"})),
        frame(failed(identity, "invalid_prompt", "Invalid prompt")),
    )
    response, _ = _in_stream_error_status(gateway, stream)
    assert response.status_code == 400, response.text
    error: Final = error_body(response)
    message: Final = string_value(error["message"])
    assert str(error["code"]) == "400", error
    assert message.startswith("litellm.BadRequestError: ") and "Invalid prompt" in message, message
    assert _SENTINEL_PREFIX not in message, message


def test_bridged_error_event_without_an_error_object_is_a_provider_typed_internal_error(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    stream: Final = (frame(created(identity)), frame(error_event(None)))
    response, _ = _in_stream_error_status(gateway, stream)
    assert response.status_code == 500, response.text
    error: Final = error_body(response)
    message: Final = string_value(error["message"])
    assert str(error["code"]) == "500", error
    assert message.startswith("litellm.APIError: ") and "Response API in-stream error" in message, message
    assert _SENTINEL_PREFIX not in message, message


def test_bridged_error_event_with_a_numeric_code_is_a_throttling_error(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    stream: Final = (frame(created(identity)), frame(error_event({"code": "429", "message": RATE_LIMIT_MESSAGE})))
    response, _ = _in_stream_error_status(gateway, stream)
    assert response.status_code == 429, response.text
    assert_provider_typed_rate_limit(error_body(response))


def test_bridged_response_failed_without_an_error_event_is_a_throttling_error(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    stream: Final = (frame(created(identity)), frame(failed(identity, "rate_limit_exceeded", RATE_LIMIT_MESSAGE)))
    response, _ = _in_stream_error_status(gateway, stream)
    assert response.status_code == 429, response.text
    assert_provider_typed_rate_limit(error_body(response))


def test_bridged_rate_limit_after_output_is_a_provider_typed_error_frame_behind_the_text(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    stream: Final = (
        frame(created(identity)),
        frame(delta(identity, "Hello")),
        frame(delta(identity, " there")),
        frame(error_event(azure_rate_limit())),
        frame(failed(identity, "rate_limit_exceeded", RATE_LIMIT_MESSAGE)),
    )
    response, _ = _in_stream_error_status(gateway, stream)
    assert response.status_code == 200, response.text
    frames: Final = data_frames(response.text)
    content: Final = chat_content(frames)
    assert content == "Hello there", response.text
    error: Final = object_value(frames[-1]["error"])
    assert str(error["code"]) == "429", error
    assert error["type"] == "throttling_error", error
    message: Final = string_value(error["message"])
    assert message.startswith(_RATE_LIMIT_PREFIX) and RATE_LIMIT_MESSAGE in message, message
    assert _SENTINEL_PREFIX not in message, message


def test_bridged_transport_drop_after_response_created_is_a_500_without_the_sentinel(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.target.startswith(AZURE_TARGET), request.target
        return Reply(
            content_type="text/event-stream",
            chunks=(frame(created(identity)), frame(delta(identity, "never sent"))),
            abort_after=1,
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure/gpt-6", api_base=wire.url, api_key="synthetic-azure-key")
        response: Final = _chat(gateway, chat_body(model, identity))
        assert response.status_code == 500, response.text
        error: Final = error_body(response)
        message: Final = string_value(error["message"])
        assert str(error["code"]) == "500", error
        assert "never sent" not in response.text
        assert _SENTINEL_PREFIX not in message, message
        assert_failed_once(wire, response.headers["x-litellm-call-id"], model)


def test_plain_chat_http_rate_limit_is_a_throttling_error_on_both_legs(gateway: Gateway) -> None:
    identity: Final = uuid.uuid4().hex
    denial: Final = {"error": {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}}

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/chat/completions", request.target
        return Reply(status=429, body=json.dumps(denial).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        with pytest.raises(openai.RateLimitError) as raised:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": identity}],
                stream=True,
                extra_body={"num_retries": 0, "cache": {"no-cache": True}},
            )
        assert raised.value.status_code == 429
        error: Final = object_value(raised.value.body)
        assert error["type"] == "throttling_error" and str(error["code"]) == "429", error
        message: Final = string_value(error["message"])
        assert message.startswith(_RATE_LIMIT_PREFIX) and "Rate limit reached" in message, message
        assert_failed_once(wire, raised.value.response.headers["x-litellm-call-id"], model)


def sse_events(text: str) -> tuple[tuple[str, Mapping[str, JsonValue]], ...]:
    def parse(block: str) -> tuple[str, Mapping[str, JsonValue]]:
        lines: Final = block.splitlines()
        event: Final = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data: Final = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        return event, JSON_OBJECT.validate_json(data)

    return tuple(parse(block) for block in text.strip().split("\n\n") if "event: " in block)


def assert_messages_errorframe(events: Sequence[tuple[str, Mapping[str, JsonValue]]]) -> None:
    assert events[0][0] == "message_start", events
    assert events[-1][0] == "error", events
    error: Final = object_value(events[-1][1]["error"])
    assert error["type"] == "rate_limit_error", error
    message: Final = string_value(error["message"])
    assert message.startswith(_RATE_LIMIT_PREFIX) and RATE_LIMIT_MESSAGE in message, message
    assert message.count(_RATE_LIMIT_PREFIX) == 1 and _SENTINEL_PREFIX not in message, message


def test_messages_over_the_bridged_stream_carry_the_provider_error_once_in_the_errorframe(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(identity), AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure/gpt-6", api_base=wire.url, api_key="synthetic-azure-key")
        response: Final = gateway.request(
            "POST", "/v1/messages", messages_body(model, identity), headers={"anthropic-version": "2023-06-01"}
        )
        assert response.status_code == 200, response.text
        assert_messages_errorframe(sse_events(response.text))
        assert len(wire.drain()) == 1


async def _consume_anthropic_stream(client: anthropic.AsyncAnthropic, model: str, identity: str) -> None:
    async with client.messages.stream(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": identity}],
        tools=[
            {
                "name": "get_weather",
                "description": "Weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        extra_body={"num_retries": 0, "cache": {"no-cache": True}},
    ) as stream:
        async for _ in stream:
            pass


async def test_messages_over_the_bridged_stream_raise_the_error_frame_in_the_anthropic_sdk(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(identity), AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure/gpt-6", api_base=wire.url, api_key="synthetic-azure-key")
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        with pytest.raises(anthropic.APIStatusError) as raised:
            await _consume_anthropic_stream(client, model, identity)
        body: Final = object_value(raised.value.body)
        assert_messages_errorframe((("message_start", {}), ("error", body)))
        assert len(wire.drain()) == 1


def test_native_responses_stream_forwards_the_failed_response_on_both_legs(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(identity), AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure/gpt-6", api_base=wire.url, api_key="synthetic-azure-key")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": identity, "stream": True, "cache": {"no-cache": True}},
        )
        assert response.status_code == 200, response.text
        frames: Final = data_frames(response.text)
        assert [frame["type"] for frame in frames] == ["response.created", "response.failed"], response.text
        failed: Final = object_value(frames[-1]["response"])
        assert failed["status"] == "failed", failed
        assert object_value(failed["error"])["code"] == "rate_limit_exceeded", failed
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        assert len(wire.drain()) == 1
        assert spend_row(response.headers["x-litellm-call-id"])["status"] == "failure"
