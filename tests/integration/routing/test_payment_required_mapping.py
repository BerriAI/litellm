from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable, Mapping
from typing import Final, TypeVar

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply, openai_error
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

T = TypeVar("T")

PROXY_WORKERS: Final = int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1"))
WORKER_SYNC_SECONDS: Final = 0.0 if PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + 5.0
FRESH_CONNECTION: Final = {"Connection": "close"}
UNKNOWN_MODEL: Final = "Invalid model name passed in model="
PAYMENT_REQUIRED: Final = "litellm.PaymentRequiredError: AnthropicException"
BAD_REQUEST: Final = "litellm.BadRequestError: AnthropicException"
ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-5"
OPENAI_MODEL: Final = "openai/gpt-5.4"
CREDIT_BALANCE_TEXT: Final = "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."
FAILURE_ROW: Final = (
    "SELECT status, model_id, metadata::jsonb -> 'error_information' AS error "
    'FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
)
MODEL_INFO_PROBE: Final = ("GET", "/v1/models")


def is_model_info_probe(request: Request) -> bool:
    return (request.method, request.target) == MODEL_INFO_PROBE


def model_list_reply() -> Reply:
    return Reply(body=b'{"object": "list", "data": []}')


def openai_refusal(request: Request) -> Reply:
    return model_list_reply() if is_model_info_probe(request) else openai_error(402)


def provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if not is_model_info_probe(request))


def new_marker() -> str:
    return f"pr402-{uuid.uuid4().hex}"


def new_group() -> str:
    return f"integration-402-{uuid.uuid4().hex[:12]}"


def anthropic_error(status: int, message: str, *, content_type: str = "application/json") -> Reply:
    body: Final = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}})
    return Reply(status=status, body=body.encode(), content_type=content_type)


def anthropic_params(wire: Wire) -> dict[str, JsonValue]:
    return {"model": ANTHROPIC_MODEL, "api_base": wire.url, "api_key": "synthetic-anthropic-key"}


def openai_params(wire: Wire) -> dict[str, JsonValue]:
    return {"model": OPENAI_MODEL, "api_base": wire.url + "/v1", "api_key": "synthetic-openai-key"}


def deployment(
    scenario: Scenario,
    group: str,
    litellm_params: Mapping[str, JsonValue],
    *,
    model_info: Mapping[str, JsonValue] | None = None,
) -> str:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": group,
            "litellm_params": dict(litellm_params),
            "model_info": dict(model_info) if model_info is not None else {},
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def settled(send: Callable[[], T], text: Callable[[T], str]) -> T:
    return eventually(send, lambda observed: UNKNOWN_MODEL not in text(observed), seconds=WORKER_SYNC_SECONDS + 10)


def post(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return settled(
        lambda: gateway.request("POST", path, body, headers=FRESH_CONNECTION), lambda response: response.text
    )


def chat_body(group: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": group, "messages": [{"role": "user", "content": marker}], **extra}


def upstream_calls(wire: Wire, marker: str) -> tuple[str, ...]:
    received: Final = provider_calls(wire)
    assert all(marker.encode() in request.body for request in received), received
    return tuple(f"{request.method} {request.target}" for request in received)


def failure_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(FAILURE_ROW, (call_id,)), lambda found: len(found) == 1, seconds=30)
    return rows[0]


def assert_payment_required_row(call_id: str, deployment_id: str) -> None:
    row: Final = failure_row(call_id)
    error: Final = object_value(row["error"])
    assert row["status"] == "failure", row
    assert row["model_id"] == deployment_id, row
    assert (error["error_class"], error["error_code"], error["llm_provider"]) == (
        "PaymentRequiredError",
        "402",
        "anthropic",
    ), row


def openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        api_key=gateway.key,
        max_retries=0,
        default_headers=FRESH_CONNECTION,
    )


def anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url).rstrip("/"),
        api_key=gateway.key,
        max_retries=0,
        default_headers=FRESH_CONNECTION,
    )


def test_chat_completions_openai_sdk_gets_402_payment_required(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(lambda _: anthropic_error(402, f"scripted 402 {marker}")) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))
        client: Final = openai_client(gateway)

        def send() -> openai.APIStatusError:
            with pytest.raises(openai.APIStatusError) as caught:
                client.chat.completions.create(model=group, messages=[{"role": "user", "content": marker}])
            return caught.value

        error: Final = settled(send, lambda failure: failure.message)
        assert error.status_code == 402, error.message
        assert PAYMENT_REQUIRED in error.message, error.message
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        assert_payment_required_row(error.response.headers["x-litellm-call-id"], deployment_id)


async def drain_chat_stream(client: openai.AsyncOpenAI, group: str, marker: str) -> None:
    stream: Final = await client.chat.completions.create(
        model=group, messages=[{"role": "user", "content": marker}], stream=True
    )
    async for _ in stream:
        pass


async def drain_messages_stream(client: anthropic.AsyncAnthropic, group: str, marker: str) -> None:
    async with client.messages.stream(
        model=group, max_tokens=32, messages=[{"role": "user", "content": marker}]
    ) as stream:
        async for _ in stream:
            pass


def test_chat_completions_stream_openai_async_sdk_gets_402_payment_required(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(lambda _: anthropic_error(402, f"scripted 402 {marker}")) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))

        async def attempt() -> openai.APIStatusError:
            async with openai.AsyncOpenAI(
                base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
                api_key=gateway.key,
                max_retries=0,
                default_headers=FRESH_CONNECTION,
            ) as client:
                with pytest.raises(openai.APIStatusError) as caught:
                    await drain_chat_stream(client, group, marker)
                return caught.value

        error: Final = settled(lambda: asyncio.run(attempt()), lambda failure: failure.message)
        assert error.status_code == 402, error.message
        assert PAYMENT_REQUIRED in error.message, error.message
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        assert_payment_required_row(error.response.headers["x-litellm-call-id"], deployment_id)


def test_messages_anthropic_sdk_gets_402_payment_required(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(lambda _: anthropic_error(402, f"scripted 402 {marker}")) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))
        client: Final = anthropic_client(gateway)

        def send() -> anthropic.APIStatusError:
            with pytest.raises(anthropic.APIStatusError) as caught:
                client.messages.create(model=group, max_tokens=32, messages=[{"role": "user", "content": marker}])
            return caught.value

        error: Final = settled(send, lambda failure: failure.message)
        assert error.status_code == 402, error.message
        assert PAYMENT_REQUIRED in error.message, error.message
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        assert_payment_required_row(error.response.headers["x-litellm-call-id"], deployment_id)


def test_messages_stream_anthropic_async_sdk_gets_402_payment_required(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(lambda _: anthropic_error(402, f"scripted 402 {marker}")) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))

        async def attempt() -> anthropic.APIStatusError:
            async with anthropic.AsyncAnthropic(
                base_url=str(gateway.client.base_url).rstrip("/"),
                api_key=gateway.key,
                max_retries=0,
                default_headers=FRESH_CONNECTION,
            ) as client:
                with pytest.raises(anthropic.APIStatusError) as caught:
                    await drain_messages_stream(client, group, marker)
                return caught.value

        error: Final = settled(lambda: asyncio.run(attempt()), lambda failure: failure.message)
        assert error.status_code == 402, error.message
        assert PAYMENT_REQUIRED in error.message, error.message
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        assert_payment_required_row(error.response.headers["x-litellm-call-id"], deployment_id)


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_responses_httpx_gets_402_payment_required(gateway: Gateway, stream: bool) -> None:
    marker: Final = new_marker()
    with wire_server(lambda _: anthropic_error(402, f"scripted 402 {marker}")) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))
        response: Final = post(gateway, "/v1/responses", {"model": group, "input": marker, "stream": stream})
        assert response.status_code == 402, response.text
        assert PAYMENT_REQUIRED in response.text, response.text
        assert response.headers["content-type"].startswith("application/json"), response.headers
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        assert_payment_required_row(response.headers["x-litellm-call-id"], deployment_id)


def test_402_with_a_text_body_still_maps_to_payment_required(gateway: Gateway) -> None:
    marker: Final = new_marker()
    text: Final = f"credits exhausted {marker}"
    with (
        wire_server(lambda _: Reply(status=402, body=text.encode(), content_type="text/plain")) as wire,
        gateway.scenario() as scenario,
    ):
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))
        response: Final = post(gateway, "/v1/chat/completions", chat_body(group, marker))
        assert response.status_code == 402, response.text
        assert PAYMENT_REQUIRED in response.text, response.text
        assert text in response.text, response.text
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        assert_payment_required_row(response.headers["x-litellm-call-id"], deployment_id)


def test_402_with_an_empty_body_still_maps_to_payment_required(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(lambda _: Reply(status=402, body=b"")) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))
        response: Final = post(gateway, "/v1/chat/completions", chat_body(group, marker))
        assert response.status_code == 402, response.text
        assert PAYMENT_REQUIRED in response.text, response.text
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        assert_payment_required_row(response.headers["x-litellm-call-id"], deployment_id)


def test_402_with_a_5kb_message_maps_and_a_healthy_deployment_keeps_serving(gateway: Gateway) -> None:
    marker: Final = new_marker()
    long_message: Final = f"{marker} " + "x" * 5120
    with (
        wire_server(lambda _: anthropic_error(402, long_message)) as broke,
        wire_server(
            lambda _: chat_reply(f"chatcmpl-{marker}", OPENAI_MODEL, f"served {marker}", stream=False)
        ) as healthy,
        gateway.scenario() as scenario,
    ):
        broke_group: Final = new_group()
        healthy_group: Final = new_group()
        deployment_id: Final = deployment(scenario, broke_group, anthropic_params(broke))
        deployment(scenario, healthy_group, openai_params(healthy))
        refused: Final = post(gateway, "/v1/chat/completions", chat_body(broke_group, marker))
        assert refused.status_code == 402, refused.text
        assert PAYMENT_REQUIRED in refused.text, refused.text
        assert "x" * 512 in refused.text, refused.text
        assert upstream_calls(broke, marker) == ("POST /v1/messages",)
        assert_payment_required_row(refused.headers["x-litellm-call-id"], deployment_id)
        served: Final = post(gateway, "/v1/chat/completions", chat_body(healthy_group, f"{marker} healthy"))
        assert served.status_code == 200, served.text
        assert served.json()["choices"][0]["message"]["content"] == f"served {marker}", served.text
        assert upstream_calls(healthy, marker) == ("POST /v1/chat/completions",)


def test_unauthenticated_request_to_a_402_deployment_is_refused_before_the_wire(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(lambda _: anthropic_error(402, f"scripted 402 {marker}")) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment(scenario, group, anthropic_params(wire))
        response: Final = gateway.client.post("/v1/chat/completions", json=chat_body(group, marker))
        assert response.status_code == 401, response.text
        assert provider_calls(wire) == ()


def test_anthropic_400_credit_balance_text_stays_a_bad_request(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with (
        wire_server(lambda _: anthropic_error(400, f"{CREDIT_BALANCE_TEXT} {marker}")) as wire,
        gateway.scenario() as scenario,
    ):
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, anthropic_params(wire))
        response: Final = post(gateway, "/v1/chat/completions", chat_body(group, marker))
        assert response.status_code == 400, response.text
        assert BAD_REQUEST in response.text, response.text
        assert upstream_calls(wire, marker) == ("POST /v1/messages",)
        row: Final = failure_row(response.headers["x-litellm-call-id"])
        error: Final = object_value(row["error"])
        assert (row["status"], row["model_id"]) == ("failure", deployment_id), row
        assert (error["error_class"], error["error_code"]) == ("BadRequestError", "400"), row


def test_openai_compatible_402_reaches_the_caller_as_402(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(openai_refusal) as wire, gateway.scenario() as scenario:
        group: Final = new_group()
        deployment_id: Final = deployment(scenario, group, openai_params(wire))
        response: Final = post(gateway, "/v1/chat/completions", chat_body(group, marker))
        assert response.status_code == 402, response.text
        assert "scripted 402" in response.text, response.text
        assert upstream_calls(wire, marker) == ("POST /v1/chat/completions",)
        row: Final = failure_row(response.headers["x-litellm-call-id"])
        error: Final = object_value(row["error"])
        assert (row["status"], row["model_id"]) == ("failure", deployment_id), row
        assert (error["error_code"], error["llm_provider"]) == ("402", "openai"), row


def test_disabled_cooldowns_keep_picking_the_402_deployment(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with (
        wire_server(lambda _: anthropic_error(402, f"scripted 402 {marker}")) as broke,
        wire_server(
            lambda _: chat_reply(f"chatcmpl-{marker}", OPENAI_MODEL, f"served {marker}", stream=False)
        ) as healthy,
        gateway.scenario() as scenario,
    ):
        group: Final = new_group()
        deployment(scenario, group, anthropic_params(broke))
        deployment(scenario, group, openai_params(healthy))
        responses: Final = tuple(
            post(gateway, "/v1/chat/completions", chat_body(group, f"{marker}-{index}")) for index in range(40)
        )
        statuses: Final = tuple(response.status_code for response in responses)
        assert statuses.count(402) >= 2 and statuses.count(200) >= 1, statuses
        assert set(statuses) == {200, 402}, statuses
        refusals: Final = tuple(response for response in responses if response.status_code == 402)
        assert all(PAYMENT_REQUIRED in response.text for response in refusals), [r.text for r in refusals]
        assert len(upstream_calls(broke, marker)) == len(refusals)
        assert len(upstream_calls(healthy, marker)) == statuses.count(200)
