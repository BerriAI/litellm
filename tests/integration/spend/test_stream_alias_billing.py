"""A model name that only matches a capability rule never zeroes the deployment's price (LIT-9065).

The proxy restamps every streamed chunk with the client's alias, so end-of-stream cost calculation can see
"claude-opus-4.8-<digits>" before the deployment's model. That name is no cost-map key but matches the claude
capability generalization rules, whose model info carries no prices, so the dotted alias must bill exactly what
the plain alias "integration-<hex>" bills at the same deployment rates. The same holds for a deployment whose
model_info.base_model only matches a rule, on every endpoint and client
"""

import asyncio
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final
from uuid import uuid4

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


def _sse_event(name: str, payload: dict[str, JsonValue]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _anthropic_stream(request: Request) -> Reply:
    assert request.target.endswith("/v1/messages"), request.target
    body: Final = json.loads(request.body)
    assert body["model"] == "claude-opus-4-8" and body["stream"] is True, body
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid4().hex[:12]}",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-4-8",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 30, "output_tokens": 1},
                    },
                },
            ),
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 40},
                },
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        ),
    )


def _deployment(
    scenario: Scenario,
    model_name: str,
    litellm_params: dict[str, JsonValue],
    model_info: dict[str, JsonValue] | None = None,
) -> str:
    created: Final = scenario.gateway.post(
        "/model/new", {"model_name": model_name, "litellm_params": litellm_params, "model_info": model_info or {}}
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return model_name


def _streamed_spend(gateway: Gateway, scenario: Scenario, model: str, content: str) -> dict[str, JsonValue]:
    key: Final = scenario.key(models=[model])
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        key=key,
    )
    assert response.status_code == 200, response.text
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _listed_deployments(gateway: Gateway, model_name: str) -> tuple[dict[str, JsonValue], ...]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    return tuple(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model_name)


def _deployment_pricing(gateway: Gateway, model_name: str) -> dict[str, JsonValue]:
    listed: Final = eventually(lambda: _listed_deployments(gateway, model_name), lambda found: len(found) == 1)
    return object_value(listed[0]["model_info"])


@pytest.mark.parametrize(
    "litellm_params",
    (
        pytest.param(
            lambda _: {"model": "vertex_ai/claude-opus-4-8@default", "mock_response": "hi"},
            id="vertex-mock-response",
        ),
        pytest.param(
            lambda wire_url: {
                "model": "anthropic/claude-opus-4-8",
                "api_key": "integration-provider-key",
                "api_base": wire_url,
            },
            id="anthropic-upstream",
        ),
    ),
)
@pytest.mark.timeout(180)
def test_streamed_alias_matching_a_capability_rule_bills_the_deployment_price(
    gateway: Gateway, litellm_params: Callable[[str], dict[str, JsonValue]]
) -> None:
    with wire_server(_anthropic_stream) as wire, gateway.scenario() as scenario:
        content: Final = f"alias billing {uuid4().hex}"
        plain_alias: Final = f"integration-{uuid4().hex}"
        rule_alias: Final = f"claude-opus-4.8-{uuid4().int % 10**8:08d}"
        exact_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, plain_alias, litellm_params(wire.url)), content
        )
        alias_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, rule_alias, litellm_params(wire.url)), content
        )

        for model_name, row in ((plain_alias, exact_row), (rule_alias, alias_row)):
            pricing: Final = _deployment_pricing(gateway, model_name)
            input_rate: Final = float(str(pricing["input_cost_per_token"]))
            output_rate: Final = float(str(pricing["output_cost_per_token"]))
            uplift: Final = float(str(pricing["regional_endpoint_uplift_multiplier"] or 1))
            assert input_rate > 0 and output_rate > 0, pricing
            assert float(str(row["spend"])) == pytest.approx(
                uplift
                * (float(str(row["prompt_tokens"])) * input_rate + float(str(row["completion_tokens"])) * output_rate)
            ), (model_name, row, pricing)


INPUT_TOKENS: Final = 30
OUTPUT_TOKENS: Final = 40
DEPLOYMENT_MODEL: Final = "anthropic/claude-opus-4-8"
ENDPOINTS: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")


def _rule_only_name() -> str:
    return f"claude-opus-4.8-{uuid4().int % 10**8:08d}"


def _anthropic_reply(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    if body.get("stream") is True:
        return _anthropic_stream(request)
    assert request.target.endswith("/v1/messages") and body["model"] == "claude-opus-4-8", (request.target, body)
    return Reply(
        body=json.dumps(
            {
                "id": f"msg_{uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS},
            }
        ).encode()
    )


def _anthropic_params(wire_url: str) -> dict[str, JsonValue]:
    return {"model": DEPLOYMENT_MODEL, "api_key": "integration-provider-key", "api_base": wire_url}


def _listed_rates(scenario: Scenario, model: str, wire_url: str) -> tuple[float, float]:
    model_name: Final = _deployment(
        scenario, f"integration-{uuid4().hex}", {**_anthropic_params(wire_url), "model": model}
    )
    pricing: Final = _deployment_pricing(scenario.gateway, model_name)
    uplift: Final = float(str(pricing.get("regional_endpoint_uplift_multiplier") or 1))
    rates: Final = (
        uplift * float(str(pricing["input_cost_per_token"])),
        uplift * float(str(pricing["output_cost_per_token"])),
    )
    assert rates[0] > 0 and rates[1] > 0, pricing
    return rates


def _body(path: str, model: str, content: str, stream: bool) -> dict[str, JsonValue]:
    match path:
        case "/v1/chat/completions":
            return {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "stream": stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
            }
        case "/v1/messages":
            return {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": content}],
                "stream": stream,
            }
        case _:
            return {"model": model, "input": content, "stream": stream}


def _spend_rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, spend, prompt_tokens, completion_tokens, status, cache_hit FROM "LiteLLM_SpendLogs"'
            ' WHERE api_key=%s ORDER BY "startTime"',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda values: len(values) == count,
        seconds=90,
    )


def _rule_only_base_model_deployment(scenario: Scenario, wire_url: str) -> str:
    return _deployment(
        scenario, f"integration-{uuid4().hex}", _anthropic_params(wire_url), {"base_model": _rule_only_name()}
    )


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.timeout(180)
def test_rule_only_base_model_bills_the_deployment_price(gateway: Gateway, path: str, stream: bool) -> None:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        input_rate, output_rate = _listed_rates(scenario, DEPLOYMENT_MODEL, wire.url)
        model: Final = _rule_only_base_model_deployment(scenario, wire.url)
        key: Final = scenario.key(models=[model])

        response: Final = gateway.request(
            "POST", path, _body(path, model, f"base model {uuid4().hex}", stream), key=key
        )

        assert response.status_code == 200, response.text
        row: Final = _spend_rows(key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (INPUT_TOKENS, OUTPUT_TOKENS), row
        assert float(str(row["spend"])) == pytest.approx(INPUT_TOKENS * input_rate + OUTPUT_TOKENS * output_rate), row


def _openai_sync_chat_stream(base_url: str, key: str, model: str, content: str) -> None:
    with openai.OpenAI(base_url=f"{base_url}/v1", api_key=key, max_retries=0) as client:
        chunks: Final = tuple(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                stream=True,
                stream_options={"include_usage": True},
            )
        )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "hi", chunks


def _openai_async_responses(base_url: str, key: str, model: str, content: str) -> None:
    async def call() -> str:
        async with openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=key, max_retries=0) as client:
            return (await client.responses.create(model=model, input=content)).output_text

    assert asyncio.run(call()) == "hi"


def _anthropic_async_messages_stream(base_url: str, key: str, model: str, content: str) -> None:
    async def call() -> int:
        async with anthropic.AsyncAnthropic(base_url=base_url, api_key=key, max_retries=0) as client:
            async with client.messages.stream(
                model=model, max_tokens=64, messages=[{"role": "user", "content": content}]
            ) as stream:
                return (await stream.get_final_message()).usage.output_tokens

    assert asyncio.run(call()) == OUTPUT_TOKENS


@pytest.mark.parametrize(
    "client_call",
    (
        pytest.param(_openai_sync_chat_stream, id="openai-sync-chat-stream"),
        pytest.param(_openai_async_responses, id="openai-async-responses"),
        pytest.param(_anthropic_async_messages_stream, id="anthropic-async-messages-stream"),
    ),
)
@pytest.mark.timeout(180)
def test_rule_only_base_model_bills_the_deployment_price_through_the_sdks(
    gateway: Gateway, client_call: Callable[[str, str, str, str], None]
) -> None:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        input_rate, output_rate = _listed_rates(scenario, DEPLOYMENT_MODEL, wire.url)
        model: Final = _rule_only_base_model_deployment(scenario, wire.url)
        key: Final = scenario.key(models=[model])

        client_call(str(gateway.client.base_url).rstrip("/"), key, model, f"sdk {uuid4().hex}")

        row: Final = _spend_rows(key, 1)[0]
        assert float(str(row["spend"])) == pytest.approx(INPUT_TOKENS * input_rate + OUTPUT_TOKENS * output_rate), row


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
@pytest.mark.timeout(180)
def test_custom_pricing_still_beats_a_rule_only_base_model(gateway: Gateway, stream: bool) -> None:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(
            scenario,
            f"integration-{uuid4().hex}",
            {**_anthropic_params(wire.url), "input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
            {"base_model": _rule_only_name()},
        )
        key: Final = scenario.key(models=[model])
        path: Final = "/v1/chat/completions"

        response: Final = gateway.request("POST", path, _body(path, model, f"custom {uuid4().hex}", stream), key=key)

        assert response.status_code == 200, response.text
        assert float(str(_spend_rows(key, 1)[0]["spend"])) == pytest.approx(30 * 0.001 + 40 * 0.002)


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
@pytest.mark.timeout(180)
def test_priced_base_model_still_bills_its_own_price(gateway: Gateway, stream: bool) -> None:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        input_rate, output_rate = _listed_rates(scenario, "anthropic/claude-haiku-4-5", wire.url)
        model: Final = _deployment(
            scenario, f"integration-{uuid4().hex}", _anthropic_params(wire.url), {"base_model": "claude-haiku-4-5"}
        )
        key: Final = scenario.key(models=[model])
        path: Final = "/v1/chat/completions"

        response: Final = gateway.request("POST", path, _body(path, model, f"priced {uuid4().hex}", stream), key=key)

        assert response.status_code == 200, response.text
        row: Final = _spend_rows(key, 1)[0]
        assert float(str(row["spend"])) == pytest.approx(INPUT_TOKENS * input_rate + OUTPUT_TOKENS * output_rate), row


@pytest.mark.parametrize(
    "base_model",
    (
        pytest.param("", id="empty"),
        pytest.param(f"claude-opus-4.8-{'9' * 5000}", id="5kb-rule-only"),
        pytest.param(f"integration-unmapped-{uuid4().hex}", id="unmapped-no-rule"),
    ),
)
@pytest.mark.timeout(180)
def test_odd_base_model_values_bill_the_deployment_price(gateway: Gateway, base_model: str) -> None:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        input_rate, output_rate = _listed_rates(scenario, DEPLOYMENT_MODEL, wire.url)
        model: Final = _deployment(
            scenario, f"integration-{uuid4().hex}", _anthropic_params(wire.url), {"base_model": base_model}
        )
        key: Final = scenario.key(models=[model])
        path: Final = "/v1/chat/completions"

        response: Final = gateway.request("POST", path, _body(path, model, f"odd {uuid4().hex}", True), key=key)

        assert response.status_code == 200, response.text
        row: Final = _spend_rows(key, 1)[0]
        assert float(str(row["spend"])) == pytest.approx(INPUT_TOKENS * input_rate + OUTPUT_TOKENS * output_rate), row


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.timeout(180)
def test_upstream_failure_on_a_rule_only_base_model_logs_a_zero_spend_failure(gateway: Gateway, path: str) -> None:
    failure: Final = Reply(
        status=500, body=b'{"type":"error","error":{"type":"api_error","message":"integration upstream down"}}'
    )
    with wire_server(lambda _: failure) as wire, gateway.scenario() as scenario:
        model: Final = _rule_only_base_model_deployment(scenario, wire.url)
        key: Final = scenario.key(models=[model])

        response: Final = gateway.request("POST", path, _body(path, model, f"down {uuid4().hex}", False), key=key)

        assert response.status_code == 500, response.text
        row: Final = _spend_rows(key, 1)[0]
        assert (row["status"], float(str(row["spend"]))) == ("failure", 0.0), row


@pytest.mark.timeout(180)
def test_cache_hit_on_a_rule_only_base_model_bills_only_the_first_call(gateway: Gateway) -> None:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        input_rate, output_rate = _listed_rates(scenario, DEPLOYMENT_MODEL, wire.url)
        model: Final = _rule_only_base_model_deployment(scenario, wire.url)
        key: Final = scenario.key(models=[model])
        path: Final = "/v1/chat/completions"
        body: Final = _body(path, model, f"cached {uuid4().hex}", False)

        responses: Final = tuple(gateway.request("POST", path, body, key=key) for _ in range(2))

        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        rows: Final = _spend_rows(key, 2)
        assert [(row["cache_hit"], float(str(row["spend"]))) for row in rows] == [
            ("None", pytest.approx(INPUT_TOKENS * input_rate + OUTPUT_TOKENS * output_rate)),
            ("True", 0.0),
        ], rows
        assert len([request for request in wire.drain() if request.target.endswith("/v1/messages")]) == 1


@pytest.mark.timeout(300)
def test_burst_through_an_upstream_outage_bills_every_recovered_request_once(gateway: Gateway) -> None:
    outage: Final = threading.Event()
    overloaded: Final = Reply(status=529, body=b'{"type":"error","error":{"type":"overloaded_error","message":"x"}}')
    burst: Final = tuple((path, stream) for path in ENDPOINTS for stream in (False, True)) * 4
    with (
        wire_server(lambda request: overloaded if outage.is_set() else _anthropic_reply(request)) as wire,
        gateway.scenario() as scenario,
    ):
        input_rate, output_rate = _listed_rates(scenario, DEPLOYMENT_MODEL, wire.url)
        model: Final = _rule_only_base_model_deployment(scenario, wire.url)
        key: Final = scenario.key(models=[model])

        def send(cell: tuple[str, bool]) -> httpx.Response:
            return gateway.request("POST", cell[0], _body(cell[0], model, f"burst {uuid4().hex}", cell[1]), key=key)

        outage.set()
        with ThreadPoolExecutor(max_workers=len(burst)) as pool:
            during: Final = tuple(pool.map(send, burst))
        outage.clear()
        with ThreadPoolExecutor(max_workers=len(burst)) as pool:
            after: Final = tuple(pool.map(send, burst))

        assert all(response.status_code != 200 for response in during), [r.status_code for r in during]
        assert [response.status_code for response in after] == [200] * len(burst), [r.text for r in after]
        rows: Final = _spend_rows(key, 2 * len(burst))
        succeeded: Final = tuple(row for row in rows if row["status"] == "success")
        assert len({row["request_id"] for row in succeeded}) == len(succeeded) == len(burst), rows
        assert {float(str(row["spend"])) for row in rows if row["status"] != "success"} == {0.0}, rows
        assert all(
            float(str(row["spend"])) == pytest.approx(INPUT_TOKENS * input_rate + OUTPUT_TOKENS * output_rate)
            for row in succeeded
        ), succeeded
