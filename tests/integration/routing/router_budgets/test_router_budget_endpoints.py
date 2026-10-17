"""What a caller sees from a router budget on every endpoint, streaming and not, through every client.

Each (endpoint, client) cell owns one deployment capped at 0.01, so its first call (0.03125) crosses the cap.
The cell drives that first call to completion through the real client, checks the deployment spend in Redis
is exactly one call, then probes through the same client until the router rejects it. Probes carry
``PROVIDER_FAILURE`` so an admitted probe is a free upstream 500, never a charge. The held and abandoned
stream cells gate the upstream after the first SSE frame to observe when the charge lands.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.wire import Reply, Request, wire_server
from integration.routing.router_budgets import _rig as rig

CLIENTS: Final = ("sdk_sync", "sdk_async", "httpx")
CELLS: Final = tuple((endpoint, client) for endpoint in rig.ENDPOINTS for client in CLIENTS)
HOLD_MARKER: Final = "router-budget-hold-stream"
ABANDON_MARKER: Final = "router-budget-abandon-stream"
HELD_GATE: Final = threading.Event()
ABANDON_GATE: Final = threading.Event()
TERMINAL_FRAME: Final = {"chat": "[DONE]", "messages": "message_stop", "responses": "response.completed"}


def _gated(reply: Reply, gate: threading.Event) -> Reply:
    frames: Final = tuple(frame + b"\n\n" for frame in reply.body.split(b"\n\n") if frame)
    return Reply(content_type=reply.content_type, chunks=frames, gate_after_first=gate)


def _respond(request: Request) -> Reply:
    reply: Final = rig.upstream(request)
    if HOLD_MARKER.encode() in request.body:
        return _gated(reply, HELD_GATE)
    if ABANDON_MARKER.encode() in request.body:
        return _gated(reply, ABANDON_GATE)
    return reply


def _model(endpoint: str, client: str) -> str:
    return f"ep-{endpoint}-{client}".replace("_", "-")


@dataclass(frozen=True, slots=True)
class EndpointRig:
    gateway: Gateway
    budget: rig.BudgetRig


@pytest.fixture(scope="module")
def endpoints(tmp_path_factory: pytest.TempPathFactory) -> Iterator[EndpointRig]:
    tmp_path: Final = tmp_path_factory.mktemp("budget-endpoints")
    with gateway_from_environment() as gateway, wire_server(_respond) as upstream:
        capped: Final = tuple(
            rig.deployment(
                name,
                f"openai/{name}",
                upstream.url,
                model_id=name,
                max_budget=0.01,
                budget_duration="1d",
            )
            for name in (*(_model(endpoint, client) for endpoint, client in CELLS), "held-stream", "abandoned-stream")
        )
        config: Final = rig.write_config(tmp_path / "endpoints.yaml", capped)
        with rig.budget_proxy(gateway, tmp_path, config) as budget:
            yield EndpointRig(budget.gateway, budget)


@dataclass(frozen=True, slots=True)
class Outcome:
    status: int
    message: str
    error_class: str
    completed: bool


def _sdk_error(error: openai.APIStatusError | anthropic.APIStatusError) -> Outcome:
    return Outcome(error.status_code, str(error), type(error).__name__, False)


def _openai_sync(base_url: str, key: str, endpoint: str, model: str, text: str) -> Outcome:
    client: Final = openai.OpenAI(base_url=f"{base_url}/v1", api_key=key, max_retries=0)
    try:
        if endpoint == "chat":
            return Outcome(
                200,
                client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}]).id,
                "",
                True,
            )
        if endpoint == "chat_stream":
            chunks: Final = tuple(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
            return Outcome(200, chunks[-1].id, "", any(chunk.usage is not None for chunk in chunks))
        if endpoint == "responses":
            return Outcome(200, client.responses.create(model=model, input=text).id, "", True)
        events: Final = tuple(client.responses.create(model=model, input=text, stream=True))
        return Outcome(200, events[-1].type, "", events[-1].type == "response.completed")
    except openai.APIStatusError as error:
        return _sdk_error(error)


async def _openai_async(base_url: str, key: str, endpoint: str, model: str, text: str) -> Outcome:
    client: Final = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=key, max_retries=0)
    try:
        if endpoint == "chat":
            completion: Final = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": text}]
            )
            return Outcome(200, completion.id, "", True)
        if endpoint == "chat_stream":
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks: Final = tuple([chunk async for chunk in stream])
            return Outcome(200, chunks[-1].id, "", any(chunk.usage is not None for chunk in chunks))
        if endpoint == "responses":
            response: Final = await client.responses.create(model=model, input=text)
            return Outcome(200, response.id, "", True)
        events_stream: Final = await client.responses.create(model=model, input=text, stream=True)
        events: Final = tuple([event async for event in events_stream])
        return Outcome(200, events[-1].type, "", events[-1].type == "response.completed")
    except openai.APIStatusError as error:
        return _sdk_error(error)
    finally:
        await client.close()


def _anthropic_sync(base_url: str, key: str, endpoint: str, model: str, text: str) -> Outcome:
    client: Final = anthropic.Anthropic(base_url=base_url, api_key=key, max_retries=0)
    try:
        if endpoint == "messages":
            message: Final = client.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": text}]
            )
            return Outcome(200, message.id, "", True)
        events: Final = tuple(
            client.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": text}], stream=True
            )
        )
        return Outcome(200, events[-1].type, "", events[-1].type == "message_stop")
    except anthropic.APIStatusError as error:
        return _sdk_error(error)


async def _anthropic_async(base_url: str, key: str, endpoint: str, model: str, text: str) -> Outcome:
    client: Final = anthropic.AsyncAnthropic(base_url=base_url, api_key=key, max_retries=0)
    try:
        if endpoint == "messages":
            message: Final = await client.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": text}]
            )
            return Outcome(200, message.id, "", True)
        stream: Final = await client.messages.create(
            model=model, max_tokens=16, messages=[{"role": "user", "content": text}], stream=True
        )
        events: Final = tuple([event async for event in stream])
        return Outcome(200, events[-1].type, "", events[-1].type == "message_stop")
    except anthropic.APIStatusError as error:
        return _sdk_error(error)
    finally:
        await client.close()


def _raw(gateway: Gateway, endpoint: str, model: str, text: str) -> Outcome:
    response: Final = rig.send(gateway, endpoint, model, text)
    terminal: Final = TERMINAL_FRAME[endpoint.removesuffix("_stream")]
    completed: Final = response.status_code == 200 and (not endpoint.endswith("_stream") or terminal in response.text)
    return Outcome(response.status_code, response.text, "", completed)


def _invoke(gateway: Gateway, endpoint: str, client: str, model: str, text: str) -> Outcome:
    base_url: Final = str(gateway.client.base_url).rstrip("/")
    if client == "httpx":
        return _raw(gateway, endpoint, model, text)
    if endpoint.startswith("messages"):
        if client == "sdk_sync":
            return _anthropic_sync(base_url, gateway.key, endpoint, model, text)
        return asyncio.run(_anthropic_async(base_url, gateway.key, endpoint, model, text))
    if client == "sdk_sync":
        return _openai_sync(base_url, gateway.key, endpoint, model, text)
    return asyncio.run(_openai_async(base_url, gateway.key, endpoint, model, text))


@pytest.mark.parametrize(("endpoint", "client"), CELLS)
def test_a_deployment_over_budget_rejects_the_caller_with_429_after_one_charged_call(
    endpoints: EndpointRig, endpoint: str, client: str
) -> None:
    model: Final = _model(endpoint, client)
    first: Final = _invoke(endpoints.gateway, endpoint, client, model, f"{model} first")
    assert (first.status, first.completed) == (200, True), first

    assert endpoints.budget.settled(f"deployment_spend:{model}:1d", rig.CALL_COST) == rig.CALL_COST

    rejected: Final = eventually(
        lambda: _invoke(endpoints.gateway, endpoint, client, model, rig.probe_text(model)),
        lambda outcome: outcome.status != 500,
        seconds=30,
    )
    assert rejected.status == 429, rejected
    assert rig.BUDGET_ERROR in rejected.message, rejected
    assert f"model_id: {model}" in rejected.message, rejected
    assert rejected.error_class == ("" if client == "httpx" else "RateLimitError"), rejected
    assert endpoints.budget.redis_float(f"deployment_spend:{model}:1d") == rig.CALL_COST


def test_a_stream_is_charged_only_once_it_completes(endpoints: EndpointRig) -> None:
    body: Final = rig.body_for("chat_stream", "held-stream", f"held {HOLD_MARKER}")
    with endpoints.gateway.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {endpoints.gateway.key}"}
    ) as stream:
        lines: Final = stream.iter_lines()
        assert next(lines).startswith("data: ")
        mid_stream: Final = rig.chat(endpoints.gateway, "held-stream", rig.probe_text("held mid stream"))
        assert mid_stream.status_code == 500, mid_stream.text
        HELD_GATE.set()
        rest: Final = tuple(lines)
    assert "data: [DONE]" in rest

    assert endpoints.budget.settled("deployment_spend:held-stream:1d", rig.CALL_COST) == rig.CALL_COST
    rig.until_rejected(endpoints.gateway, "held-stream", rig.probe_text("held after"))


def test_a_stream_the_client_abandons_midway_is_still_charged(endpoints: EndpointRig) -> None:
    body: Final = rig.body_for("chat_stream", "abandoned-stream", f"abandoned {ABANDON_MARKER}")
    with httpx.Client(base_url=str(endpoints.gateway.client.base_url), trust_env=False, timeout=15) as client:
        with client.stream(
            "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {endpoints.gateway.key}"}
        ) as stream:
            assert next(stream.iter_lines()).startswith("data: ")
    ABANDON_GATE.set()

    rig.until_rejected(endpoints.gateway, "abandoned-stream", rig.probe_text("abandoned after"))
    charged: Final = eventually(
        lambda: endpoints.budget.redis_float("deployment_spend:abandoned-stream:1d"), lambda spend: spend is not None
    )
    assert charged is not None and 0 < charged <= rig.CALL_COST, charged
    assert (charged / rig.PRICE).is_integer(), charged
