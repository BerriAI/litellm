import json
import math
import os
import re
import signal
import threading
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.anthropic_thinking import JSON_OBJECT
from integration._support.client import (
    GATEWAY_LIMITS,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
)
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.authorization._hidden_alias_budget import (
    BUDGET,
    BUDGET_EXCEEDED,
    CHAT_REPLY,
    GATEWAY_BURST,
    NO_EXTRA,
    PEER_BURST,
    anthropic_client,
    assert_free_row,
    async_anthropic_client,
    async_openai_client,
    base_url,
    error_type,
    fresh_chat,
    fresh_message,
    fresh_response,
    hidden,
    install_aliases,
    landed,
    landed_all_once,
    landed_once,
    openai_client,
    remove_aliases,
    settle_candidate,
    spend_marker,
    upstream_hits,
    upstream_requests,
)
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(240)

PAID_INPUT_COST_PER_TOKEN: Final = 0.001
PAID_OUTPUT_COST_PER_TOKEN: Final = 0.002
UPSTREAM_PROMPT_TOKENS: Final = 20
UPSTREAM_COMPLETION_TOKENS: Final = 20
PAID_REPLY_COST: Final = (
    UPSTREAM_PROMPT_TOKENS * PAID_INPUT_COST_PER_TOKEN + UPSTREAM_COMPLETION_TOKENS * PAID_OUTPUT_COST_PER_TOKEN
)
HEADROOM: Final = 5.0
BURST_FAILURES: Final = 12
LIVELINESS_PROBES: Final = 4
BURST_KINDS: Final = ("chat", "chat-stream", "responses", "responses-stream", "messages", "messages-stream")
ROUTES: Final = (("chat", fresh_chat), ("responses", fresh_response), ("messages", fresh_message))
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
WORKER_PIDS: Final = TypeAdapter(tuple[int, ...])


@dataclass(frozen=True, slots=True)
class ChainRig:
    gateway: Gateway
    peer: Gateway
    paid: str
    mid_free: str
    tail_free: str
    mid_paid: str
    unpriced: str
    ghost_mid: str
    failing_free: str
    failing_provider_model: str
    entry: str
    hidden_entry: str
    shown_entry: str
    null_entry: str
    reverse_entry: str
    unpriced_entry: str
    ghost_entry: str
    failing_entry: str

    @property
    def candidates(self) -> tuple[Gateway, Gateway]:
        return (self.gateway, self.peer)


@dataclass(frozen=True, slots=True)
class GroupPrice:
    input_cost_per_token: float
    output_cost_per_token: float
    providers: tuple[str, ...]


def _free_deployment(scenario: Scenario, label: str) -> str:
    return scenario.model(
        model=f"lemonade/{label}-{uuid.uuid4().hex[:8]}", input_cost_per_token=0, output_cost_per_token=0
    )


def _paid_deployment(scenario: Scenario) -> str:
    return scenario.model(
        input_cost_per_token=PAID_INPUT_COST_PER_TOKEN, output_cost_per_token=PAID_OUTPUT_COST_PER_TOKEN
    )


def _wildcard_route(rig: ChainRig, scenario: Scenario, prefix: str, input_cost: float, output_cost: float) -> str:
    created: Final = rig.gateway.post(
        "/model/new",
        {
            "model_name": f"{prefix}/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "integration-provider-key",
                "api_base": f"{rig.gateway.upstream_url}/v1",
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
            },
            "model_info": {},
        },
    )
    identity: Final = str(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def _deployment_id(name: str) -> str:
    rows: Final = read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s', (name,))
    assert len(rows) == 1, rows
    return str(rows[0]["model_id"])


def _serving_deployments(candidate: Gateway, model: str, count: int) -> frozenset[str | None]:
    def serving(marker: str) -> str | None:
        response: Final = fresh_chat(candidate, model, candidate.key, marker)
        return response.headers.get("x-litellm-model-id") if response.status_code == 200 else None

    with ThreadPoolExecutor(max_workers=count) as pool:
        return frozenset(pool.map(serving, (f"route-{uuid.uuid4().hex}" for _ in range(count))))


def _await_served_by(candidate: Gateway, name: str, deployment: str) -> None:
    eventually(
        lambda: _serving_deployments(candidate, name, 16),
        lambda seen: seen == frozenset({deployment}),
        seconds=120,
    )


def _settle_both(rig: ChainRig, model: str, key: str, status: int, extra: Mapping[str, JsonValue] = NO_EXTRA) -> None:
    settle_candidate(rig.gateway, model, key, status, extra=extra)
    settle_candidate(rig.peer, model, key, status, burst=PEER_BURST, extra=extra)


def _exhausted_key(rig: ChainRig, scenario: Scenario) -> str:
    key: Final = scenario.key(max_budget=BUDGET)
    first: Final = fresh_chat(rig.gateway, rig.paid, key, "exhaust-" + uuid.uuid4().hex)
    assert first.status_code == 200, first.text
    _settle_both(rig, rig.paid, key, BUDGET_EXCEEDED)
    return key


def _refusal(response: httpx.Response) -> tuple[int, str]:
    return response.status_code, error_type(response)


def _assert_refused(
    rig: ChainRig,
    candidate: Gateway,
    key: str,
    model: str,
    prefix: str,
    send: Callable[[Gateway, str, str, str], httpx.Response] = fresh_chat,
) -> None:
    marker: Final = f"{prefix}-" + uuid.uuid4().hex
    refused: Final = send(candidate, model, key, marker)
    assert _refusal(refused) == (BUDGET_EXCEEDED, "budget_exceeded"), refused.text
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def _assert_refused_on_every_route(rig: ChainRig, candidate: Gateway, key: str, model: str) -> None:
    for route, send in ROUTES:
        _assert_refused(rig, candidate, key, model, f"refused-{route}", send)


def _assert_provider_failure(response: httpx.Response) -> None:
    assert response.status_code == 500, response.text
    assert "Controlled provider failure" in response.text, response.text
    assert "budget" not in response.text.lower(), response.text


def _assert_no_served_row(key: str, marker: str) -> None:
    assert all(row["status"] != "success" for row in landed(key, marker)), landed(key, marker)


def _group_rows(candidate: Gateway, name: str) -> tuple[Mapping[str, JsonValue], ...]:
    response: Final = candidate.request("GET", "/model_group/info", params={"model_group": name})
    assert response.status_code == 200, response.text
    rows: Final = JSON_OBJECT.validate_json(response.content)["data"]
    assert isinstance(rows, list), response.text
    return tuple(object_value(row) for row in rows)


def _cost(value: JsonValue) -> float | None:
    return float(str(value)) if value is not None else None


def _price_of(candidate: Gateway, name: str) -> GroupPrice:
    rows: Final = _group_rows(candidate, name)
    assert len(rows) == 1, rows
    providers: Final = rows[0]["providers"]
    assert isinstance(providers, list), rows
    input_cost: Final = _cost(rows[0]["input_cost_per_token"])
    output_cost: Final = _cost(rows[0]["output_cost_per_token"])
    assert input_cost is not None and output_cost is not None, rows
    return GroupPrice(input_cost, output_cost, tuple(str(provider) for provider in providers))


def _assert_price(price: GroupPrice, input_cost: float, output_cost: float, providers: tuple[str, ...]) -> None:
    assert (price.input_cost_per_token, price.output_cost_per_token) == (input_cost, output_cost), price
    assert price.providers == providers, price


def _listed_input_prices(candidate: Gateway) -> Mapping[str, float | None]:
    rows: Final = candidate.get("/model_group/info")["data"]
    assert isinstance(rows, list), rows
    return {str(object_value(row)["model_group"]): _cost(object_value(row)["input_cost_per_token"]) for row in rows}


def _listed_models(candidate: Gateway) -> frozenset[str]:
    models: Final = candidate.get("/v1/models")["data"]
    assert isinstance(models, list), models
    return frozenset(str(object_value(entry)["id"]) for entry in models)


def _upstream_model_of(observed: tuple[str, ...], marker: str) -> str:
    entries: Final = tuple(entry for entry in observed if marker in entry)
    assert len(entries) == 1, entries
    return str(object_value(JSON_OBJECT.validate_json(entries[0])["body"])["model"])


def _script_failing(rig: ChainRig, statuses: tuple[int, ...]) -> None:
    scripted: Final = httpx.post(
        f"{rig.gateway.upstream_url}/__scripts/{rig.failing_provider_model}",
        json={"statuses": list(statuses)},
        timeout=15,
        trust_env=False,
    )
    assert scripted.status_code == 200, scripted.text


def _clear_failing(rig: ChainRig) -> None:
    cleared: Final = httpx.delete(
        f"{rig.gateway.upstream_url}/__scripts/{rig.failing_provider_model}", timeout=15, trust_env=False
    )
    assert cleared.status_code in (200, 404), cleared.text


def _await_rig(rig: ChainRig) -> None:
    admin: Final = rig.gateway.key
    for name in (
        rig.entry,
        rig.hidden_entry,
        rig.shown_entry,
        rig.null_entry,
        rig.reverse_entry,
        rig.unpriced_entry,
        rig.failing_entry,
    ):
        _settle_both(rig, name, admin, 200)
    paid_id: Final = _deployment_id(rig.paid)
    tail_id: Final = _deployment_id(rig.tail_free)
    for candidate in rig.candidates:
        _await_served_by(candidate, rig.mid_free, paid_id)
        _await_served_by(candidate, rig.failing_free, paid_id)
        _await_served_by(candidate, rig.mid_paid, tail_id)
        _await_served_by(candidate, rig.unpriced, tail_id)


@contextmanager
def _chain_rig() -> Generator[ChainRig]:
    suffix: Final = uuid.uuid4().hex[:12]
    with (
        gateway_from_environment() as gateway,
        httpx.Client(
            base_url=os.environ["INTEGRATION_PEER_URL"], timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as peer_client,
        gateway.scenario() as scenario,
    ):
        failing_provider_model: Final = f"chain-failing-{suffix}"
        rig: Final = ChainRig(
            gateway=gateway,
            peer=Gateway(peer_client, gateway.key, gateway.upstream_url),
            paid=_paid_deployment(scenario),
            mid_free=_free_deployment(scenario, "chain-mid"),
            tail_free=_free_deployment(scenario, "chain-tail"),
            mid_paid=_paid_deployment(scenario),
            unpriced=scenario.model(),
            ghost_mid=f"chain-ghost-mid-{suffix}",
            failing_free=scenario.model(
                model=f"lemonade/{failing_provider_model}", input_cost_per_token=0, output_cost_per_token=0
            ),
            failing_provider_model=failing_provider_model,
            entry=f"chain-entry-{suffix}",
            hidden_entry=f"chain-hidden-entry-{suffix}",
            shown_entry=f"chain-shown-entry-{suffix}",
            null_entry=f"chain-null-entry-{suffix}",
            reverse_entry=f"chain-reverse-entry-{suffix}",
            unpriced_entry=f"chain-unpriced-entry-{suffix}",
            ghost_entry=f"chain-ghost-entry-{suffix}",
            failing_entry=f"chain-failing-entry-{suffix}",
        )
        aliases: Final[Mapping[str, JsonValue]] = {
            rig.entry: rig.mid_free,
            rig.hidden_entry: hidden(rig.mid_free),
            rig.shown_entry: {"model": rig.mid_free, "hidden": False},
            rig.null_entry: {"model": rig.mid_free, "hidden": None},
            rig.mid_free: rig.paid,
            rig.reverse_entry: rig.mid_paid,
            rig.mid_paid: rig.tail_free,
            rig.unpriced_entry: rig.unpriced,
            rig.unpriced: rig.tail_free,
            rig.ghost_entry: rig.ghost_mid,
            rig.ghost_mid: rig.paid,
            rig.failing_entry: rig.failing_free,
            rig.failing_free: rig.paid,
        }
        install_aliases(gateway, aliases)
        scenario.cleanups.callback(remove_aliases, gateway, frozenset(aliases))
        _await_rig(rig)
        yield rig


@pytest.fixture(scope="module")
def rig() -> Iterator[ChainRig]:
    with _chain_rig() as built:
        yield built


def test_exhausted_key_reaches_chain_entry_through_openai_chat(rig: ChainRig) -> None:
    marker: Final = "chat-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        with openai_client(rig.gateway, key) as client:
            completion: Final = client.chat.completions.create(
                model=rig.entry,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            )
        assert completion.choices[0].message.content == CHAT_REPLY, completion
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        row: Final = landed_once(key, marker)
        assert row["request_id"] == completion.id, row
        assert_free_row(row, rig.entry)


async def test_exhausted_key_reaches_chain_entry_through_streamed_openai_chat(rig: ChainRig) -> None:
    marker: Final = "chat-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        async with async_openai_client(rig.gateway, key) as client:
            stream: Final = await client.chat.completions.create(
                model=rig.entry,
                messages=[{"role": "user", "content": marker}],
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=spend_marker(marker),
            )
            chunks: Final = tuple([chunk async for chunk in stream])
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == CHAT_REPLY
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        row: Final = landed_once(key, marker)
        assert row["request_id"] == chunks[0].id, row
        assert_free_row(row, rig.entry)


def test_exhausted_key_reaches_chain_entry_through_openai_responses(rig: ChainRig) -> None:
    marker: Final = "responses-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        with openai_client(rig.gateway, key) as client:
            response: Final = client.responses.create(model=rig.entry, input=marker, extra_headers=spend_marker(marker))
        assert response.output_text == CHAT_REPLY, response
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        assert_free_row(landed_once(key, marker), rig.entry)


async def test_exhausted_key_reaches_chain_entry_through_streamed_openai_responses(rig: ChainRig) -> None:
    marker: Final = "responses-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        async with async_openai_client(rig.gateway, key) as client:
            stream: Final = await client.responses.create(
                model=rig.entry, input=marker, stream=True, extra_headers=spend_marker(marker)
            )
            events: Final = tuple([event async for event in stream])
        assert events[-1].type == "response.completed", events
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == CHAT_REPLY
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        assert_free_row(landed_once(key, marker), rig.entry)


def test_exhausted_key_reaches_chain_entry_through_anthropic_messages(rig: ChainRig) -> None:
    marker: Final = "messages-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        with anthropic_client(rig.gateway, key) as client:
            message: Final = client.messages.create(
                model=rig.entry,
                max_tokens=16,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            )
        assert [block.text for block in message.content if block.type == "text"] == [CHAT_REPLY], message
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        assert_free_row(landed_once(key, marker), rig.entry)


async def test_exhausted_key_reaches_chain_entry_through_streamed_anthropic_messages(rig: ChainRig) -> None:
    marker: Final = "messages-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        async with (
            async_anthropic_client(rig.gateway, key) as client,
            client.messages.stream(
                model=rig.entry,
                max_tokens=16,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            ) as stream,
        ):
            text: Final = "".join([piece async for piece in stream.text_stream])
            final: Final = await stream.get_final_message()
        assert text == CHAT_REPLY, final
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        assert_free_row(landed_once(key, marker), rig.entry)


def _assert_raw_chat_served(rig: ChainRig, candidate: Gateway, entry: str, key: str, prefix: str) -> None:
    marker: Final = f"{prefix}-" + uuid.uuid4().hex
    response: Final = fresh_chat(candidate, entry, key, marker)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == CHAT_REPLY, response.text
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
    row: Final = landed_once(key, marker)
    assert row["request_id"] == response.json()["id"], row
    assert_free_row(row, entry)


def test_exhausted_key_reaches_chain_entry_over_raw_http_on_both_replicas(rig: ChainRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_raw_chat_served(rig, candidate, rig.entry, key, "raw")


@pytest.mark.parametrize("shape", ["string", "hidden_false", "hidden_null", "hidden_true"])
def test_chain_entry_shape_keeps_the_waiver(rig: ChainRig, shape: str) -> None:
    entries: Final = {
        "string": rig.entry,
        "hidden_false": rig.shown_entry,
        "hidden_null": rig.null_entry,
        "hidden_true": rig.hidden_entry,
    }
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_raw_chat_served(rig, candidate, entries[shape], key, f"shape-{shape}")


def _assert_chain_prices(rig: ChainRig, candidate: Gateway) -> None:
    _assert_price(_price_of(candidate, rig.entry), 0.0, 0.0, ("lemonade",))
    _assert_price(
        _price_of(candidate, rig.mid_free), PAID_INPUT_COST_PER_TOKEN, PAID_OUTPUT_COST_PER_TOKEN, ("openai",)
    )
    _assert_price(_price_of(candidate, rig.paid), PAID_INPUT_COST_PER_TOKEN, PAID_OUTPUT_COST_PER_TOKEN, ("openai",))
    listed: Final = _listed_input_prices(candidate)
    assert listed[rig.entry] == 0.0, listed
    assert rig.hidden_entry not in listed, listed
    assert _group_rows(candidate, rig.hidden_entry) == ()
    models: Final = _listed_models(candidate)
    assert rig.entry in models and rig.hidden_entry not in models, models


def test_chain_entry_is_priced_by_the_hop_the_router_takes(rig: ChainRig) -> None:
    for candidate in rig.candidates:
        _assert_chain_prices(rig, candidate)


def test_chain_middle_by_name_is_judged_by_its_own_alias(rig: ChainRig) -> None:
    billed_marker: Final = "middle-billed-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        exhausted: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_refused(rig, candidate, exhausted, rig.mid_free, "middle-refused")
            _assert_refused(rig, candidate, exhausted, rig.paid, "middle-refused")
        budgeted: Final = scenario.key(max_budget=HEADROOM)
        served: Final = fresh_chat(rig.gateway, rig.mid_free, budgeted, billed_marker)
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-model-id"] == _deployment_id(rig.paid), dict(served.headers)
        row: Final = landed_once(budgeted, billed_marker)
        assert math.isclose(float(str(row["spend"])), PAID_REPLY_COST), row
        assert row["model_group"] == rig.mid_free, row


def test_reverse_chain_is_priced_by_the_middle_it_is_served_from(rig: ChainRig) -> None:
    free_marker: Final = "reverse-free-" + uuid.uuid4().hex
    billed_marker: Final = "reverse-billed-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        exhausted: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_refused_on_every_route(rig, candidate, exhausted, rig.reverse_entry)
        served: Final = fresh_chat(rig.gateway, rig.mid_paid, exhausted, free_marker)
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-model-id"] == _deployment_id(rig.tail_free), dict(served.headers)
        assert_free_row(landed_once(exhausted, free_marker), rig.mid_paid)
        budgeted: Final = scenario.key(max_budget=HEADROOM)
        billed: Final = fresh_chat(rig.gateway, rig.reverse_entry, budgeted, billed_marker)
        assert billed.status_code == 200, billed.text
        assert billed.headers["x-litellm-model-id"] == _deployment_id(rig.mid_paid), dict(billed.headers)
        row: Final = landed_once(budgeted, billed_marker)
        assert math.isclose(float(str(row["spend"])), PAID_REPLY_COST), row
        assert row["model_group"] == rig.reverse_entry, row


def test_chain_through_a_cost_map_priced_middle_stays_budgeted(rig: ChainRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_refused_on_every_route(rig, candidate, key, rig.unpriced_entry)


def _assert_ghost_chain(rig: ChainRig, candidate: Gateway, key: str) -> None:
    assert _group_rows(candidate, rig.ghost_entry) == ()
    assert rig.ghost_entry not in _listed_input_prices(candidate)
    refused: Final = fresh_chat(candidate, rig.ghost_entry, key, "ghost-exhausted-" + uuid.uuid4().hex)
    assert refused.status_code == BUDGET_EXCEEDED, refused.text
    unroutable: Final = fresh_chat(candidate, rig.ghost_entry, rig.gateway.key, "ghost-admin-" + uuid.uuid4().hex)
    assert unroutable.status_code == 400, unroutable.text
    assert "no healthy deployments" in unroutable.text, unroutable.text
    for path in ("/health/liveliness", "/model/info", "/v1/models"):
        assert candidate.request("GET", path).status_code == 200, path


def test_ghost_chain_reports_no_group_and_stays_unroutable(rig: ChainRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_ghost_chain(rig, candidate, key)
        control: Final = fresh_chat(rig.gateway, rig.entry, key, "ghost-control-" + uuid.uuid4().hex)
        assert control.status_code == 200, control.text


def test_provider_failure_behind_a_chain_entry_reaches_the_caller(rig: ChainRig) -> None:
    failed_marker: Final = "provider-failure-" + uuid.uuid4().hex
    recovered_marker: Final = "provider-recovered-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        scenario.cleanups.callback(_clear_failing, rig)
        _script_failing(rig, (500,))
        _assert_provider_failure(fresh_chat(rig.gateway, rig.failing_entry, key, failed_marker))
        _clear_failing(rig)
        recovered: Final = fresh_chat(rig.gateway, rig.failing_entry, key, recovered_marker)
        assert recovered.status_code == 200, recovered.text
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert upstream_hits(observed, failed_marker) == 1
        assert upstream_hits(observed, recovered_marker) == 1
        assert_free_row(landed_once(key, recovered_marker), rig.failing_entry)
        _assert_no_served_row(key, failed_marker)


def test_chain_through_a_free_wildcard_route_is_priced_by_that_route(rig: ChainRig) -> None:
    prefix: Final = "wc" + uuid.uuid4().hex[:8]
    middle: Final = f"{prefix}/gpt-4o-mini"
    other: Final = f"{prefix}/gpt-4o"
    entry: Final = "wildcard-entry-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        route: Final = _wildcard_route(rig, scenario, prefix, 0.0, 0.0)
        install_aliases(rig.gateway, {entry: middle, middle: rig.paid})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({entry, middle}))
        paid_id: Final = _deployment_id(rig.paid)
        for candidate in rig.candidates:
            _await_served_by(candidate, entry, route)
            _await_served_by(candidate, middle, paid_id)
        key: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_served_by_route(candidate, other, key, route, "wildcard-other")
            _assert_price(_price_of(candidate, other), 0.0, 0.0, ("openai",))
            _assert_price(
                _price_of(candidate, middle), PAID_INPUT_COST_PER_TOKEN, PAID_OUTPUT_COST_PER_TOKEN, ("openai",)
            )
            _assert_served_by_route(candidate, entry, key, route, "wildcard-served")
            _assert_refused(rig, candidate, key, middle, "wildcard-middle")


def _assert_served_by_route(candidate: Gateway, model: str, key: str, route: str, prefix: str) -> None:
    marker: Final = f"{prefix}-" + uuid.uuid4().hex
    served: Final = fresh_chat(candidate, model, key, marker)
    assert served.status_code == 200, served.text
    assert served.headers["x-litellm-model-id"] == route, dict(served.headers)
    assert_free_row(landed_once(key, marker), model)


def test_chain_through_a_priced_wildcard_route_stays_budgeted(rig: ChainRig) -> None:
    prefix: Final = "pwc" + uuid.uuid4().hex[:8]
    middle: Final = f"{prefix}/gpt-4o-mini"
    entry: Final = "priced-wildcard-entry-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        route: Final = _wildcard_route(rig, scenario, prefix, PAID_INPUT_COST_PER_TOKEN, PAID_OUTPUT_COST_PER_TOKEN)
        install_aliases(rig.gateway, {entry: middle, middle: rig.tail_free})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({entry, middle}))
        tail_id: Final = _deployment_id(rig.tail_free)
        for candidate in rig.candidates:
            _await_served_by(candidate, entry, route)
            _await_served_by(candidate, middle, tail_id)
        key: Final = _exhausted_key(rig, scenario)
        for candidate in rig.candidates:
            _assert_refused_on_every_route(rig, candidate, key, entry)


def _assert_hostile_group_queries_answered(rig: ChainRig, candidate: Gateway) -> None:
    for value in ("7", "", "a,b", "x" * 5120):
        assert isinstance(_group_rows(candidate, value), tuple), value
    repeated: Final = httpx.get(
        f"{base_url(candidate)}/model_group/info",
        params=[("model_group", "7"), ("model_group", rig.entry)],
        headers={"Authorization": f"Bearer {candidate.key}"},
        timeout=15,
        trust_env=False,
    )
    assert repeated.status_code == 200, repeated.text
    assert isinstance(JSON_OBJECT.validate_json(repeated.content)["data"], list), repeated.text
    assert _group_rows(candidate, rig.entry) == _group_rows(candidate, rig.entry)
    unauthenticated: Final = httpx.get(
        f"{base_url(candidate)}/model_group/info", params={"model_group": rig.entry}, timeout=15, trust_env=False
    )
    assert unauthenticated.status_code == 401, unauthenticated.text
    assert candidate.request("GET", "/health/liveliness").status_code == 200


def test_hostile_model_group_queries_leave_the_proxy_healthy(rig: ChainRig) -> None:
    for candidate in rig.candidates:
        _assert_hostile_group_queries_answered(rig, candidate)


@pytest.mark.parametrize(
    ("shape", "status"),
    [
        ("int", BUDGET_EXCEEDED),
        ("list", 400),
        ("free_list", 400),
        ("empty", BUDGET_EXCEEDED),
        ("oversized", BUDGET_EXCEEDED),
    ],
)
def test_malformed_model_value_never_takes_the_chain_waiver(rig: ChainRig, shape: str, status: int) -> None:
    marker: Final = f"malformed-{shape}-" + uuid.uuid4().hex
    models: Final[Mapping[str, JsonValue]] = {
        "int": 5,
        "list": [rig.entry],
        "free_list": [rig.tail_free],
        "empty": "",
        "oversized": rig.entry + "x" * 5120,
    }
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": models[shape], "messages": [{"role": "user", "content": marker}]},
            key=key,
        )
        assert response.status_code == status, response.text
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0
        control: Final = fresh_chat(rig.gateway, rig.tail_free, key, "malformed-control-" + uuid.uuid4().hex)
        assert control.status_code == 200, control.text


def test_unauthenticated_request_to_chain_entry_is_rejected(rig: ChainRig) -> None:
    marker: Final = "unauthenticated-" + uuid.uuid4().hex
    response: Final = httpx.post(
        f"{base_url(rig.gateway)}/v1/chat/completions",
        json={"model": rig.entry, "messages": [{"role": "user", "content": marker}]},
        timeout=60,
        trust_env=False,
    )
    assert response.status_code == 401, response.text
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def _duplicate_model_post(rig: ChainRig, key: str, first: str, last: str, marker: str) -> httpx.Response:
    messages: Final = json.dumps([{"role": "user", "content": marker}])
    return httpx.post(
        f"{base_url(rig.gateway)}/v1/chat/completions",
        content=f'{{"model": {json.dumps(first)}, "model": {json.dumps(last)}, "messages": {messages}}}'.encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", **spend_marker(marker)},
        timeout=60,
        trust_env=False,
    )


def test_duplicate_model_field_is_judged_by_its_last_value(rig: ChainRig) -> None:
    free_marker: Final = "duplicate-free-" + uuid.uuid4().hex
    paid_marker: Final = "duplicate-paid-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        served: Final = _duplicate_model_post(rig, key, rig.paid, rig.entry, free_marker)
        assert served.status_code == 200, served.text
        refused: Final = _duplicate_model_post(rig, key, rig.entry, rig.paid, paid_marker)
        assert refused.status_code == BUDGET_EXCEEDED, refused.text
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert upstream_hits(observed, free_marker) == 1
        assert upstream_hits(observed, paid_marker) == 0
        assert_free_row(landed_once(key, free_marker), rig.entry)


def test_key_restricted_to_the_chain_entry_keeps_the_waiver(rig: ChainRig) -> None:
    marker: Final = "restricted-served-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        outsider: Final = scenario.key(models=[rig.tail_free, rig.paid], max_budget=BUDGET)
        denied: Final = fresh_chat(rig.gateway, rig.entry, outsider, "restricted-denied-" + uuid.uuid4().hex)
        assert _refusal(denied) == (403, "key_model_access_denied"), denied.text
        insider: Final = scenario.key(models=[rig.entry, rig.paid], max_budget=BUDGET)
        first: Final = fresh_chat(rig.gateway, rig.paid, insider, "restricted-exhaust-" + uuid.uuid4().hex)
        assert first.status_code == 200, first.text
        _settle_both(rig, rig.paid, insider, BUDGET_EXCEEDED)
        served: Final = fresh_chat(rig.gateway, rig.entry, insider, marker)
        assert served.status_code == 200, served.text
        assert_free_row(landed_once(insider, marker), rig.entry)


def test_repointing_the_chain_middle_follows_the_served_deployment(rig: ChainRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        scenario.cleanups.callback(install_aliases, rig.gateway, {rig.entry: rig.mid_free, rig.mid_free: rig.paid})
        install_aliases(rig.gateway, {rig.mid_free: rig.tail_free})
        _settle_both(rig, rig.mid_free, key, 200)
        _settle_both(rig, rig.entry, key, 200)
        install_aliases(rig.gateway, {rig.mid_free: rig.paid})
        _settle_both(rig, rig.mid_free, key, BUDGET_EXCEEDED)
        _settle_both(rig, rig.entry, key, 200)
        install_aliases(rig.gateway, {rig.entry: rig.paid})
        _settle_both(rig, rig.entry, key, BUDGET_EXCEEDED)
        install_aliases(rig.gateway, {rig.entry: rig.mid_free})
        _settle_both(rig, rig.entry, key, 200)


def test_exhausted_key_is_served_a_cached_reply_through_chain_entry(rig: ChainRig) -> None:
    marker: Final = "cache-twin-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        first: Final = fresh_chat(rig.gateway, rig.entry, key, marker)
        assert first.status_code == 200, first.text
        second: Final = fresh_chat(rig.gateway, rig.entry, key, marker)
        assert second.status_code == 200, second.text
        first_id: Final = str(JSON_OBJECT.validate_json(first.content)["id"])
        assert str(JSON_OBJECT.validate_json(second.content)["id"]) == first_id, second.text
        assert "x-litellm-cache-key" in second.headers, dict(second.headers)
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        rows: Final = eventually(lambda: landed(key, marker), lambda found: len(found) >= 2, seconds=70)
        assert len(rows) == 2, rows
        assert {str(row["request_id"]).split("_cache_hit")[0] for row in rows} == {first_id}, rows
        assert sum(1 for row in rows if "_cache_hit" in str(row["request_id"])) == 1, rows
        for row in rows:
            assert_free_row(row, rig.entry)


@dataclass(frozen=True, slots=True)
class BurstCall:
    candidate: Gateway
    kind: str
    marker: str


def _burst_calls(candidate: Gateway, count: int, prefix: str) -> tuple[BurstCall, ...]:
    return tuple(
        BurstCall(candidate, BURST_KINDS[index % len(BURST_KINDS)], f"{prefix}-" + uuid.uuid4().hex)
        for index in range(count)
    )


def _send_kind(call: BurstCall, model: str, key: str) -> httpx.Response:
    stream: Final[Mapping[str, JsonValue]] = {"stream": True}
    match call.kind:
        case "chat":
            return fresh_chat(call.candidate, model, key, call.marker)
        case "chat-stream":
            return fresh_chat(call.candidate, model, key, call.marker, stream)
        case "responses":
            return fresh_response(call.candidate, model, key, call.marker)
        case "responses-stream":
            return fresh_response(call.candidate, model, key, call.marker, stream)
        case "messages":
            return fresh_message(call.candidate, model, key, call.marker)
        case "messages-stream":
            return fresh_message(call.candidate, model, key, call.marker, stream)
        case _:
            raise AssertionError(call.kind)


def test_chain_entry_burst_lands_every_served_id_once(rig: ChainRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        calls: Final = _burst_calls(rig.gateway, GATEWAY_BURST, "burst") + _burst_calls(rig.peer, PEER_BURST, "burst")
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            responses: Final = dict(
                zip(calls, pool.map(partial(_send_kind, model=rig.entry, key=key), calls), strict=True)
            )
        failed: Final = {
            call.marker: response.status_code for call, response in responses.items() if response.status_code != 200
        }
        assert failed == {}, failed
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert all(upstream_hits(observed, call.marker) == 1 for call in calls), observed
        for row in landed_all_once(key, frozenset(call.marker for call in calls)):
            assert_free_row(row, rig.entry)


def _base_config() -> Mapping[str, JsonValue]:
    return JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))


def _routing_group_config(rig: ChainRig, directory: Path, group: str, member: str) -> Path:
    base: Final = _base_config()
    model_list: Final = base["model_list"]
    assert isinstance(model_list, list), model_list
    deployment: Final[JsonValue] = {
        "model_name": member,
        "litellm_params": {
            "model": f"lemonade/{member}",
            "api_key": "integration-provider-key",
            "api_base": f"{rig.gateway.upstream_url}/v1",
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    }
    routing_groups: Final[JsonValue] = [{"group_name": group, "models": [member], "routing_strategy": "simple-shuffle"}]
    path: Final = directory / "routing-group-chain.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "model_list": [*model_list, deployment],
                "router_settings": {**object_value(base["router_settings"]), "routing_groups": routing_groups},
            }
        )
    )
    return path


@pytest.mark.timeout(480)
def test_chain_through_a_routing_group_keeps_the_waiver(rig: ChainRig, tmp_path: Path) -> None:
    suffix: Final = uuid.uuid4().hex[:8]
    group: Final = f"rg-chain-{suffix}"
    member: Final = f"rg-member-{suffix}"
    entry: Final = f"rg-entry-{suffix}"
    marker: Final = "routing-group-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        install_aliases(rig.gateway, {entry: group})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({entry}))
        key: Final = _exhausted_key(rig, scenario)
        with owned_proxy(
            rig.gateway, tmp_path, {}, config=_routing_group_config(rig, tmp_path, group, member)
        ) as candidate:
            settle_candidate(candidate, entry, key, 200, seconds=120)
            _assert_price(_price_of(candidate, entry), 0.0, 0.0, ("lemonade",))
            served: Final = fresh_chat(candidate, entry, key, marker)
            assert served.status_code == 200, served.text
            assert _upstream_model_of(upstream_requests(rig.gateway.upstream_url), marker) == member
            assert_free_row(landed_once(key, marker), entry)


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    return WORKER_PIDS.validate_python(STARTED_WORKER.findall(text)), text.count("Application startup complete.")


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _marker_of(request: Request) -> str:
    messages: Final = JSON_OBJECT.validate_json(request.body)["messages"]
    assert isinstance(messages, list) and messages, request.body
    return str(object_value(messages[0])["content"])


def _response_or_none(send: Callable[[], httpx.Response]) -> httpx.Response | None:
    try:
        return send()
    except httpx.TransportError:
        return None


@pytest.mark.timeout(600)
def test_killing_one_worker_mid_burst_keeps_the_chain_entry_served(rig: ChainRig, tmp_path: Path) -> None:
    suffix: Final = uuid.uuid4().hex[:8]
    entry: Final = f"chaos-entry-{suffix}"
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        if request.method != "POST" or not request.target.endswith("/chat/completions"):
            return Reply(status=404)
        marker: Final = _marker_of(request)
        if marker.startswith("hold-"):
            held_markers.put(marker)
            assert release.wait(timeout=120), "The burst was never released"
        model: Final = str(JSON_OBJECT.validate_json(request.body)["model"])
        return chat_reply(f"chatcmpl-{marker}", model, CHAT_REPLY, stream=False)

    with wire_server(respond) as wire, rig.gateway.scenario() as scenario:
        chaos_free: Final = scenario.model(
            model=f"lemonade/chaos-free-{suffix}",
            api_base=f"{wire.url}/v1",
            input_cost_per_token=0,
            output_cost_per_token=0,
        )
        install_aliases(rig.gateway, {entry: chaos_free, chaos_free: rig.paid})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({entry, chaos_free}))
        key: Final = _exhausted_key(rig, scenario)
        with owned_proxy_process(rig.gateway, tmp_path, {}, workers=2) as owned:
            candidate: Final = owned.gateway
            workers, _ = eventually(
                lambda: _worker_startups(owned.log),
                lambda found: len(found[0]) == 2 and found[1] == 2,
                seconds=120,
            )
            settle_candidate(candidate, entry, key, 200, seconds=120)
            held_calls: Final = tuple("hold-" + uuid.uuid4().hex for _ in range(GATEWAY_BURST))
            with ThreadPoolExecutor(max_workers=GATEWAY_BURST) as pool:
                pending: Final = {
                    marker: pool.submit(_response_or_none, partial(fresh_chat, candidate, entry, key, marker))
                    for marker in held_calls
                }
                eventually(held_markers.qsize, lambda size: size == GATEWAY_BURST, seconds=60)
                held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
                assert sum(held_by.values()) == GATEWAY_BURST, held_by
                victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                victim: Final = psutil.Process(victim_pid)
                victim.suspend()
                victim.send_signal(signal.SIGKILL)
                release.set()
                responses: Final = {marker: future.result() for marker, future in pending.items()}
            served: Final = frozenset(
                marker for marker, response in responses.items() if response is not None and response.status_code == 200
            )
            lost: Final = frozenset(responses) - served
            assert len(served) == held_by[survivor_pid], (held_by, len(served), len(lost))
            follow_up_marker: Final = "after-kill-" + uuid.uuid4().hex
            follow_up: Final = fresh_chat(candidate, entry, key, follow_up_marker)
            assert follow_up.status_code == 200, follow_up.text
            eventually(
                lambda: _worker_startups(owned.log),
                lambda found: len(found[0]) == 3 and found[1] == 3,
                seconds=180,
            )
            for row in landed_all_once(key, served | {follow_up_marker}):
                assert_free_row(row, entry)
            for marker in lost:
                _assert_no_served_row(key, marker)


def test_chain_entry_burst_with_provider_outage_lands_every_served_id_once(rig: ChainRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _exhausted_key(rig, scenario)
        scenario.cleanups.callback(_clear_failing, rig)
        _script_failing(rig, (500,) * BURST_FAILURES + (200,) * (GATEWAY_BURST - BURST_FAILURES))
        calls: Final = _burst_calls(rig.gateway, GATEWAY_BURST, "outage")
        with ThreadPoolExecutor(max_workers=len(calls) + LIVELINESS_PROBES) as pool:
            probes: Final = tuple(
                pool.submit(httpx.get, f"{base_url(rig.gateway)}/health/liveliness", timeout=60, trust_env=False)
                for _ in range(LIVELINESS_PROBES)
            )
            responses: Final = dict(
                zip(calls, pool.map(partial(_send_kind, model=rig.failing_entry, key=key), calls), strict=True)
            )
        failed: Final = {call: response for call, response in responses.items() if response.status_code != 200}
        assert len(failed) == BURST_FAILURES, {call.kind: response.status_code for call, response in failed.items()}
        for response in failed.values():
            _assert_provider_failure(response)
        for probe in probes:
            assert probe.result().status_code == 200, probe.result().text
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert all(upstream_hits(observed, call.marker) == 1 for call in calls), observed
        served_markers: Final = frozenset(call.marker for call in calls if call not in failed)
        assert len(served_markers) == GATEWAY_BURST - BURST_FAILURES
        for row in landed_all_once(key, served_markers):
            assert_free_row(row, rig.failing_entry)
        for call in failed:
            _assert_no_served_row(key, call.marker)
