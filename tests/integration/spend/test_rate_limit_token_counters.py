import asyncio
import json
import os
import uuid
from collections.abc import Callable, Iterator, Mapping, Set
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, held
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue
from redis import Redis

USAGE: Final = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
WIRE_TOTAL_TOKENS: Final = 8
Fields = Callable[[str], Mapping[str, JsonValue]]


def completion(request: Request) -> Reply:
    assert request.target == "/v1/chat/completions"
    body: Final = json.loads(request.body)
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": USAGE,
            }
        ).encode()
    )


def outage(request: Request) -> Reply:
    assert request.target == "/v1/chat/completions"
    return Reply(status=500, body=json.dumps({"error": {"message": "synthetic outage"}}).encode())


def openai_endpoints(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    stream: Final = body.get("stream") is True
    if request.target == "/v1/chat/completions":
        return chat_reply("chatcmpl-" + uuid.uuid4().hex, body["model"], "ok", stream=stream)
    assert request.target == "/v1/responses", request.target
    return responses_reply("resp_" + uuid.uuid4().hex, body["model"], "ok", stream=stream)


def counter(scope: str, value: str) -> str:
    return f"{{{scope}:{value}}}:tokens"


def read_counters(cache: Redis, keys: Mapping[str, str]) -> Mapping[str, float | None]:
    values: Final = cache.mget(list(keys.values()))
    return {name: None if raw is None else float(raw) for name, raw in zip(keys, values, strict=True)}


def no_fields(_model: str) -> Mapping[str, JsonValue]:
    return {}


@dataclass(frozen=True, slots=True)
class Caller:
    key: str
    user: str
    team: str
    scopes: Mapping[str, str]
    body: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Cell:
    name: str
    charged: str
    also_charged: frozenset[str] = frozenset()
    key: Fields = no_fields
    user: Fields = no_fields
    team: Fields = no_fields
    organization: Fields | None = None
    project: Fields | None = None
    customer: Fields | None = None
    tag: Fields | None = None


def caller(scenario: Scenario, cell: Cell, model: str) -> Caller:
    gateway: Final = scenario.gateway
    organization: Final = None if cell.organization is None else scenario.organization(**cell.organization(model))
    team: Final = scenario.team(
        **cell.team(model), **({} if organization is None else {"organization_id": organization})
    )
    user: Final = scenario.user(user_role="internal_user", **cell.user(model))
    gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
    project: Final = None if cell.project is None else scenario.project(team, **cell.project(model))
    key: Final = scenario.key(
        user_id=user, team_id=team, **({} if project is None else {"project_id": project}), **cell.key(model)
    )
    hashed: Final = sha256(key.encode()).hexdigest()
    scopes: Final = {
        "api_key": counter("api_key", hashed),
        "user": counter("user", user),
        "team": counter("team", team),
        "team_member": counter("team_member", f"{team}:{user}"),
        "model_per_key": counter("model_per_key", f"{hashed}:{model}"),
        "model_per_team": counter("model_per_team", f"{team}:{model}"),
    }
    body: Final[dict[str, JsonValue]] = {}
    if organization is not None:
        scopes["organization"] = counter("organization", organization)
        scopes["model_per_organization"] = counter("model_per_organization", f"{organization}:{model}")
    if project is not None:
        scopes["model_per_project"] = counter("model_per_project", f"{project}:{model}")
    if cell.customer is not None:
        customer: Final = f"integration-customer-{uuid.uuid4().hex}"
        gateway.post("/customer/new", {"user_id": customer, **cell.customer(model)})
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [customer]})
        scopes["end_user"] = counter("end_user", customer)
        body["user"] = customer
    if cell.tag is not None:
        tag: Final = f"integration-tag-{uuid.uuid4().hex}"
        gateway.post("/tag/new", {"name": tag, **cell.tag(model)})
        scenario.cleanups.callback(gateway.request, "POST", "/tag/delete", {"name": tag})
        scopes["tag"] = counter("tag", tag)
        body["metadata"] = {"tags": [tag]}
    return Caller(key, user, team, scopes, body)


def key_rpm_only(_model: str) -> Mapping[str, JsonValue]:
    return {"rpm_limit": 1000}


CELLS: Final = (
    Cell("organization_rpm_only", charged="organization", organization=lambda _model: {"rpm_limit": 1000}),
    Cell(
        "organization_model_rpm_only",
        charged="model_per_organization",
        organization=lambda model: {"model_rpm_limit": {model: 1000}},
    ),
    Cell("key_rpm_only", charged="api_key", key=key_rpm_only),
    Cell("user_rpm_only", charged="user", user=lambda _model: {"rpm_limit": 1000}),
    Cell("team_rpm_only", charged="team", team=lambda _model: {"rpm_limit": 1000}),
    Cell("team_member_rpm_only", charged="team_member", team=lambda _model: {"team_member_rpm_limit": 1000}),
    Cell("key_model_rpm_only", charged="model_per_key", key=lambda model: {"model_rpm_limit": {model: 1000}}),
    Cell(
        "team_model_rpm_only",
        charged="model_per_team",
        also_charged=frozenset({"model_per_key"}),
        team=lambda model: {"model_rpm_limit": {model: 1000}},
    ),
    Cell(
        "project_model_rpm_only", charged="model_per_project", project=lambda model: {"model_rpm_limit": {model: 1000}}
    ),
    Cell("end_user_rpm_only", charged="end_user", customer=lambda _model: {"rpm_limit": 1000}),
    Cell("tag_tpm", charged="tag", tag=lambda _model: {"tpm_limit": 100000}),
    Cell("key_tpm_reserved", charged="api_key", key=lambda _model: {"tpm_limit": 100000}),
    Cell(
        "key_dynamic_tpm",
        charged="api_key",
        key=lambda _model: {"tpm_limit": 100000, "tpm_limit_type": "dynamic"},
    ),
)


@contextmanager
def redis_counters() -> Iterator[Redis]:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        yield cache


def assert_only_charged(cache: Redis, scopes: Mapping[str, str], charged: Set[str], tokens: float) -> None:
    counters: Final = eventually(
        lambda: read_counters(cache, scopes),
        lambda values: all(values[name] == tokens for name in charged),
        seconds=30,
    )
    assert {name: value for name, value in counters.items() if name not in charged} == {
        name: None for name in scopes if name not in charged
    }


def chat(gateway: Gateway, identity: Caller, model: str, prompt: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}], **identity.body, **extra},
        key=identity.key,
    )


@pytest.mark.parametrize("cell", CELLS, ids=[cell.name for cell in CELLS])
def test_success_charges_token_counters_only_for_rate_limited_scopes(gateway: Gateway, cell: Cell) -> None:
    prompt: Final = f"token counters {cell.name} {uuid.uuid4().hex}"
    with wire_server(completion) as upstream, gateway.scenario() as scenario, redis_counters() as cache:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        identity: Final = caller(scenario, cell, model)
        response: Final = chat(gateway, identity, model, prompt, max_tokens=500)
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["total_tokens"] == USAGE["total_tokens"]
        assert [json.loads(seen.body)["messages"] for seen in upstream.drain()] == [
            [{"role": "user", "content": prompt}]
        ]

        assert_only_charged(cache, identity.scopes, {cell.charged, *cell.also_charged}, USAGE["total_tokens"])


def gateway_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def openai_chat_sync(gateway: Gateway, key: str, model: str, prompt: str) -> int | None:
    with (
        httpx.Client(timeout=30, trust_env=False) as transport,
        openai.OpenAI(
            api_key=key, base_url=gateway_url(gateway) + "/v1", max_retries=0, http_client=transport
        ) as client,
    ):
        response: Final = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return None if response.usage is None else response.usage.total_tokens


async def openai_chat_stream_async(gateway: Gateway, key: str, model: str, prompt: str) -> int | None:
    async with (
        httpx.AsyncClient(timeout=30, trust_env=False) as transport,
        openai.AsyncOpenAI(
            api_key=key, base_url=gateway_url(gateway) + "/v1", max_retries=0, http_client=transport
        ) as client,
    ):
        stream: Final = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )
        usage: Final = [chunk.usage.total_tokens async for chunk in stream if chunk.usage is not None]
    return usage[-1] if usage else None


def anthropic_messages_sync(gateway: Gateway, key: str, model: str, prompt: str) -> int | None:
    with (
        httpx.Client(timeout=30, trust_env=False) as transport,
        anthropic.Anthropic(api_key=key, base_url=gateway_url(gateway), max_retries=0, http_client=transport) as client,
    ):
        message: Final = client.messages.create(
            model=model, max_tokens=50, messages=[{"role": "user", "content": prompt}]
        )
    return message.usage.input_tokens + message.usage.output_tokens


async def anthropic_messages_stream_async(gateway: Gateway, key: str, model: str, prompt: str) -> int | None:
    async with (
        httpx.AsyncClient(timeout=30, trust_env=False) as transport,
        anthropic.AsyncAnthropic(
            api_key=key, base_url=gateway_url(gateway), max_retries=0, http_client=transport
        ) as client,
        client.messages.stream(model=model, max_tokens=50, messages=[{"role": "user", "content": prompt}]) as stream,
    ):
        message: Final = await stream.get_final_message()
    return message.usage.input_tokens + message.usage.output_tokens


def openai_responses_sync(gateway: Gateway, key: str, model: str, prompt: str) -> int | None:
    with (
        httpx.Client(timeout=30, trust_env=False) as transport,
        openai.OpenAI(
            api_key=key, base_url=gateway_url(gateway) + "/v1", max_retries=0, http_client=transport
        ) as client,
    ):
        response: Final = client.responses.create(model=model, input=prompt)
    return None if response.usage is None else response.usage.total_tokens


async def openai_responses_stream_async(gateway: Gateway, key: str, model: str, prompt: str) -> int | None:
    async with (
        httpx.AsyncClient(timeout=30, trust_env=False) as transport,
        openai.AsyncOpenAI(
            api_key=key, base_url=gateway_url(gateway) + "/v1", max_retries=0, http_client=transport
        ) as client,
    ):
        stream: Final = await client.responses.create(model=model, input=prompt, stream=True)
        usage: Final = [
            event.response.usage.total_tokens
            async for event in stream
            if event.type == "response.completed" and event.response.usage is not None
        ]
    return usage[-1] if usage else None


ENDPOINTS: Final[Mapping[str, Callable[[Gateway, str, str, str], int | None]]] = {
    "chat_sync": openai_chat_sync,
    "chat_stream_async": lambda *args: asyncio.run(openai_chat_stream_async(*args)),
    "messages_sync": anthropic_messages_sync,
    "messages_stream_async": lambda *args: asyncio.run(anthropic_messages_stream_async(*args)),
    "responses_sync": openai_responses_sync,
    "responses_stream_async": lambda *args: asyncio.run(openai_responses_stream_async(*args)),
}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_every_endpoint_charges_only_the_rate_limited_scope(gateway: Gateway, endpoint: str) -> None:
    prompt: Final = f"token counters {endpoint} {uuid.uuid4().hex}"
    with wire_server(openai_endpoints) as upstream, gateway.scenario() as scenario, redis_counters() as cache:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        identity: Final = caller(scenario, Cell(endpoint, charged="api_key", key=key_rpm_only), model)

        assert ENDPOINTS[endpoint](gateway, identity.key, model, prompt) == WIRE_TOTAL_TOKENS
        assert len(upstream.drain()) == 1

        assert_only_charged(cache, identity.scopes, {"api_key"}, WIRE_TOTAL_TOKENS)


def test_fallback_hop_charges_the_rate_limited_model_it_served(gateway: Gateway) -> None:
    prompt: Final = f"token counters fallback {uuid.uuid4().hex}"
    with (
        wire_server(outage) as primary_upstream,
        wire_server(completion) as fallback_upstream,
        gateway.scenario() as scenario,
        redis_counters() as cache,
    ):
        primary: Final = scenario.model(api_base=primary_upstream.url + "/v1")
        fallback: Final = scenario.model(api_base=fallback_upstream.url + "/v1")
        cell: Final = Cell(
            "fallback",
            charged="model_per_key",
            key=lambda _model: {"model_tpm_limit": {fallback: 100000}},
        )
        identity: Final = caller(scenario, cell, fallback)
        response: Final = chat(gateway, identity, primary, prompt, fallbacks=[fallback])
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-model-group"] == fallback
        assert len(primary_upstream.drain()) == 1
        assert len(fallback_upstream.drain()) == 1

        hashed: Final = sha256(identity.key.encode()).hexdigest()
        scopes: Final = {
            name: identity.scopes[name] for name in ("api_key", "user", "team", "team_member", "model_per_key")
        }
        assert_only_charged(
            cache,
            {**scopes, "primary_model_per_key": counter("model_per_key", f"{hashed}:{primary}")},
            {"model_per_key"},
            USAGE["total_tokens"],
        )


def same_key(_scenario: Scenario, identity: Caller) -> Caller:
    return identity


def another_key_of_the_same_user(scenario: Scenario, identity: Caller) -> Caller:
    return replace(identity, key=scenario.key(user_id=identity.user, team_id=identity.team))


@pytest.mark.parametrize(
    "cell, next_caller",
    [
        (Cell("key_tpm", charged="api_key", key=lambda _model: {"tpm_limit": 21}), same_key),
        (
            Cell("user_tpm_across_keys", charged="user", user=lambda _model: {"tpm_limit": 21}),
            another_key_of_the_same_user,
        ),
    ],
    ids=["key_tpm", "user_tpm_across_keys"],
)
def test_rate_limited_scope_rejects_once_charged_tokens_reach_its_limit(
    gateway: Gateway, cell: Cell, next_caller: Callable[[Scenario, Caller], Caller]
) -> None:
    with wire_server(completion) as upstream, gateway.scenario() as scenario, redis_counters() as cache:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        identity: Final = caller(scenario, cell, model)
        first: Final = chat(gateway, identity, model, "hi", max_tokens=1)
        assert first.status_code == 200, first.text
        eventually(
            lambda: read_counters(cache, identity.scopes)[cell.charged],
            lambda tokens: tokens == USAGE["total_tokens"],
            seconds=30,
        )

        rejected: Final = chat(gateway, next_caller(scenario, identity), model, "hi again", max_tokens=1)
        assert rejected.status_code == 429, rejected.text
        assert len(upstream.drain()) == 1
        assert_only_charged(cache, identity.scopes, {cell.charged}, USAGE["total_tokens"])


def test_concurrent_burst_charges_every_request_to_the_rate_limited_scope_only(gateway: Gateway) -> None:
    requests: Final = 20
    with wire_server(completion) as upstream, gateway.scenario() as scenario, redis_counters() as cache:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        identity: Final = caller(scenario, Cell("burst", charged="api_key", key=key_rpm_only), model)
        with ThreadPoolExecutor(max_workers=requests) as pool:
            statuses: Final = list(
                pool.map(
                    lambda index: chat(gateway, identity, model, f"token counters burst {index} {uuid.uuid4().hex}"),
                    range(requests),
                )
            )
        assert [response.status_code for response in statuses] == [200] * requests
        assert len(upstream.drain()) == requests

        assert_only_charged(cache, identity.scopes, {"api_key"}, requests * USAGE["total_tokens"])


def test_upstream_outage_charges_no_token_counter(gateway: Gateway) -> None:
    with wire_server(outage) as upstream, gateway.scenario() as scenario, redis_counters() as cache:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        identity: Final = caller(scenario, Cell("outage", charged="api_key", key=key_rpm_only), model)
        response: Final = chat(gateway, identity, model, f"token counters outage {uuid.uuid4().hex}")
        assert response.status_code == 500, response.text
        assert len(upstream.drain()) == 1

        held(
            lambda: read_counters(cache, identity.scopes),
            lambda values: set(values.values()) == {None},
            holding=3,
        )
