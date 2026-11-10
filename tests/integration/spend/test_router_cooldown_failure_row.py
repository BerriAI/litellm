import asyncio
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import GATEWAY_LIMITS, Gateway, eventually, gateway_from_environment, object_value
from integration._support.process import owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue
from redis import Redis

pytestmark: Final = pytest.mark.timeout(300)

COOLDOWN_SECONDS: Final = 120
NO_DEPLOYMENTS: Final = "No deployments available"
ROW_SECONDS: Final = 70
CHAT: Final = "/v1/chat/completions"
MESSAGES: Final = "/v1/messages"
RESPONSES: Final = "/v1/responses"
FAILURE_ROW: Final = (
    "SELECT status, model_group, model_id, api_base, metadata::jsonb -> 'error_information' AS error "
    'FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
)
ENDPOINT_CELLS: Final = tuple((path, stream) for path in (CHAT, MESSAGES, RESPONSES) for stream in (False, True))
ENDPOINT_GROUPS: Final = tuple(
    f"{path.rsplit('/', 1)[-1]}-{'stream' if stream else 'json'}" for path, stream in ENDPOINT_CELLS
)
SDK_CELLS: Final = ("openai-sync", "openai-async-stream", "anthropic-sync")
TAGGED: Final = "tagged"
REGIONAL: Final = "regional"
GROUPS: Final = (*ENDPOINT_GROUPS, *SDK_CELLS, "neighbour", "scoped", "burst", TAGGED, REGIONAL)
LANES: Final = ("a", "b")
LANE_PARAMS: Final[Mapping[tuple[str, str], Mapping[str, JsonValue]]] = {
    (TAGGED, "a"): {"tags": ["tenant-a"]},
    (TAGGED, "b"): {"tags": ["default"]},
    (REGIONAL, "a"): {"region_name": "eu"},
    (REGIONAL, "b"): {"region_name": "us"},
}
TENANT_A: Final[Mapping[str, JsonValue]] = {"metadata": {"tags": ["tenant-a"]}}
UNTAGGED: Final[Mapping[str, JsonValue]] = {}


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    cache: OwnedRedis
    prefix: str
    wire: Wire

    def group(self, name: str) -> str:
        return f"{self.prefix}-{name}"

    def deployments(self, name: str) -> Mapping[str, str]:
        return {f"{self.group(name)}-{lane}": f"{self.wire.url}/{self.group(name)}/{lane}/v1" for lane in LANES}


@dataclass(frozen=True, slots=True)
class Outcome:
    status: int
    detail: str
    call_id: str


def is_model_probe(request: Request) -> bool:
    return (request.method, request.target) == ("GET", "/v1/models")


def rate_limited(request: Request) -> Reply:
    if is_model_probe(request):
        return Reply(body=b'{"object": "list", "data": []}')
    body: Final = {"error": {"message": "scripted 429", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}
    return Reply(status=429, body=json.dumps(body).encode())


def cooldown_config(directory: Path, cache: OwnedRedis, prefix: str, wire: Wire) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    assert isinstance(base, dict)
    model_list: Final = [
        {
            "model_name": f"{prefix}-{name}",
            "litellm_params": {
                "model": "openai/gpt-5.4",
                "api_base": f"{wire.url}/{prefix}-{name}/{lane}/v1",
                "api_key": "synthetic-openai-key",
                **LANE_PARAMS.get((name, lane), {}),
            },
            "model_info": {
                "id": f"{prefix}-{name}-{lane}",
                **({"enable_tag_filtering": True} if name == TAGGED else {}),
            },
        }
        for name, lane in product(GROUPS, LANES)
    ]
    config: Final = {
        **base,
        "model_list": model_list,
        "router_settings": {
            "num_retries": 0,
            "allowed_fails": 0,
            "cooldown_time": COOLDOWN_SECONDS,
            "enable_pre_call_checks": True,
            "redis_host": cache.host,
            "redis_port": cache.port,
        },
    }
    path: Final = directory / f"cooldown-{prefix}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("router-cooldown-failure-row")
    prefix: Final = f"cooldown-{uuid.uuid4().hex[:8]}"
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(directory))
        wire: Final = stack.enter_context(wire_server(rate_limited))
        config: Final = cooldown_config(directory, cache, prefix, wire)
        owned: Final = stack.enter_context(owned_proxy_process(upstream, directory, {}, config=config, workers=2))
        yield Rig(owned.gateway, cache, prefix, wire)


def cooled(cache: OwnedRedis, deployment_id: str) -> bool:
    with Redis(host=cache.host, port=cache.port) as client:
        return int(client.exists(f"deployment:{deployment_id}:cooldown")) == 1


def request_body(path: str, group: str, stream: bool) -> dict[str, JsonValue]:
    content: Final = f"cooldown row {uuid.uuid4().hex}"
    match path:
        case "/v1/messages":
            return {
                "model": group,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": content}],
                "stream": stream,
            }
        case "/v1/responses":
            return {"model": group, "input": content, "stream": stream}
        case _:
            return {"model": group, "messages": [{"role": "user", "content": content}], "stream": stream}


def send(gateway: Gateway, path: str, group: str, stream: bool = False) -> Outcome:
    response: Final = gateway.request("POST", path, request_body(path, group, stream), headers={"Connection": "close"})
    return Outcome(response.status_code, response.text, response.headers["x-litellm-call-id"])


def send_chat(rig: Rig, name: str, extra: Mapping[str, JsonValue]) -> Outcome:
    body: Final = {**request_body(CHAT, rig.group(name), False), **extra}
    response: Final = rig.gateway.request("POST", CHAT, body, headers={"Connection": "close"})
    return Outcome(response.status_code, response.text, response.headers["x-litellm-call-id"])


def failure_row(gateway: Gateway, call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: gateway.rows(FAILURE_ROW, (call_id,)), lambda found: len(found) == 1, seconds=ROW_SECONDS
    )
    assert rows[0]["status"] == "failure", rows
    return rows[0]


def until_router_refusal(attempt: Callable[[], Outcome], tries: int = 10) -> tuple[Outcome, ...]:
    def walk() -> Iterator[Outcome]:
        for _ in range(tries):
            outcome: Final = attempt()
            yield outcome
            if NO_DEPLOYMENTS in outcome.detail:
                return

    history: Final = tuple(walk())
    assert NO_DEPLOYMENTS in history[-1].detail, [(o.status, o.detail[:200]) for o in history]
    return history


def cool_group(rig: Rig, name: str) -> tuple[Outcome, ...]:
    history: Final = until_router_refusal(lambda: send(rig.gateway, CHAT, rig.group(name)))
    assert all(cooled(rig.cache, identity) for identity in rig.deployments(name)), history
    return history


def expected_cooldown(deployments: Mapping[str, str]) -> list[JsonValue]:
    return [{"model_id": identity, "api_base": deployments[identity]} for identity in sorted(deployments)]


def cooldown_entries(error: Mapping[str, JsonValue]) -> list[JsonValue] | None:
    entries: Final = error.get("error_cooldown_deployments")
    if not isinstance(entries, list):
        return None
    return sorted(entries, key=lambda entry: str(object_value(entry)["model_id"]))


def assert_refusal_row(gateway: Gateway, call_id: str, deployments: Mapping[str, str]) -> Mapping[str, JsonValue]:
    row: Final = failure_row(gateway, call_id)
    error: Final = object_value(row["error"])
    assert (error["error_class"], error["error_code"]) == ("RouterRateLimitError", "429"), row
    assert cooldown_entries(error) == expected_cooldown(deployments), error.get("error_message")
    assert (row["model_id"], row["api_base"]) == ("", ""), row
    return error


def provider_row(gateway: Gateway, call_id: str) -> tuple[str, str, str, JsonValue]:
    row: Final = failure_row(gateway, call_id)
    error: Final = object_value(row["error"])
    return str(row["model_id"]), str(row["api_base"]), str(error["error_code"]), error.get("error_cooldown_deployments")


def assert_provider_rows(gateway: Gateway, history: tuple[Outcome, ...], deployments: Mapping[str, str]) -> None:
    provider: Final = history[:-1]
    assert all(outcome.status == 429 and NO_DEPLOYMENTS not in outcome.detail for outcome in provider), [
        (o.status, o.detail[:200]) for o in provider
    ]
    rows: Final = tuple(provider_row(gateway, outcome.call_id) for outcome in provider)
    assert {identity for identity, _, _, _ in rows} == set(deployments), rows
    assert all(base.startswith(deployments[identity]) for identity, base, _, _ in rows), rows
    assert all((code, cooldown) == ("429", None) for _, _, code, cooldown in rows), rows


@pytest.mark.parametrize(("path", "stream"), ENDPOINT_CELLS, ids=ENDPOINT_GROUPS)
def test_all_cooled_refusal_row_records_429_and_every_cooled_deployment(rig: Rig, path: str, stream: bool) -> None:
    name: Final = ENDPOINT_GROUPS[ENDPOINT_CELLS.index((path, stream))]
    history: Final = until_router_refusal(lambda: send(rig.gateway, path, rig.group(name), stream))
    refusal: Final = history[-1]
    assert refusal.status == 429, (refusal.status, refusal.detail[:300])
    assert_refusal_row(rig.gateway, refusal.call_id, rig.deployments(name))
    assert_provider_rows(rig.gateway, history, rig.deployments(name))


def openai_chat(rig: Rig, group: str) -> Outcome:
    client: Final = openai.OpenAI(base_url=f"{rig.gateway.client.base_url}/v1", api_key=rig.gateway.key, max_retries=0)
    with pytest.raises(openai.RateLimitError) as raised:
        client.chat.completions.create(model=group, messages=[{"role": "user", "content": uuid.uuid4().hex}])
    return Outcome(429, raised.value.message, raised.value.response.headers["x-litellm-call-id"])


def openai_async_stream(rig: Rig, group: str) -> Outcome:
    async def call() -> Outcome:
        client: Final = openai.AsyncOpenAI(
            base_url=f"{rig.gateway.client.base_url}/v1", api_key=rig.gateway.key, max_retries=0
        )
        try:
            with pytest.raises(openai.RateLimitError) as raised:
                await client.chat.completions.create(
                    model=group, messages=[{"role": "user", "content": uuid.uuid4().hex}], stream=True
                )
            return Outcome(429, raised.value.message, raised.value.response.headers["x-litellm-call-id"])
        finally:
            await client.close()

    return asyncio.run(call())


def anthropic_message(rig: Rig, group: str) -> Outcome:
    client: Final = anthropic.Anthropic(
        base_url=str(rig.gateway.client.base_url), api_key=rig.gateway.key, max_retries=0
    )
    with pytest.raises(anthropic.RateLimitError) as raised:
        client.messages.create(model=group, max_tokens=8, messages=[{"role": "user", "content": uuid.uuid4().hex}])
    return Outcome(429, raised.value.message, raised.value.response.headers["x-litellm-call-id"])


SDK_CLIENTS: Final = {
    "openai-sync": openai_chat,
    "openai-async-stream": openai_async_stream,
    "anthropic-sync": anthropic_message,
}


@pytest.mark.parametrize("client", SDK_CELLS)
def test_sdk_clients_get_429_and_the_row_names_the_cooled_deployments(rig: Rig, client: str) -> None:
    history: Final = until_router_refusal(lambda: SDK_CLIENTS[client](rig, rig.group(client)))
    assert_refusal_row(rig.gateway, history[-1].call_id, rig.deployments(client))
    assert_provider_rows(rig.gateway, history, rig.deployments(client))


def test_the_row_lists_only_the_requested_groups_deployments_while_another_group_cools(rig: Rig) -> None:
    cool_group(rig, "neighbour")
    cool_group(rig, "scoped")
    refusal: Final = eventually(
        lambda: send(rig.gateway, CHAT, rig.group("scoped")),
        lambda outcome: (
            NO_DEPLOYMENTS in outcome.detail
            and all(identity in outcome.detail for identity in rig.deployments("neighbour"))
        ),
    )
    assert_refusal_row(rig.gateway, refusal.call_id, rig.deployments("scoped"))


def test_a_concurrent_burst_on_a_cooled_group_writes_one_complete_row_per_refusal(rig: Rig) -> None:
    cool_group(rig, "burst")
    rig.wire.drain()
    calls: Final = tuple(ENDPOINT_CELLS[index % len(ENDPOINT_CELLS)] for index in range(30))
    with (
        httpx.Client(
            base_url=rig.gateway.client.base_url, timeout=60, trust_env=False, limits=GATEWAY_LIMITS
        ) as client,
        ThreadPoolExecutor(max_workers=30) as pool,
    ):
        patient: Final = Gateway(client, rig.gateway.key, rig.gateway.upstream_url)
        outcomes: Final = tuple(pool.map(lambda call: send(patient, call[0], rig.group("burst"), call[1]), calls))
    assert all(outcome.status == 429 for outcome in outcomes), [(o.status, o.detail[:200]) for o in outcomes]
    assert len({outcome.call_id for outcome in outcomes}) == len(outcomes), outcomes
    refusals: Final = tuple(outcome for outcome in outcomes if NO_DEPLOYMENTS in outcome.detail)
    reached: Final = tuple(outcome for outcome in outcomes if NO_DEPLOYMENTS not in outcome.detail)
    assert refusals, [(o.status, o.detail[:200]) for o in outcomes]
    assert len(tuple(request for request in rig.wire.drain() if not is_model_probe(request))) == len(reached)
    for outcome in refusals:
        assert_refusal_row(rig.gateway, outcome.call_id, rig.deployments("burst"))
    for outcome in reached:
        identity, api_base, code, cooldown = provider_row(rig.gateway, outcome.call_id)
        assert api_base.startswith(rig.deployments("burst")[identity]), (identity, api_base)
        assert (code, cooldown) == ("429", None), outcome


def cool_lane(rig: Rig, name: str, lane: str, extra: Mapping[str, JsonValue]) -> None:
    identity: Final = f"{rig.group(name)}-{lane}"
    outcomes: Final = eventually(lambda: send_chat(rig, name, extra), lambda _: cooled(rig.cache, identity))
    assert NO_DEPLOYMENTS not in outcomes.detail, outcomes.detail[:300]


def router_refusal(rig: Rig, name: str, extra: Mapping[str, JsonValue]) -> Outcome:
    refusal: Final = eventually(lambda: send_chat(rig, name, extra), lambda outcome: NO_DEPLOYMENTS in outcome.detail)
    assert refusal.status == 429, (refusal.status, refusal.detail[:300])
    return refusal


def test_a_tag_routed_refusal_row_names_only_the_cooled_deployments_its_tags_reach(rig: Rig) -> None:
    deployments: Final = rig.deployments(TAGGED)
    tenant: Final = f"{rig.group(TAGGED)}-a"
    cool_lane(rig, TAGGED, "a", TENANT_A)
    cool_lane(rig, TAGGED, "b", UNTAGGED)
    assert_refusal_row(rig.gateway, router_refusal(rig, TAGGED, TENANT_A).call_id, {tenant: deployments[tenant]})
    assert_refusal_row(rig.gateway, router_refusal(rig, TAGGED, UNTAGGED).call_id, deployments)


def test_a_region_restricted_refusal_row_names_only_the_cooled_deployments_in_its_region(rig: Rig) -> None:
    deployments: Final = rig.deployments(REGIONAL)
    european: Final = f"{rig.group(REGIONAL)}-a"
    customer: Final = f"{rig.prefix}-eu-customer"
    with ExitStack() as cleanups:
        rig.gateway.post("/customer/new", {"user_id": customer, "allowed_model_region": "eu"})
        cleanups.callback(rig.gateway.post, "/customer/delete", {"user_ids": [customer]})
        cool_group(rig, REGIONAL)
        assert_refusal_row(
            rig.gateway, router_refusal(rig, REGIONAL, {"user": customer}).call_id, {european: deployments[european]}
        )
