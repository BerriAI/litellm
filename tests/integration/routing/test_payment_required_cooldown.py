from __future__ import annotations

import json
import re
import signal
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import GATEWAY_LIMITS, Gateway, Scenario, eventually, gateway_from_environment
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.prometheus_series import PROXY_FAILURES, Sample, scrape
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing.test_payment_required_mapping import (
    OPENAI_MODEL,
    PAYMENT_REQUIRED,
    anthropic_client,
    anthropic_error,
    anthropic_params,
    assert_payment_required_row,
    chat_body,
    deployment,
    is_model_info_probe,
    model_list_reply,
    new_group,
    new_marker,
    openai_client,
    openai_params,
    openai_refusal,
    post,
    provider_calls,
)
from pydantic import JsonValue
from redis import Redis

pytestmark: Final = pytest.mark.timeout(300)

COOLDOWN_SECONDS: Final = 300
RELOAD_SECONDS: Final = 2
NO_DEPLOYMENTS: Final = "No deployments available"
COOLED_DOWN: Final = "litellm_deployment_cooled_down_total"
PRIMARY_GROUP: Final = new_group()
FALLBACK_GROUP: Final = new_group()
MARKER: Final = re.compile(rb"pr402-[0-9a-f]{32}(?:-\d+)?")
WORKER_PID: Final = re.compile(r"Started server process \[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Outcome:
    marker: str
    status: int
    detail: str
    call_id: str


@dataclass(frozen=True, slots=True)
class Call:
    marker: str
    path: str
    body: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class CooldownRig:
    gateway: Gateway
    cache: OwnedRedis
    log: Path

    def cooled(self, deployment_id: str) -> bool:
        return cooled(self.cache, deployment_id)


def cooled(cache: OwnedRedis, deployment_id: str) -> bool:
    with Redis(host=cache.host, port=cache.port) as client:
        return int(client.exists(f"deployment:{deployment_id}:cooldown")) == 1


def marker_in(request: Request) -> str:
    found: Final = MARKER.search(request.body)
    assert found is not None, (request.method, request.target, request.headers, request.body[:300])
    return found.group(0).decode()


def markers_received(wire: Wire) -> tuple[str, ...]:
    return tuple(marker_in(request) for request in provider_calls(wire))


def refusing_reply(request: Request) -> Reply:
    return anthropic_error(402, f"scripted 402 {marker_in(request)}")


def healthy_reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    marker: Final = marker_in(request)
    body: Final = json.loads(request.body)
    stream: Final = isinstance(body, dict) and body.get("stream") is True
    if request.target.endswith("/responses"):
        return responses_reply(f"resp_{marker}", OPENAI_MODEL, f"served {marker}", stream=stream)
    return chat_reply(f"chatcmpl-{marker}", OPENAI_MODEL, f"served {marker}", stream=stream)


def router_settings(cache: OwnedRedis, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "num_retries": 0,
        "cooldown_time": COOLDOWN_SECONDS,
        "redis_host": cache.host,
        "redis_port": cache.port,
        **extra,
    }


def cooldown_config(
    directory: Path,
    settings: Mapping[str, JsonValue],
    *,
    model_list: Sequence[Mapping[str, JsonValue]] = (),
    prometheus: bool = True,
) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    assert isinstance(base, dict)
    litellm_settings: Final = (
        {**base["litellm_settings"], "callbacks": ["prometheus"]} if prometheus else base["litellm_settings"]
    )
    config: Final = {
        **base,
        "litellm_settings": litellm_settings,
        "router_settings": dict(settings),
        **({"model_list": [dict(entry) for entry in model_list]} if model_list else {}),
    }
    path: Final = directory / f"cooldown-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CooldownRig]:
    directory: Final = tmp_path_factory.mktemp("payment-required-cooldown")
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(directory))
        settings: Final = router_settings(cache, fallbacks=[{PRIMARY_GROUP: [FALLBACK_GROUP]}])
        owned: Final = stack.enter_context(
            owned_proxy_process(
                upstream,
                directory,
                {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(RELOAD_SECONDS)},
                config=cooldown_config(directory, settings),
                remove_environment=("PROMETHEUS_MULTIPROC_DIR",),
            )
        )
        yield CooldownRig(owned.gateway, cache, owned.log)


def two_deployments(scenario: Scenario, group: str, refusing: Wire, healthy: Wire) -> str:
    refused_id: Final = deployment(scenario, group, anthropic_params(refusing))
    deployment(scenario, group, openai_params(healthy))
    return refused_id


def raw_chat(gateway: Gateway, group: str, marker: str, **extra: JsonValue) -> Outcome:
    response: Final = post(gateway, "/v1/chat/completions", chat_body(group, marker, **extra))
    return Outcome(marker, response.status_code, response.text, response.headers["x-litellm-call-id"])


def raw_call(gateway: Gateway, call: Call) -> Outcome:
    response: Final = post(gateway, call.path, call.body)
    return Outcome(call.marker, response.status_code, response.text, response.headers["x-litellm-call-id"])


def openai_chat(client: openai.OpenAI, group: str, marker: str) -> Outcome:
    try:
        raw: Final = client.chat.completions.with_raw_response.create(
            model=group, messages=[{"role": "user", "content": marker}]
        )
    except openai.APIStatusError as error:
        return Outcome(marker, error.status_code, error.message, error.response.headers["x-litellm-call-id"])
    return Outcome(marker, raw.status_code, raw.parse().id, raw.headers["x-litellm-call-id"])


def anthropic_message(client: anthropic.Anthropic, group: str, marker: str) -> Outcome:
    try:
        raw: Final = client.messages.with_raw_response.create(
            model=group, max_tokens=32, messages=[{"role": "user", "content": marker}]
        )
    except anthropic.APIStatusError as error:
        return Outcome(marker, error.status_code, error.message, error.response.headers["x-litellm-call-id"])
    block: Final = raw.parse().content[0]
    assert isinstance(block, anthropic.types.TextBlock), block
    return Outcome(marker, raw.status_code, block.text, raw.headers["x-litellm-call-id"])


def until_refused(attempt: Callable[[str], Outcome], tries: int = 40) -> tuple[Outcome, ...]:
    def walk() -> Iterator[Outcome]:
        for _ in range(tries):
            outcome: Final = attempt(new_marker())
            yield outcome
            if outcome.status != 200:
                return

    history: Final = tuple(walk())
    assert history[-1].status != 200, "the refusing deployment was never picked"
    return history


def assert_refused(outcome: Outcome) -> None:
    assert outcome.status == 402, (outcome.status, outcome.detail)
    assert PAYMENT_REQUIRED in outcome.detail, outcome.detail


def assert_served(outcomes: Sequence[Outcome], identity: Callable[[str], str]) -> None:
    assert all(outcome.status == 200 for outcome in outcomes), [(o.status, o.detail[:200]) for o in outcomes]
    assert all(identity(outcome.marker) in outcome.detail for outcome in outcomes), [o.detail[:200] for o in outcomes]


def chat_identity(marker: str) -> str:
    return f"chatcmpl-{marker}"


def served_text(marker: str) -> str:
    return f"served {marker}"


def response_identity(marker: str) -> str:
    return f"resp_{marker}"


def markers(outcomes: Sequence[Outcome]) -> tuple[str, ...]:
    return tuple(outcome.marker for outcome in outcomes)


def failure_samples(gateway: Gateway, group: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in scrape(gateway)
        if sample.name == PROXY_FAILURES and sample.labels.get("requested_model") == group
    )


def cooled_down_samples(gateway: Gateway, deployment_id: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in scrape(gateway)
        if sample.name == COOLED_DOWN and sample.labels.get("model_id") == deployment_id
    )


def assert_payment_required_metrics(gateway: Gateway, group: str, deployment_id: str) -> None:
    failures: Final = eventually(lambda: failure_samples(gateway, group), lambda found: len(found) >= 1, seconds=10)
    assert [(sample.labels["exception_class"], sample.labels["exception_status"]) for sample in failures] == [
        ("Anthropic.PaymentRequiredError", "402")
    ], failures
    cooldowns: Final = eventually(
        lambda: cooled_down_samples(gateway, deployment_id), lambda found: len(found) >= 1, seconds=10
    )
    assert [sample.labels["exception_status"] for sample in cooldowns] == ["402"], cooldowns


def test_chat_openai_sdk_402_cools_the_deployment_and_the_healthy_one_serves_after(rig: CooldownRig) -> None:
    group: Final = new_group()
    with (
        wire_server(refusing_reply) as refusing,
        wire_server(healthy_reply) as healthy,
        rig.gateway.scenario() as scenario,
    ):
        refused_id: Final = two_deployments(scenario, group, refusing, healthy)
        client: Final = openai_client(rig.gateway)
        history: Final = until_refused(lambda marker: openai_chat(client, group, marker))
        assert_refused(history[-1])
        assert_served(history[:-1], chat_identity)
        eventually(lambda: rig.cooled(refused_id), lambda seen: seen, seconds=10)
        assert markers_received(refusing) == (history[-1].marker,)
        assert markers_received(healthy) == markers(history[:-1])
        served: Final = tuple(openai_chat(client, group, new_marker()) for _ in range(20))
        assert_served(served, chat_identity)
        assert markers_received(healthy) == markers(served)
        assert markers_received(refusing) == ()
        assert_payment_required_row(history[-1].call_id, refused_id)
        assert_payment_required_metrics(rig.gateway, group, refused_id)


def test_messages_anthropic_sdk_402_cools_the_deployment_and_the_healthy_one_serves_after(rig: CooldownRig) -> None:
    group: Final = new_group()
    with (
        wire_server(refusing_reply) as refusing,
        wire_server(healthy_reply) as healthy,
        rig.gateway.scenario() as scenario,
    ):
        refused_id: Final = two_deployments(scenario, group, refusing, healthy)
        client: Final = anthropic_client(rig.gateway)
        history: Final = until_refused(lambda marker: anthropic_message(client, group, marker))
        assert_refused(history[-1])
        assert_served(history[:-1], served_text)
        eventually(lambda: rig.cooled(refused_id), lambda seen: seen, seconds=10)
        assert markers_received(refusing) == (history[-1].marker,)
        assert markers_received(healthy) == markers(history[:-1])
        served: Final = tuple(anthropic_message(client, group, new_marker()) for _ in range(20))
        assert_served(served, served_text)
        assert markers_received(healthy) == markers(served)
        assert markers_received(refusing) == ()
        assert_payment_required_row(history[-1].call_id, refused_id)


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_responses_httpx_402_cools_the_deployment_and_the_healthy_one_serves_after(
    rig: CooldownRig, stream: bool
) -> None:
    group: Final = new_group()
    with (
        wire_server(refusing_reply) as refusing,
        wire_server(healthy_reply) as healthy,
        rig.gateway.scenario() as scenario,
    ):
        refused_id: Final = two_deployments(scenario, group, refusing, healthy)

        def attempt(marker: str) -> Outcome:
            return raw_call(
                rig.gateway, Call(marker, "/v1/responses", {"model": group, "input": marker, "stream": stream})
            )

        history: Final = until_refused(attempt)
        assert_refused(history[-1])
        assert_served(history[:-1], response_identity)
        eventually(lambda: rig.cooled(refused_id), lambda seen: seen, seconds=10)
        assert markers_received(refusing) == (history[-1].marker,)
        assert markers_received(healthy) == markers(history[:-1])
        served: Final = tuple(attempt(new_marker()) for _ in range(20))
        assert_served(served, response_identity)
        assert markers_received(healthy) == markers(served)
        assert markers_received(refusing) == ()
        assert_payment_required_row(history[-1].call_id, refused_id)


@pytest.mark.parametrize(
    "model_info",
    ({}, {"allowed_fails_policy": None}, {"allowed_fails_policy": {}}),
    ids=("missing", "null", "empty"),
)
def test_a_lone_402_deployment_without_a_policy_is_never_cooled(
    rig: CooldownRig, model_info: dict[str, JsonValue]
) -> None:
    group: Final = new_group()
    with wire_server(refusing_reply) as refusing, rig.gateway.scenario() as scenario:
        refused_id: Final = deployment(scenario, group, anthropic_params(refusing), model_info=model_info)
        refused: Final = tuple(raw_chat(rig.gateway, group, new_marker()) for _ in range(5))
        for outcome in refused:
            assert_refused(outcome)
            assert_payment_required_row(outcome.call_id, refused_id)
        assert not rig.cooled(refused_id)
        sixth: Final = raw_chat(rig.gateway, group, new_marker())
        assert_refused(sixth)
        assert markers_received(refusing) == (*markers(refused), sixth.marker)


@pytest.mark.timeout(900)
def test_a_router_allowed_fails_policy_cools_a_lone_402_deployment_past_its_count(tmp_path: Path) -> None:
    group: Final = new_group()
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(tmp_path))
        settings: Final = router_settings(cache, allowed_fails_policy={"BadRequestErrorAllowedFails": 1})
        owned: Final = stack.enter_context(
            owned_proxy_process(
                upstream,
                tmp_path,
                {},
                config=cooldown_config(tmp_path, settings),
                remove_environment=("PROMETHEUS_MULTIPROC_DIR",),
            )
        )
        refusing: Final = stack.enter_context(wire_server(refusing_reply))
        scenario: Final = stack.enter_context(owned.gateway.scenario())
        refused_id: Final = deployment(scenario, group, anthropic_params(refusing))
        first: Final = raw_chat(owned.gateway, group, new_marker())
        second: Final = raw_chat(owned.gateway, group, new_marker())
        assert_refused(first)
        assert_refused(second)
        eventually(lambda: cooled(cache, refused_id), lambda seen: seen, seconds=10)
        third: Final = raw_chat(owned.gateway, group, new_marker())
        assert third.status == 429 and NO_DEPLOYMENTS in third.detail, (third.status, third.detail)
        assert markers_received(refusing) == (first.marker, second.marker)


def test_a_deployment_allowed_fails_policy_cools_a_lone_402_deployment(rig: CooldownRig) -> None:
    group: Final = new_group()
    policy: Final = {"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}
    with wire_server(refusing_reply) as refusing, rig.gateway.scenario() as scenario:
        refused_id: Final = deployment(scenario, group, anthropic_params(refusing), model_info=policy)
        first: Final = raw_chat(rig.gateway, group, new_marker())
        assert_refused(first)
        eventually(lambda: rig.cooled(refused_id), lambda seen: seen, seconds=10)
        second: Final = raw_chat(rig.gateway, group, new_marker())
        assert second.status == 429 and NO_DEPLOYMENTS in second.detail, (second.status, second.detail)
        assert markers_received(refusing) == (first.marker,)


def test_a_plain_allowed_fails_count_leaves_a_lone_402_deployment_warm(rig: CooldownRig) -> None:
    group: Final = new_group()
    with wire_server(refusing_reply) as refusing, rig.gateway.scenario() as scenario:
        refused_id: Final = deployment(scenario, group, anthropic_params(refusing), model_info={"allowed_fails": 0})
        refused: Final = tuple(raw_chat(rig.gateway, group, new_marker()) for _ in range(5))
        for outcome in refused:
            assert_refused(outcome)
        assert not rig.cooled(refused_id)
        sixth: Final = raw_chat(rig.gateway, group, new_marker())
        assert_refused(sixth)
        assert markers_received(refusing) == (*markers(refused), sixth.marker)


def test_an_openai_compatible_402_cools_the_deployment_too(rig: CooldownRig) -> None:
    group: Final = new_group()
    with (
        wire_server(openai_refusal) as refusing,
        wire_server(healthy_reply) as healthy,
        rig.gateway.scenario() as scenario,
    ):
        refused_id: Final = deployment(scenario, group, openai_params(refusing))
        deployment(scenario, group, openai_params(healthy))
        history: Final = until_refused(lambda marker: raw_chat(rig.gateway, group, marker))
        assert history[-1].status == 402 and "scripted 402" in history[-1].detail, history[-1]
        assert_served(history[:-1], chat_identity)
        eventually(lambda: rig.cooled(refused_id), lambda seen: seen, seconds=10)
        assert markers_received(refusing) == (history[-1].marker,)
        served: Final = tuple(raw_chat(rig.gateway, group, new_marker()) for _ in range(20))
        assert_served(served, chat_identity)
        assert markers_received(refusing) == ()
        assert markers_received(healthy) == (*markers(history[:-1]), *markers(served))


def test_a_lone_402_primary_with_a_fallback_group_stays_warm_and_the_fallback_answers(rig: CooldownRig) -> None:
    with (
        wire_server(refusing_reply) as refusing,
        wire_server(healthy_reply) as healthy,
        rig.gateway.scenario() as scenario,
    ):
        primary_id: Final = deployment(scenario, PRIMARY_GROUP, anthropic_params(refusing))
        deployment(scenario, FALLBACK_GROUP, openai_params(healthy))
        served: Final = tuple(raw_chat(rig.gateway, PRIMARY_GROUP, new_marker()) for _ in range(3))
        assert_served(served, chat_identity)
        assert markers_received(refusing) == markers(served)
        assert markers_received(healthy) == markers(served)
        assert not rig.cooled(primary_id)


def test_a_deployment_policy_added_while_402_traffic_flows_starts_the_cooldown(rig: CooldownRig) -> None:
    group: Final = new_group()
    with wire_server(refusing_reply) as refusing, rig.gateway.scenario() as scenario:
        refused_id: Final = deployment(scenario, group, anthropic_params(refusing))
        before: Final = tuple(raw_chat(rig.gateway, group, new_marker()) for _ in range(3))
        for outcome in before:
            assert_refused(outcome)
        assert not rig.cooled(refused_id)
        policy: Final = {"id": refused_id, "allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}
        patched: Final = rig.gateway.request("PATCH", f"/model/{refused_id}/update", {"model_info": policy})
        assert patched.status_code == 200, patched.text

        def probe() -> tuple[int, bool]:
            return raw_chat(rig.gateway, group, new_marker()).status, rig.cooled(refused_id)

        eventually(probe, lambda seen: seen[1], seconds=RELOAD_SECONDS * 10)
        after: Final = raw_chat(rig.gateway, group, new_marker())
        assert after.status == 429 and NO_DEPLOYMENTS in after.detail, (after.status, after.detail)
        assert len(markers_received(refusing)) >= 4


def test_a_cooled_402_deployment_comes_back_after_its_own_cooldown_time(rig: CooldownRig) -> None:
    group: Final = new_group()
    with (
        wire_server(refusing_reply) as refusing,
        wire_server(healthy_reply) as healthy,
        rig.gateway.scenario() as scenario,
    ):
        refused_id: Final = deployment(scenario, group, {**anthropic_params(refusing), "cooldown_time": 2})
        deployment(scenario, group, openai_params(healthy))
        history: Final = until_refused(lambda marker: raw_chat(rig.gateway, group, marker))
        assert_refused(history[-1])
        eventually(lambda: rig.cooled(refused_id), lambda seen: seen, seconds=10)
        eventually(lambda: rig.cooled(refused_id), lambda seen: not seen, seconds=10)
        again: Final = until_refused(lambda marker: raw_chat(rig.gateway, group, marker))
        assert_refused(again[-1])
        assert markers_received(refusing) == (history[-1].marker, again[-1].marker)


def burst_call(group: str, index: int) -> Call:
    marker: Final = f"{new_marker()}-{index}"
    stream: Final = index % 2 == 1
    match index % 3:
        case 0:
            return Call(marker, "/v1/chat/completions", chat_body(group, marker, stream=stream))
        case 1:
            body: Final = {"model": group, "max_tokens": 32, "messages": [{"role": "user", "content": marker}]}
            return Call(marker, "/v1/messages", {**body, "stream": stream})
        case _:
            return Call(marker, "/v1/responses", {"model": group, "input": marker, "stream": stream})


def worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(pid) for pid in WORKER_PID.findall(log.read_text()))


def ready_workers(log: Path) -> int:
    return log.read_text().count("Application startup complete.")


@pytest.mark.timeout(900)
def test_chaos_burst_cooldown_holds_across_both_workers_and_a_worker_kill(tmp_path: Path) -> None:
    group: Final = new_group()
    refused_id: Final = f"refusing-{uuid.uuid4().hex[:12]}"
    healthy_id: Final = f"healthy-{uuid.uuid4().hex[:12]}"
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(tmp_path))
        refusing: Final = stack.enter_context(wire_server(refusing_reply))
        healthy: Final = stack.enter_context(wire_server(healthy_reply))
        model_list: Final = (
            {"model_name": group, "litellm_params": anthropic_params(refusing), "model_info": {"id": refused_id}},
            {"model_name": group, "litellm_params": openai_params(healthy), "model_info": {"id": healthy_id}},
        )
        config: Final = cooldown_config(tmp_path, router_settings(cache), model_list=model_list, prometheus=False)
        owned: Final = stack.enter_context(owned_proxy_process(upstream, tmp_path, {}, config=config, workers=2))
        eventually(lambda: ready_workers(owned.log), lambda ready: ready == 2, seconds=graceful_stop_seconds())
        client: Final = stack.enter_context(
            httpx.Client(base_url=owned.gateway.client.base_url, timeout=60, trust_env=False, limits=GATEWAY_LIMITS)
        )
        patient: Final = Gateway(client, owned.gateway.key, owned.gateway.upstream_url)
        calls: Final = tuple(burst_call(group, index) for index in range(30))
        with ThreadPoolExecutor(max_workers=30) as pool:
            outcomes: Final = tuple(pool.map(lambda call: raw_call(patient, call), calls))
        assert {outcome.status for outcome in outcomes} <= {200, 402}, [(o.status, o.detail[:200]) for o in outcomes]
        refused: Final = tuple(outcome for outcome in outcomes if outcome.status == 402)
        served: Final = tuple(outcome for outcome in outcomes if outcome.status == 200)
        assert refused, "the burst never reached the refusing deployment"
        for outcome in refused:
            assert_refused(outcome)
        assert all(outcome.marker in outcome.detail for outcome in served), [o.detail[:200] for o in served]
        assert sorted(markers_received(refusing)) == sorted(markers(refused))
        assert sorted(markers_received(healthy)) == sorted(markers(served))
        eventually(lambda: cooled(cache, refused_id), lambda seen: seen, seconds=10)
        warm: Final = tuple(raw_chat(patient, group, new_marker()) for _ in range(20))
        assert_served(warm, chat_identity)
        assert markers_received(refusing) == ()
        assert markers_received(healthy) == markers(warm)
        victim: Final = next(pid for pid in sorted(worker_pids(owned.log)) if psutil.pid_exists(pid))
        ready_before: Final = ready_workers(owned.log)
        psutil.Process(victim).send_signal(signal.SIGKILL)
        eventually(
            lambda: ready_workers(owned.log), lambda ready: ready > ready_before, seconds=graceful_stop_seconds()
        )
        after_kill: Final = tuple(raw_chat(patient, group, new_marker()) for _ in range(20))
        assert_served(after_kill, chat_identity)
        assert markers_received(refusing) == ()
        assert markers_received(healthy) == markers(after_kill)
