import asyncio
import hashlib
import os
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import partial
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import websockets
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    held,
    list_value,
    object_value,
    string_value,
)
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_ws_peer import (
    PROVIDER_MODEL,
    USAGE,
    Peer,
    Script,
    completed,
    create,
    drain,
    events,
    responses_peer,
    turn,
    until_terminal,
)
from integration._support.tls import write_self_signed_cert
from pydantic import JsonValue
from redis import Redis

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
TURN_SPEND: Final = USAGE["input_tokens"] * INPUT_RATE + USAGE["output_tokens"] * OUTPUT_RATE
TURN_TOKENS: Final = USAGE["total_tokens"]
MANAGED_PROVIDER: Final = "mistral"
SPEND_ROWS_FOR_KEY: Final = (
    'SELECT request_id, spend, status, model_group, metadata, request_tags FROM "LiteLLM_SpendLogs" WHERE api_key = %s'
)
TURN_CHECK_FAULT: Final = "turn_check_fault"
FAULTING_TURN_CHECK: Final = f"""
from litellm.integrations.custom_logger import CustomLogger


class FaultingTurnCheck(CustomLogger):
    async def async_pre_call_check(self, deployment, parent_otel_span):
        fault = (deployment.get("model_info") or {{}}).get("{TURN_CHECK_FAULT}")
        if fault:
            raise RuntimeError(fault)
        return None


instance = FaultingTurnCheck()
"""
FrameReader = Callable[[], int]


@dataclass(frozen=True, slots=True)
class Deployment:
    name: str
    identity: str
    forwarded_turns: FrameReader


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _redis() -> Redis:
    return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True)


def _config(directory: Path, router_settings: Mapping[str, JsonValue], callbacks: tuple[str, ...]) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(PROXY_CONFIG.read_text()))
    path: Final = directory / "turn-admission.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "litellm_settings": {
                    **object_value(base["litellm_settings"]),
                    **({"callbacks": list(callbacks)} if callbacks else {}),
                },
                "router_settings": {
                    **object_value(base["router_settings"]),
                    "redis_host": os.environ["REDIS_HOST"],
                    "redis_port": int(os.environ["REDIS_PORT"]),
                    **router_settings,
                },
            }
        )
    )
    return path


@contextmanager
def _admitting_proxy(
    tmp_path_factory: pytest.TempPathFactory,
    cert: tuple[Path, Path],
    router_settings: Mapping[str, JsonValue],
    callback_sources: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("responses-ws-turn-admission")
    for module, source in callback_sources.items():
        (directory / f"{module}.py").write_text(source)
    config: Final = _config(directory, router_settings, tuple(f"{module}.instance" for module in callback_sources))
    with gateway_from_environment() as base:
        with owned_proxy(base, directory, {"SSL_CERT_FILE": str(cert[0])}, config=config) as proxy:
            yield proxy


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("responses-ws-turn-admission-cert"))


@pytest.fixture(scope="module")
def limited(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with _admitting_proxy(tmp_path_factory, cert, {"optional_pre_call_checks": ["enforce_model_rate_limits"]}) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def faulting(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with _admitting_proxy(tmp_path_factory, cert, {}, {"faulting_turn_check": FAULTING_TURN_CHECK}) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def usage_routed(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with _admitting_proxy(tmp_path_factory, cert, {"routing_strategy": "usage-based-routing-v2"}) as proxy:
        yield proxy


def _native(scenario: Scenario, peer: Peer, **limits: JsonValue) -> Deployment:
    identity: Final = f"ws-admission-{uuid.uuid4().hex}"
    name: Final = scenario.model(
        model_info={"id": identity},
        model=f"openai/{PROVIDER_MODEL}",
        api_base=peer.url,
        input_cost_per_token=INPUT_RATE,
        output_cost_per_token=OUTPUT_RATE,
        **limits,
    )
    return Deployment(
        name, identity, lambda: sum(frame.get("type") == "response.create" for frame in drain(peer.frames))
    )


def _upstream_chats(gateway: Gateway, provider_model: str) -> int:
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as upstream:
        observed: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
    assert isinstance(observed, list)
    return sum(object_value(object_value(entry)["body"]).get("model") == provider_model for entry in observed)


def _managed(scenario: Scenario, **limits: JsonValue) -> Deployment:
    identity: Final = f"ws-admission-{uuid.uuid4().hex}"
    provider_model: Final = f"ws-managed-{uuid.uuid4().hex}"
    name: Final = scenario.model(
        model_info={"id": identity},
        model=f"{MANAGED_PROVIDER}/{provider_model}",
        input_cost_per_token=INPUT_RATE,
        output_cost_per_token=OUTPUT_RATE,
        **limits,
    )
    return Deployment(name, identity, lambda: _upstream_chats(scenario.gateway, provider_model))


def _socket_url(gateway: Gateway, model: str) -> str:
    return f"{str(gateway.client.base_url).rstrip('/').replace('http://', 'ws://')}/v1/responses?model={model}"


@asynccontextmanager
async def _socket(
    gateway: Gateway, key: str, model: str, headers: Mapping[str, str] | None = None
) -> AsyncIterator[websockets.ClientConnection]:
    async with websockets.connect(
        _socket_url(gateway, model),
        additional_headers={"Authorization": f"Bearer {key}", **(headers or {})},
        open_timeout=10,
    ) as connection:
        yield connection


async def _say(connection: websockets.ClientConnection, model: str, text: str = "") -> tuple[dict[str, JsonValue], ...]:
    return await turn(connection, create(model, text or f"turn {uuid.uuid4().hex}"))


def _assert_refused(received: tuple[dict[str, JsonValue], ...]) -> None:
    assert [event.get("type") for event in received] == ["error"], received
    assert received[0]["status"] == 429, received
    assert object_value(received[0]["error"])["type"] == "rate_limit_exceeded", received


def _fresh_minute() -> str:
    eventually(lambda: datetime.now(UTC), lambda now: now.second < 30, seconds=65)
    return datetime.now(UTC).strftime("%H-%M")


def _deployment_spend(deployment: Deployment) -> float:
    with _redis() as redis:
        return float(redis.get(f"deployment_spend:{deployment.identity}:1d") or 0)


def _configured(gateway: Gateway, deployment: Deployment, section: str) -> dict[str, JsonValue]:
    entries: Final = list_value(gateway.get("/model/info", {"litellm_model_id": deployment.identity})["data"])
    return object_value(object_value(entries[0])[section])


def _configured_tpm(gateway: Gateway, deployment: Deployment) -> JsonValue:
    return _configured(gateway, deployment, "litellm_params").get("tpm")


def _tokens_counted(identity: str, minute: str) -> int:
    with _redis() as redis:
        return sum(int(redis.get(key) or 0) for key in redis.scan_iter(f"{identity}:*:tpm:{minute}"))


def _settled_count(read: Callable[[], int]) -> int:
    reached: Final = eventually(read, lambda tokens: tokens > 0, seconds=15)
    return held(read, lambda tokens: tokens == reached, holding=3, seconds=20)


async def _turns_against_tpm(
    gateway: Gateway, key: str, deployment: Deployment
) -> tuple[tuple[dict[str, JsonValue], ...], tuple[dict[str, JsonValue], ...], tuple[dict[str, JsonValue], ...], int]:
    minute: Final = await asyncio.to_thread(_fresh_minute)
    tokens: Final = partial(_tokens_counted, deployment.identity, minute)
    async with _socket(gateway, key, deployment.name) as connection:
        first: Final = await _say(connection, deployment.name)
        counted_after_first: Final = await asyncio.to_thread(_settled_count, tokens)
        second: Final = await _say(connection, deployment.name)
        if second[-1].get("type") == "response.completed":
            await _until(tokens, lambda counted: counted > counted_after_first)
        return first, second, await _say(connection, deployment.name), counted_after_first


async def _until(read: Callable[[], float], satisfied: Callable[[float], bool]) -> float:
    return await asyncio.to_thread(eventually, read, satisfied, 15)


def _failure_rows(gateway: Gateway, key: str) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: gateway.rows(SPEND_ROWS_FOR_KEY, (_key_hash(key),)),
        lambda found: any(row["status"] == "failure" for row in found),
        seconds=70,
    )
    return [row for row in rows if row["status"] == "failure"]


@pytest.mark.parametrize("relay", ("native", "managed"))
def test_a_turn_past_the_deployment_budget_is_refused_on_the_open_socket_and_logged_once_as_its_failure(
    limited: Gateway, cert: tuple[Path, Path], relay: str
) -> None:
    with responses_peer(cert) as peer, limited.scenario() as scenario:
        budget: Final = {"max_budget": TURN_SPEND / 2, "budget_duration": "1d"}
        deployment: Final = _native(scenario, peer, **budget) if relay == "native" else _managed(scenario, **budget)
        key: Final = scenario.key(models=[deployment.name])
        tag: Final = f"ws-turn-{uuid.uuid4().hex}"

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(limited, key, deployment.name, {"x-litellm-tags": tag}) as connection:
                first: Final = await _say(connection, deployment.name)
                await _until(lambda: _deployment_spend(deployment), lambda spent: spent > TURN_SPEND / 2)
                return (first, await _say(connection, deployment.name), await _say(connection, deployment.name))

        first, second, third = asyncio.run(session())

        completed(first)
        _assert_refused(second)
        _assert_refused(third)
        assert deployment.forwarded_turns() == 1
        failures: Final = _failure_rows(limited, key)
        assert len(failures) == 1, failures
        assert failures[0]["model_group"] == deployment.name, failures
        assert tag in list_value(failures[0]["request_tags"]), failures
        assert float(failures[0]["spend"]) == 0.0, failures
        error: Final = object_value(object_value(failures[0]["metadata"])["error_information"])
        assert str(error["error_code"]) == "429", error


def _set_turn_check_fault(gateway: Gateway, deployment: Deployment, fault: str) -> None:
    gateway.request(
        "PATCH",
        f"/model/{deployment.identity}/update",
        {"model_info": {"id": deployment.identity, TURN_CHECK_FAULT: fault}},
    ).raise_for_status()
    eventually(
        lambda: _configured(gateway, deployment, "model_info").get(TURN_CHECK_FAULT) or "",
        lambda configured: configured == fault,
        seconds=15,
    )


def _assert_turn_check_failed(received: tuple[dict[str, JsonValue], ...], fault: str) -> None:
    assert [event.get("type") for event in received] == ["error"], received
    assert received[0]["status"] == 500, received
    error: Final = object_value(received[0]["error"])
    assert error["type"] == "server_error", received
    assert fault in string_value(error["message"]), received


def _first_turn_held_until(release: threading.Event) -> Script:
    def script(response_id: str) -> tuple[dict[str, JsonValue], ...]:
        if response_id == "resp_peer_1":
            release.wait(timeout=60)
        return events(f"resp_{uuid.uuid4().hex}", "seven")

    return script


@pytest.mark.parametrize("relay", ("native", "managed"))
def test_a_turn_whose_admission_check_fails_gets_a_500_frame_and_the_socket_serves_the_next_turn(
    faulting: Gateway, cert: tuple[Path, Path], relay: str
) -> None:
    with responses_peer(cert) as peer, faulting.scenario() as scenario:
        deployment: Final = _native(scenario, peer) if relay == "native" else _managed(scenario)
        key: Final = scenario.key(models=[deployment.name])
        fault: Final = f"turn check unavailable {uuid.uuid4().hex}"

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(faulting, key, deployment.name) as connection:
                first: Final = await _say(connection, deployment.name)
                await asyncio.to_thread(_set_turn_check_fault, faulting, deployment, fault)
                second: Final = await _say(connection, deployment.name)
                third: Final = await _say(connection, deployment.name)
                await asyncio.to_thread(_set_turn_check_fault, faulting, deployment, "")
                return first, second, third, await _say(connection, deployment.name)

        first, second, third, fourth = asyncio.run(session())

        completed(first)
        _assert_turn_check_failed(second, fault)
        _assert_turn_check_failed(third, fault)
        completed(fourth)
        assert deployment.forwarded_turns() == 2
        failures: Final = _failure_rows(faulting, key)
        assert len(failures) == 1, failures
        assert failures[0]["model_group"] == deployment.name, failures
        recorded: Final = object_value(object_value(failures[0]["metadata"])["error_information"])
        assert fault in string_value(recorded["error_message"]), recorded


def test_a_turn_still_streaming_when_the_next_turns_check_fails_completes_and_is_billed(
    faulting: Gateway, cert: tuple[Path, Path]
) -> None:
    release: Final = threading.Event()
    with responses_peer(cert, _first_turn_held_until(release)) as peer, faulting.scenario() as scenario:
        deployment: Final = _native(scenario, peer)
        key: Final = scenario.key(models=[deployment.name])
        fault: Final = f"turn check unavailable {uuid.uuid4().hex}"

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(faulting, key, deployment.name) as connection:
                try:
                    await connection.send(create(deployment.name, f"streaming {uuid.uuid4().hex}"))
                    await asyncio.to_thread(_set_turn_check_fault, faulting, deployment, fault)
                    rejected: Final = await _say(connection, deployment.name)
                finally:
                    release.set()
                return rejected, await until_terminal(connection, ())

        rejected, streamed = asyncio.run(session())

        _assert_turn_check_failed(rejected, fault)
        billed: Final = completed(streamed)
        assert deployment.forwarded_turns() == 1
        rows: Final = eventually(
            lambda: faulting.rows(SPEND_ROWS_FOR_KEY, (_key_hash(key),)),
            lambda found: any(row["request_id"] == billed["id"] for row in found),
            seconds=70,
        )
        row: Final = next(row for row in rows if row["request_id"] == billed["id"])
        assert row["status"] == "success", row
        assert float(row["spend"]) == pytest.approx(TURN_SPEND), row


def test_a_socket_opened_after_websocket_turns_spent_the_budget_is_refused_and_so_is_http(
    limited: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, limited.scenario() as scenario:
        deployment: Final = _native(scenario, peer, max_budget=TURN_SPEND / 2, budget_duration="1d")
        key: Final = scenario.key(models=[deployment.name])

        async def spend_then_reopen() -> tuple[tuple[dict[str, JsonValue], ...], int | None]:
            async with _socket(limited, key, deployment.name) as spending:
                completed(await _say(spending, deployment.name))
            await _until(lambda: _deployment_spend(deployment), lambda spent: spent > TURN_SPEND / 2)
            async with _socket(limited, key, deployment.name) as refused:
                received: Final = await _say(refused, deployment.name)
                await asyncio.wait_for(refused.wait_closed(), timeout=20)
                return received, refused.close_code

        received, close_code = asyncio.run(spend_then_reopen())

        _assert_refused(received)
        assert close_code == 1011, close_code
        assert peer.paths.qsize() == 1
        assert deployment.forwarded_turns() == 1
        http: Final = limited.request(
            "POST", "/v1/responses", {"model": deployment.name, "input": f"after {uuid.uuid4().hex}"}, key=key
        )
        assert http.status_code == 429, http.text
        assert peer.paths.qsize() == 1


def test_a_socket_the_router_refuses_to_open_gets_the_status_and_message_http_returns(
    limited: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, limited.scenario() as scenario:
        name: Final = scenario.model(
            model=f"openai/{PROVIDER_MODEL}",
            api_base=peer.url,
            tags=[f"granted-{uuid.uuid4().hex}"],
            model_info={"enable_tag_filtering": True},
        )
        key: Final = scenario.key(models=[name])
        foreign: Final = {"x-litellm-tags": f"foreign-{uuid.uuid4().hex}"}
        http: Final = limited.request(
            "POST", "/v1/responses", {"model": name, "input": "tag routing"}, key=key, headers=foreign
        )
        assert http.status_code != 200, http.text
        http_error: Final = object_value(JSON_OBJECT.validate_json(http.content)["error"])

        async def refused_open() -> tuple[dict[str, JsonValue], ...]:
            async with _socket(limited, key, name, foreign) as connection:
                return await _say(connection, name)

        received: Final = asyncio.run(refused_open())

        assert [event.get("type") for event in received] == ["error"], received
        assert received[0]["status"] == http.status_code, (received, http.text)
        assert object_value(received[0]["error"])["message"] == string_value(http_error["message"]), (
            received,
            http.text,
        )
        assert peer.paths.qsize() == 0


def test_each_turn_counts_as_one_request_toward_the_deployment_rpm(limited: Gateway, cert: tuple[Path, Path]) -> None:
    with responses_peer(cert) as peer, limited.scenario() as scenario:
        deployment: Final = _native(scenario, peer, rpm=1)
        key: Final = scenario.key(models=[deployment.name])
        _fresh_minute()

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(limited, key, deployment.name) as connection:
                return (await _say(connection, deployment.name), await _say(connection, deployment.name))

        first, second = asyncio.run(session())

        completed(first)
        _assert_refused(second)
        assert deployment.forwarded_turns() == 1


def test_a_deployment_tpm_refuses_a_turn_only_once_earlier_turns_used_it_up(
    limited: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, limited.scenario() as scenario:
        deployment: Final = _native(scenario, peer, tpm=TURN_TOKENS + 2)
        key: Final = scenario.key(models=[deployment.name])

        first, second, third, counted_after_first = asyncio.run(_turns_against_tpm(limited, key, deployment))

        completed(first)
        completed(second)
        _assert_refused(third)
        assert counted_after_first == TURN_TOKENS
        assert deployment.forwarded_turns() == 2


def test_a_tpm_limit_lowered_while_the_socket_is_open_applies_to_its_next_turn(
    limited: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, limited.scenario() as scenario:
        deployment: Final = _native(scenario, peer, tpm=100 * TURN_TOKENS)
        key: Final = scenario.key(models=[deployment.name])
        minute: Final = _fresh_minute()

        def lower_tpm() -> None:
            limited.request(
                "PATCH", f"/model/{deployment.identity}/update", {"litellm_params": {"tpm": TURN_TOKENS - 2}}
            ).raise_for_status()
            eventually(lambda: _configured_tpm(limited, deployment), lambda tpm: tpm == TURN_TOKENS - 2, seconds=15)

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(limited, key, deployment.name) as connection:
                first: Final = await _say(connection, deployment.name)
                await _until(partial(_tokens_counted, deployment.identity, minute), lambda tokens: tokens > 0)
                await asyncio.to_thread(lower_tpm)
                return (first, await _say(connection, deployment.name))

        first, second = asyncio.run(session())

        completed(first)
        _assert_refused(second)
        assert deployment.forwarded_turns() == 1


def test_a_socket_on_an_otpm_limited_deployment_reserves_once_and_serves_every_turn(
    limited: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, limited.scenario() as scenario:
        deployment: Final = _native(scenario, peer, otpm=10_000)
        key: Final = scenario.key(models=[deployment.name])
        _fresh_minute()

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(limited, key, deployment.name) as connection:
                return tuple([await _say(connection, deployment.name) for _ in range(5)])

        turns: Final = asyncio.run(session())

        assert [completed(received)["status"] for received in turns] == ["completed"] * 5
        assert deployment.forwarded_turns() == 5


def test_usage_based_routing_refuses_a_turn_only_once_earlier_turns_used_up_the_deployment_tpm(
    usage_routed: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, usage_routed.scenario() as scenario:
        deployment: Final = _native(scenario, peer, tpm=TURN_TOKENS + 2)
        key: Final = scenario.key(models=[deployment.name])

        first, second, third, counted_after_first = asyncio.run(_turns_against_tpm(usage_routed, key, deployment))

        completed(first)
        completed(second)
        _assert_refused(third)
        assert counted_after_first == TURN_TOKENS
        assert deployment.forwarded_turns() == 2


def _assert_http_chat_counted_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        identity: Final = f"ws-admission-{uuid.uuid4().hex}"
        name: Final = scenario.model(model_info={"id": identity}, tpm=1_000_000)
        key: Final = scenario.key(models=[name])
        minute: Final = _fresh_minute()
        answered: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": name, "messages": [{"role": "user", "content": f"http turn {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert answered.status_code == 200, answered.text
        tokens: Final = object_value(JSON_OBJECT.validate_json(answered.content)["usage"])["total_tokens"]
        assert isinstance(tokens, int) and tokens > 0, answered.text

        assert _settled_count(partial(_tokens_counted, identity, minute)) == tokens


def test_an_http_call_counts_its_tokens_once_toward_the_deployment_tpm_under_model_rate_limits(
    limited: Gateway,
) -> None:
    _assert_http_chat_counted_once(limited)


def test_an_http_call_counts_its_tokens_once_toward_the_deployment_tpm_under_usage_based_routing(
    usage_routed: Gateway,
) -> None:
    _assert_http_chat_counted_once(usage_routed)


def test_usage_based_routing_serves_every_turn_while_the_deployment_has_headroom(
    usage_routed: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, usage_routed.scenario() as scenario:
        deployment: Final = _native(scenario, peer, tpm=1_000_000, rpm=1_000)
        key: Final = scenario.key(models=[deployment.name])

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(usage_routed, key, deployment.name) as connection:
                return tuple([await _say(connection, deployment.name) for _ in range(3)])

        turns: Final = asyncio.run(session())

        assert [completed(received)["status"] for received in turns] == ["completed"] * 3
        assert deployment.forwarded_turns() == 3


def test_a_frame_over_the_project_input_quota_gets_a_429_error_frame_and_never_reaches_the_provider(
    limited: Gateway,
) -> None:
    with limited.scenario() as scenario:
        deployment: Final = _managed(scenario)
        team: Final = scenario.team()
        key: Final = scenario.key(
            models=[deployment.name],
            team_id=team,
            project_id=scenario.project(team, model_itpm_limit={deployment.name: 100}),
        )

        async def session() -> tuple[tuple[dict[str, JsonValue], ...], ...]:
            async with _socket(limited, key, deployment.name) as connection:
                return (
                    await _say(connection, deployment.name, "short"),
                    await _say(connection, deployment.name, "long " * 400),
                    await _say(connection, deployment.name, "long " * 400),
                )

        first, second, third = asyncio.run(session())

        completed(first)
        _assert_refused(third)
        answered: Final = sum(received[-1].get("type") == "response.completed" for received in (first, second, third))
        assert deployment.forwarded_turns() == answered
