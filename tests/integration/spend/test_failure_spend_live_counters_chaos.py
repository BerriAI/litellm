import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.timing import assert_not_delayed_by, timed
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.spend._partial_stream import (
    DELIVERED_PIECES,
    assert_budget_refusal,
    broken_stream,
    chat,
    chats_sent,
    failure_spend,
    peer,
    priced_model,
    recovered_cost,
)
from pydantic import JsonValue
from redis import Redis

from litellm.constants import REDIS_SOCKET_TIMEOUT

_BURST: Final = 12
_SPREAD: Final = 6
_ATTACK_MARKER: Final = "synthetic-attack-marker"
_POLICY_ID: Final = "synthetic-priced-policy"
_POLICY_APPLY_PATH: Final = f"/guardrail/{_POLICY_ID}/version/DRAFT/apply"


def _proxy_config(directory: Path, name: str, sections: dict[str, JsonValue]) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump({**base, **sections}))
    return path


def _redis_counter_config(directory: Path) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    return _proxy_config(
        directory,
        "redis-counters",
        {
            "router_settings": {
                **base["router_settings"],
                "redis_host": "os.environ/REDIS_HOST",
                "redis_port": "os.environ/REDIS_PORT",
            }
        },
    )


def _spread_requests(gateway: Gateway, model: str, key: str) -> tuple[httpx.Response, ...]:
    return tuple(
        gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": f"spread {uuid.uuid4().hex}"}]},
            key=key,
            headers={"connection": "close"},
        )
        for _ in range(_SPREAD)
    )


@pytest.mark.timeout(600)
def test_broken_streams_through_a_redis_stall_log_once_and_the_budget_holds_on_every_worker(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    with (
        owned_redis(tmp_path) as counters,
        wire_server(peer(DELIVERED_PIECES)) as wire,
        wire_server(peer(DELIVERED_PIECES, gate=release)) as held_wire,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"REDIS_HOST": counters.host, "REDIS_PORT": str(counters.port)},
            config=_redis_counter_config(tmp_path),
            workers=2,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        scenario.cleanups.callback(release.set)
        candidate: Final = owned.gateway
        model: Final = priced_model(scenario, wire)
        stalled_model: Final = priced_model(scenario, held_wire)
        cost: Final = recovered_cost(candidate, scenario, model)
        control: Final = timed(lambda: broken_stream(candidate, model, scenario.key(models=[model])))
        burst_key: Final = scenario.key(models=[stalled_model])
        chats_sent(held_wire)

        with ThreadPoolExecutor(max_workers=_BURST) as pool:
            pending: Final = tuple(
                pool.submit(broken_stream, candidate, stalled_model, burst_key) for _ in range(_BURST)
            )
            eventually(held_wire.received.qsize, lambda held: held >= _BURST, seconds=30)
            with counters.paused():
                release.set()
                drained: Final = timed(lambda: tuple(future.result() for future in pending))

        assert_not_delayed_by((drained.seconds,), control=(control.seconds,), injected_delay=REDIS_SOCKET_TIMEOUT)
        assert failure_spend(candidate, burst_key, rows=_BURST) == pytest.approx((cost,) * _BURST)
        spent: Final = cost * _BURST
        with Redis(host=counters.host, port=counters.port) as live_counters:
            landed: Final = eventually(
                lambda: float(live_counters.get(f"spend:key:{sha256(burst_key.encode()).hexdigest()}") or 0),
                lambda counted: counted >= spent - 1e-9,
                seconds=30,
                return_last_on_timeout=True,
            )
        assert spent - 1e-9 <= landed <= 2 * spent + 1e-9, (
            f"one broken stream costs {cost}. The cold counter is seeded from the database, and a spend flush"
            " during the stall can land these rows there first, so up to twice their cost is expected, as for"
            " completed streams held past a flush"
        )

        budgeted: Final = scenario.key(models=[model], max_budget=cost / 2)
        broken_stream(candidate, model, budgeted)
        before_kill: Final = _spread_requests(candidate, model, budgeted)
        owned.kill_worker(min(owned.worker_pids()))
        after_kill: Final = _spread_requests(candidate, model, budgeted)

        for refused in (*before_kill, *after_kill):
            assert_budget_refusal(refused)


def _policy(request: Request) -> Reply:
    if request.method != "POST" or request.target != _POLICY_APPLY_PATH:
        return Reply(status=404)
    if _ATTACK_MARKER.encode() not in request.body:
        return Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())
    return Reply(
        body=json.dumps(
            {
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": "blocked by synthetic policy"}],
                "assessments": [
                    {"contentPolicy": {"filters": [{"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}]}}
                ],
                "usage": {"contentPolicyUnits": 1},
            }
        ).encode()
    )


def _policy_config(directory: Path, policy: Wire) -> Path:
    return _proxy_config(
        directory,
        "priced-policy",
        {
            "guardrails": [
                {
                    "guardrail_name": "priced-bedrock-policy",
                    "litellm_params": {
                        "guardrail": "bedrock",
                        "mode": "pre_call",
                        "default_on": True,
                        "guardrailIdentifier": _POLICY_ID,
                        "guardrailVersion": "DRAFT",
                        "aws_region_name": "us-east-1",
                        "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
                        "aws_secret_access_key": "synthetic-guardrail-secret",
                        "aws_bedrock_runtime_endpoint": policy.url,
                    },
                }
            ]
        },
    )


def _attack(gateway: Gateway, model: str, key: str) -> httpx.Response:
    blocked: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"{_ATTACK_MARKER} {uuid.uuid4().hex}"}]},
        key=key,
    )
    assert blocked.status_code == 400, blocked.text
    return blocked


@pytest.mark.timeout(480)
def test_guardrail_cost_on_blocked_requests_counts_toward_the_budget(gateway: Gateway, tmp_path: Path) -> None:
    with (
        wire_server(_policy) as policy,
        wire_server(peer(None)) as wire,
        owned_proxy_process(gateway, tmp_path, {}, config=_policy_config(tmp_path, policy), workers=2) as owned,
        owned.gateway.scenario() as scenario,
    ):
        candidate: Final = owned.gateway
        model: Final = priced_model(scenario, wire)
        meter: Final = scenario.key(models=[model], max_budget=100)
        _attack(candidate, model, meter)
        (cost,) = failure_spend(candidate, meter)
        assert cost > 0
        key: Final = scenario.key(models=[model], max_budget=cost * 1.5)
        chats_sent(wire)

        _attack(candidate, model, key)
        _attack(candidate, model, key)
        after: Final = chat(candidate, model, key)

        assert_budget_refusal(after)
        assert chats_sent(wire) == 0


@pytest.mark.timeout(480)
def test_broken_streams_leave_the_budget_alone_while_spend_updates_are_disabled(
    gateway: Gateway, tmp_path: Path
) -> None:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = _proxy_config(
        tmp_path, "spend-updates-off", {"general_settings": {**base["general_settings"], "disable_spend_updates": True}}
    )
    with (
        wire_server(peer(DELIVERED_PIECES)) as wire,
        gateway.scenario() as metering,
        owned_proxy_process(gateway, tmp_path, {}, config=config) as owned,
        owned.gateway.scenario() as scenario,
    ):
        cost: Final = recovered_cost(gateway, metering, priced_model(metering, wire))
        candidate: Final = owned.gateway
        model: Final = priced_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=cost / 2)

        broken_stream(candidate, model, key)
        after: Final = chat(candidate, model, key)

        assert after.status_code == 200, after.text
