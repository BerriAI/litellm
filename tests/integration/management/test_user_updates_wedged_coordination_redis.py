import os
import signal
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psutil
import psycopg
import pytest
from psycopg import sql
from pydantic import JsonValue
from redis import Redis
from redis.client import PubSub

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.redis_process import owned_redis

_USERS: Final = 60
_BURST: Final = 30
_HANDLER_BUDGET_SECONDS: Final = 0.75
_BULK_BUDGET_SECONDS: Final = 2.0
_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"


def _timed_post(candidate: Gateway, path: str, body: Mapping[str, JsonValue], timeout: float = 15) -> float:
    started: Final = time.monotonic()
    response: Final = candidate.client.request(
        "POST",
        path,
        json=body,
        headers={"Authorization": f"Bearer {candidate.key}"},
        timeout=timeout,
    )
    elapsed: Final = time.monotonic() - started
    assert response.status_code == 200, f"POST {path}: {response.status_code} {response.text} after {elapsed:.3f}s"
    return elapsed


def _received(pubsub: PubSub) -> tuple[dict[str, JsonValue], ...]:
    messages: list[dict[str, JsonValue]] = []
    while True:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0)
        if message is None:
            return tuple(messages)
        data = message.get("data")
        if isinstance(data, (bytes, str)):
            messages.append(JSON_OBJECT.validate_json(data))


def _worker_pid(port: int) -> int:
    for process in psutil.process_iter():
        parent = process.parent()
        if parent is None:
            continue
        try:
            cmdline = parent.cmdline()
            own_cmdline = process.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if (
            "integration._support.proxy" in cmdline
            and "--port" in cmdline
            and str(port) in cmdline
            and not any("prisma" in part for part in own_cmdline)
        ):
            return process.pid
    raise AssertionError(f"no uvicorn worker found under the owned proxy on port {port}")


def _burst_call(
    index: int, users: tuple[str, ...], key: str, team_id: str, customer_id: str
) -> tuple[str, dict[str, JsonValue]]:
    match index % 5:
        case 0:
            return "/user/update", {"user_id": users[index], "max_budget": 200.0 + index}
        case 1:
            return "/user/update", {"user_id": users[index], "tpm_limit": 1000 + index}
        case 2:
            return "/key/update", {"key": key, "max_budget": 7.0 + index}
        case 3:
            return "/team/update", {"team_id": team_id, "max_budget": 7.0 + index}
        case _:
            return "/customer/update", {"user_id": customer_id, "max_budget": 7.0 + index}


@pytest.mark.timeout(240)
@pytest.mark.covers(
    "mgmt.user.update.budget_change_returns_promptly_with_wedged_coordination_redis",
    "mgmt.user.bulk_update.budget_change_returns_promptly_with_wedged_coordination_redis",
    "mgmt.customer.update.budget_change_returns_promptly_with_wedged_coordination_redis",
    "mgmt.key.reset_spend.returns_promptly_with_wedged_coordination_redis",
    "mgmt.auth_cache_invalidation.publish_parked_by_short_redis_wedge_lands_after_recovery",
    "mgmt.auth_cache_invalidation.burst_with_worker_kill_keeps_serving_while_redis_wedged",
)
def test_user_budget_updates_return_promptly_while_coordination_redis_is_wedged(
    gateway: Gateway, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    original: Final = os.environ["DATABASE_URL"]
    identity: Final = "integration_wedged_redis_" + uuid.uuid4().hex
    parsed: Final = urlsplit(original)
    database_url: Final = urlunsplit((parsed.scheme, parsed.netloc, "/" + identity, "", ""))
    timings: dict[str, float] = {}
    with psycopg.connect(original, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(identity)))
        try:
            results_dir: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path)))
            prior_logs: Final = frozenset(results_dir.glob("owned-proxy-*.log"))
            with (
                owned_redis(tmp_path) as coordination,
                owned_proxy(
                    gateway,
                    tmp_path,
                    {
                        "DATABASE_URL": database_url,
                        "REDIS_HOST": coordination.host,
                        "REDIS_PORT": str(coordination.port),
                    },
                    config=Path("tests/integration/coordination_redis_proxy_config.yaml"),
                    workers=2,
                ) as candidate,
                Redis(host=coordination.host, port=coordination.port, socket_timeout=1) as subscriber_client,
            ):
                pubsub: Final = subscriber_client.pubsub()
                pubsub.subscribe(_CHANNEL)
                received: list[dict[str, JsonValue]] = []

                def drained() -> tuple[dict[str, JsonValue], ...]:
                    received.extend(_received(pubsub))
                    return tuple(received)

                eventually(
                    lambda: subscriber_client.pubsub_numsub(_CHANNEL)[0][1],
                    lambda count: count >= 3,
                    seconds=15,
                )
                users: Final = tuple(f"{identity}_u{index}" for index in range(_USERS))
                for user_id in users:
                    candidate.post("/user/new", {"user_id": user_id, "auto_create_key": False, "max_budget": 10.0})
                key: Final = string_value(
                    candidate.post("/key/generate", {"user_id": users[0], "max_budget": 5.0})["key"]
                )
                team_id: Final = string_value(
                    candidate.post("/team/new", {"team_alias": identity, "max_budget": 5.0})["team_id"]
                )
                customer_id: Final = identity + "_cust"
                candidate.post("/customer/new", {"user_id": customer_id, "max_budget": 5.0})
                drained()
                timings["h1_healthy"] = _timed_post(
                    candidate, "/user/update", {"user_id": users[0], "max_budget": 11.0}
                )
                assert timings["h1_healthy"] < _HANDLER_BUDGET_SECONDS, (
                    f"healthy /user/update took {timings['h1_healthy']:.3f}s"
                )
                eventually(
                    drained,
                    lambda messages: any(message.get("cache_key") == users[0] for message in messages),
                    seconds=10,
                )
                timings["h2_healthy_control"] = _timed_post(
                    candidate, "/user/update", {"user_id": users[0], "tpm_limit": 1000}
                )
                assert timings["h2_healthy_control"] < _HANDLER_BUDGET_SECONDS, (
                    f"healthy control update took {timings['h2_healthy_control']:.3f}s"
                )
                coordination.signal(signal.SIGSTOP)
                try:
                    timings["s2_wedged_control"] = _timed_post(
                        candidate, "/user/update", {"user_id": users[0], "tpm_limit": 1000}
                    )
                    assert timings["s2_wedged_control"] < _HANDLER_BUDGET_SECONDS, (
                        f"control update without a cache-relevant field took {timings['s2_wedged_control']:.3f}s"
                    )
                    timings["s1_user_update"] = _timed_post(
                        candidate, "/user/update", {"user_id": users[0], "max_budget": 98.0}
                    )
                    assert timings["s1_user_update"] < _HANDLER_BUDGET_SECONDS, (
                        f"/user/update with max_budget took {timings['s1_user_update']:.3f}s "
                        "with a wedged coordination Redis"
                    )
                    timings["s3_bulk_update"] = _timed_post(
                        candidate, "/user/bulk_update", {"all_users": True, "user_updates": {"max_budget": 79.0}}
                    )
                    assert timings["s3_bulk_update"] < _BULK_BUDGET_SECONDS, (
                        f"/user/bulk_update over {_USERS} users took {timings['s3_bulk_update']:.3f}s "
                        "with a wedged coordination Redis"
                    )
                    timings["s4_key_update"] = _timed_post(
                        candidate, "/key/update", {"key": key, "max_budget": 6.0}, timeout=60
                    )
                    assert timings["s4_key_update"] < 30, (
                        f"/key/update hung for {timings['s4_key_update']:.3f}s with a wedged coordination Redis"
                    )
                    timings["s5_team_update"] = _timed_post(
                        candidate, "/team/update", {"team_id": team_id, "max_budget": 6.0}, timeout=60
                    )
                    assert timings["s5_team_update"] < 30, (
                        f"/team/update hung for {timings['s5_team_update']:.3f}s with a wedged coordination Redis"
                    )
                    timings["s6_customer_update"] = _timed_post(
                        candidate, "/customer/update", {"user_id": customer_id, "max_budget": 6.0}
                    )
                    assert timings["s6_customer_update"] < _HANDLER_BUDGET_SECONDS, (
                        f"/customer/update took {timings['s6_customer_update']:.3f}s with a wedged coordination Redis"
                    )
                    timings["s7_reset_spend"] = _timed_post(candidate, f"/key/{key}/reset_spend", {"reset_to": 0})
                    assert timings["s7_reset_spend"] < _HANDLER_BUDGET_SECONDS, (
                        f"/key/<key>/reset_spend took {timings['s7_reset_spend']:.3f}s with a wedged coordination Redis"
                    )
                    missing_started: Final = time.monotonic()
                    missing: Final = candidate.request(
                        "POST", "/user/update", {"user_id": users[0], "max_budget": "not-a-number"}
                    )
                    timings["s8_invalid_body"] = time.monotonic() - missing_started
                    assert missing.status_code // 100 == 4, (
                        f"/user/update with an invalid body returned {missing.status_code} "
                        f"in {timings['s8_invalid_body']:.3f}s"
                    )
                    assert timings["s8_invalid_body"] < _HANDLER_BUDGET_SECONDS, (
                        f"/user/update with an invalid body took {timings['s8_invalid_body']:.3f}s"
                    )

                    def burst_request(path: str, body: Mapping[str, JsonValue]) -> tuple[object, float]:
                        started: Final = time.monotonic()
                        try:
                            response: Final = candidate.client.request(
                                "POST",
                                path,
                                json=body,
                                headers={"Authorization": f"Bearer {candidate.key}"},
                                timeout=60,
                            )
                            return response.status_code, time.monotonic() - started
                        except Exception as error:  # noqa: BLE001  # the killed worker drops in-flight requests
                            return error, time.monotonic() - started

                    port: Final = candidate.client.base_url.port
                    assert port is not None, f"owned proxy client has no port: {candidate.client.base_url}"
                    with ThreadPoolExecutor(_BURST) as pool:
                        futures: Final = [
                            pool.submit(
                                burst_request,
                                *_burst_call(i, users, key, team_id, customer_id),
                            )
                            for i in range(_BURST)
                        ]
                        os.kill(_worker_pid(port), signal.SIGKILL)
                        results: Final = [future.result() for future in futures]
                    responses: Final = [(status, elapsed) for status, elapsed in results if isinstance(status, int)]
                    failures: Final = [status for status, _elapsed in responses if status != 200]
                    assert not failures, f"burst responses that were not 200: {failures}"
                    transport_errors: Final = [status for status, _elapsed in results if not isinstance(status, int)]
                    assert len(transport_errors) <= 3, (
                        f"{len(transport_errors)} requests raised transport errors: {transport_errors!r}"
                    )
                    elapsed_sorted: Final = sorted(
                        elapsed for i, (status, elapsed) in enumerate(results) if i % 5 in (0, 1, 4) and status == 200
                    )
                    timings["c1_burst_p95"] = elapsed_sorted[int(len(elapsed_sorted) * 0.95) - 1]
                    assert timings["c1_burst_p95"] < _HANDLER_BUDGET_SECONDS, (
                        f"burst p95 {timings['c1_burst_p95']:.3f}s"
                    )
                    eventually(
                        lambda: candidate.request("GET", "/health/liveliness").status_code,
                        lambda status: status == 200,
                        seconds=15,
                    )
                    timings["c1_survivor"] = _timed_post(
                        candidate, "/user/update", {"user_id": users[0], "tpm_limit": 2000}
                    )
                    assert timings["c1_survivor"] < _HANDLER_BUDGET_SECONDS, (
                        f"control update on the surviving worker took {timings['c1_survivor']:.3f}s"
                    )
                finally:
                    coordination.signal(signal.SIGCONT)
                wedged_keys: Final = {users[i] for i in range(_BURST) if i % 5 == 0 and i != 0} | {f"team_id:{team_id}"}

                def proxy_log() -> str:
                    return "".join(
                        path.read_text() for path in results_dir.glob("owned-proxy-*.log") if path not in prior_logs
                    )

                team_wedged_key: Final = f"team_id:{team_id}"
                eventually(
                    proxy_log,
                    lambda text: (
                        all(
                            f"publish for {wedged_key} failed" in text
                            for wedged_key in wedged_keys
                            if wedged_key != team_wedged_key
                        )
                        and (
                            f"publish for {team_wedged_key} failed" in text
                            or f"internal usage cache entry {team_wedged_key}" in text
                        )
                    ),
                    seconds=45,
                )
                marker: Final = len(received)
                drained()
                recovered_keys: Final = {str(message.get("cache_key")) for message in received[marker:]}
                assert recovered_keys.isdisjoint(wedged_keys), (
                    f"wedged publishes unexpectedly landed after recovery: {sorted(recovered_keys & wedged_keys)}"
                )
                coordination.signal(signal.SIGSTOP)
                try:
                    timings["r1b_short_wedge_a"] = _timed_post(
                        candidate, "/user/update", {"user_id": users[4], "max_budget": 15.0}
                    )
                    assert timings["r1b_short_wedge_a"] < _HANDLER_BUDGET_SECONDS, (
                        f"/user/update inside a short wedge took {timings['r1b_short_wedge_a']:.3f}s"
                    )
                    timings["r1b_short_wedge_b"] = _timed_post(
                        candidate, "/user/update", {"user_id": users[5], "max_budget": 16.0}
                    )
                    assert timings["r1b_short_wedge_b"] < _HANDLER_BUDGET_SECONDS, (
                        f"/user/update inside a short wedge took {timings['r1b_short_wedge_b']:.3f}s"
                    )
                finally:
                    coordination.signal(signal.SIGCONT)
                eventually(
                    drained,
                    lambda messages: {str(message.get("cache_key")) for message in messages} >= {users[4], users[5]},
                    seconds=10,
                )
                timings["r2_resumed"] = _timed_post(
                    candidate, "/user/update", {"user_id": users[1], "max_budget": 12.0}
                )
                assert timings["r2_resumed"] < _HANDLER_BUDGET_SECONDS, (
                    f"post-recovery /user/update took {timings['r2_resumed']:.3f}s"
                )
                eventually(
                    drained,
                    lambda messages: any(message.get("cache_key") == users[1] for message in messages),
                    seconds=10,
                )
                coordination.stop()
                timings["f1_refused"] = _timed_post(
                    candidate, "/user/update", {"user_id": users[2], "max_budget": 13.0}
                )
                assert timings["f1_refused"] < _HANDLER_BUDGET_SECONDS, (
                    f"/user/update with refused coordination Redis took {timings['f1_refused']:.3f}s"
                )
                coordination.start()
                restarted_pubsub: Final = subscriber_client.pubsub()
                restarted_pubsub.subscribe(_CHANNEL)
                restarted_received: list[dict[str, JsonValue]] = []

                def drained_after_restart() -> tuple[dict[str, JsonValue], ...]:
                    restarted_received.extend(_received(restarted_pubsub))
                    return tuple(restarted_received)

                eventually(
                    lambda: subscriber_client.pubsub_numsub(_CHANNEL)[0][1],
                    lambda count: count >= 3,
                    seconds=30,
                )
                timings["f2_restarted"] = _timed_post(
                    candidate, "/user/update", {"user_id": users[3], "max_budget": 14.0}
                )
                assert timings["f2_restarted"] < _HANDLER_BUDGET_SECONDS, (
                    f"/user/update after Redis restart took {timings['f2_restarted']:.3f}s"
                )
                eventually(
                    drained_after_restart,
                    lambda messages: any(message.get("cache_key") == users[3] for message in messages),
                    seconds=10,
                )
                info_last: Final = object_value(candidate.get("/user/info", {"user_id": users[-1]})["user_info"])
                assert info_last["max_budget"] == 79.0, info_last
                info_user3: Final = object_value(candidate.get("/user/info", {"user_id": users[3]})["user_info"])
                assert info_user3["max_budget"] == 14.0, info_user3
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(identity)))
    record_property("cell_elapsed_seconds", timings)
