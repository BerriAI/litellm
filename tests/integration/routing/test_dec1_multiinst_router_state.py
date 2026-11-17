"""Router state that proxy instances share through Redis: cooldowns, allowed fails, usage counters, latency.

Two instances run one config on one Redis (the proxy attaches its coordination Redis to the router when
`router_settings` names none, `proxy_server.py:7901`). The Rust Redis stores in
`litellm/rust/crates/redis-store/src/` now own these keys and must write the bytes `main`'s Python wrote,
because Python and Rust pods share them during a rolling deploy:

`deployment:<id>:cooldown` holds a cooldown as JSON (`exception_received`, `status_code`, `timestamp`,
`cooldown_time`) with a TTL of the cooldown length; it is also accepted when another writer stored it with
`str(value)`. `deployment:<id>:allowed_fails` (or `...:allowed_fails:<suffix>` for a deployment level
`allowed_fails`, suffix `generic`) is a fleet-wide `INCRBYFLOAT` counter whose TTL is the cooldown time.
`<id>:<litellm_params.model>:rpm:<HH-MM>` and `...:tpm:<HH-MM>` are the usage-based-routing-v2 counters for
the current UTC minute with a 60 second TTL. `<model group>_map` is the latency-based-routing map, a Python
literal dict keyed by deployment id with a `latency_normalized` list, kept for `routing_strategy_args.ttl`
(default 3600) seconds.

Cooldowns are re-read from Redis for a key that missed at most once a second
(`DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS`), usage counters at most every
`default_redis_batch_cache_expiry` seconds (default 10). The failure-rate rule (five straight 5xx in a minute)
counts in each instance's memory only, so it is not shared.

Existing files already pin the basic cooldown key and a peer honouring it
(`test_dec1_routing_redis_shared_cooldown.py`) and the MGET order of usage reads
(`test_usage_based_routing_redis_reads.py`); these tests cover the rest.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeVar

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, held, object_value
from tests.integration.routing.dec1_multiinst_support import Fleet, decoded, fleet, served, utc_minute
from tests.integration.routing.dec1_routing_support import NO_DEPLOYMENTS, chat, deployment, error_message, statuses

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

COOLDOWN_SECONDS: Final = 30
SHORT_COOLDOWN_SECONDS: Final = 10  # longer than the eventually + read throttle before the peer's first request
COOLDOWN_READ_SECONDS: Final = 3  # DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS (1.0) plus a margin
T = TypeVar("T")
TOKENS_PER_ANSWER: Final = 8  # the scripted upstream's usage.total_tokens


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("state-expiry", "status-429-state-expiry", api_base, weight=1, cooldown_time=SHORT_COOLDOWN_SECONDS),
        deployment("state-expiry", "ok-state-expiry", api_base),
        deployment("state-foreign", "ok-state-foreign-target", api_base, weight=1),
        deployment("state-foreign", "ok-state-foreign-sibling", api_base),
        deployment("state-literal", "ok-state-literal-target", api_base, weight=1),
        deployment("state-literal", "ok-state-literal-sibling", api_base),
        deployment("state-fast-read", "script-state-fast-read", api_base, weight=1),
        deployment("state-fast-read", "ok-state-fast-read", api_base),
        deployment("state-rate", "status-500-state-rate", api_base, weight=1),
        deployment("state-rate", "ok-state-rate", api_base),
        deployment("state-embed", "status-429-state-embed", api_base, weight=1),
        deployment("state-embed", "ok-state-embed", api_base),
    )


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-state"),
        models=_models,
        router_settings={"num_retries": 0, "cooldown_time": COOLDOWN_SECONDS},
    ) as value:
        yield value


def _cooldown(pair: Fleet, deployment_id: str) -> JsonValue:
    return pair.raw(f"deployment:{deployment_id}:cooldown")


def test_a_peer_routes_to_a_deployment_again_once_its_shared_cooldown_ended(pair: Fleet) -> None:
    """The weighted deployment of `state-expiry` sets `cooldown_time: 10` and answers 429 through the first instance.
    (With 3 seconds the cooldown had already ended when the peer's first request arrived after the 3 second read
    wait, so the peer correctly reached the recovered deployment.)

    The second instance must first serve the group from the healthy sibling, and once
    `deployment:dec1-status-429-state-expiry:cooldown` has expired in Redis (within 20 seconds) the second instance
    must send the weighted deployment traffic again within 10 seconds (a 429 from it). A peer that kept its own
    memory copy past the cooldown's end would keep a recovered deployment out of rotation.
    """
    assert chat(pair.first, "state-expiry").status_code == 429
    eventually(lambda: _cooldown(pair, "dec1-status-429-state-expiry"), lambda raw: raw is not None, seconds=5)
    time.sleep(COOLDOWN_READ_SECONDS)
    assert chat(pair.second, "state-expiry").status_code == 200
    eventually(lambda: _cooldown(pair, "dec1-status-429-state-expiry"), lambda raw: raw is None, seconds=20)
    eventually(lambda: chat(pair.second, "state-expiry").status_code, lambda status: status == 429, seconds=10)


def test_a_cooldown_written_into_redis_by_another_pod_is_honoured(pair: Fleet) -> None:
    """The test itself plays a pod on the other stack: it sets `deployment:dec1-ok-state-foreign-target:cooldown`
    to `json.dumps({"exception_received": "scripted", "status_code": "429", "timestamp": <now>,
    "cooldown_time": 30})` with `EX 30`.

    After the 1 second read throttle, five requests to `state-foreign` through the second instance must all be
    served by `ok-state-foreign-sibling`, although the target deployment never failed. This pins the value format a
    Rust reader must accept from a Python writer.
    """
    value: Final = {
        "exception_received": "scripted",
        "status_code": "429",
        "timestamp": time.time(),
        "cooldown_time": COOLDOWN_SECONDS,
    }
    with pair.store() as client:
        client.set("deployment:dec1-ok-state-foreign-target:cooldown", json.dumps(value), ex=COOLDOWN_SECONDS)
    time.sleep(COOLDOWN_READ_SECONDS)
    names: Final = tuple(served(chat(pair.second, "state-foreign")) for _ in range(5))
    assert names == ("ok-state-foreign-sibling",) * 5
    assert pair.upstream.count("ok-state-foreign-target") == 0


def test_a_cooldown_stored_as_a_python_literal_is_also_honoured(pair: Fleet) -> None:
    """Python's sync `DualCache.set_cache` stores `str(value)`, a Python literal with single quotes and `True`, not
    JSON. The test stores such a value for `deployment:dec1-ok-state-literal-target:cooldown` with `EX 30`.

    Five requests to `state-literal` through the first instance must all be served by `ok-state-literal-sibling`.
    UNVERIFIED: that the Rust cooldown reader decodes the literal form (`RouterCooldowns::put_str` says the store
    writes it, `redis-store/src/router_cooldowns/mod.rs:62`; a reader that only parses JSON would ignore it).
    """
    value: Final = {
        "exception_received": "scripted",
        "status_code": "429",
        "timestamp": time.time(),
        "cooldown_time": COOLDOWN_SECONDS,
    }
    with pair.store() as client:
        client.set("deployment:dec1-ok-state-literal-target:cooldown", str(value), ex=COOLDOWN_SECONDS)
    time.sleep(COOLDOWN_READ_SECONDS)
    names: Final = tuple(served(chat(pair.first, "state-literal")) for _ in range(5))
    assert names == ("ok-state-literal-sibling",) * 5


def test_a_peer_sees_a_new_cooldown_within_seconds_even_right_after_it_looked(pair: Fleet) -> None:
    """The weighted deployment of `state-fast-read` answers 200 until a 429 is queued for it. The second instance
    first serves the group from it once, so it has just looked the cooldown key up and missed. Then one 429 is
    queued and a request through the first instance receives it.

    Within 3 seconds of the cooldown key appearing, five requests through the second instance must all be served by
    `ok-state-fast-read`. The cooldown family re-reads a missed key once a second
    (`DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS`, `litellm/constants.py:162`), not every 10 seconds like the
    other batch-read families; the older shared cooldown file waits 11 seconds and would not notice a regression.
    """
    assert served(chat(pair.second, "state-fast-read")) == "script-state-fast-read"
    pair.upstream.script("script-state-fast-read", 429)
    assert chat(pair.first, "state-fast-read").status_code == 429
    eventually(lambda: _cooldown(pair, "dec1-script-state-fast-read"), lambda raw: raw is not None, seconds=5)
    eventually(
        lambda: statuses(tuple(chat(pair.second, "state-fast-read") for _ in range(5))),
        lambda found: found == (200,) * 5 and pair.upstream.count("ok-state-fast-read") >= 5,
        seconds=COOLDOWN_READ_SECONDS,
    )


def test_the_failure_rate_rule_counts_each_instance_on_its_own(pair: Fleet) -> None:
    """The weighted deployment of `state-rate` always answers 500. Three requests go through each instance.

    All six must return 500 and Redis must hold no `deployment:dec1-status-500-state-rate:cooldown` key: each
    instance has seen three failures, under the five-request minimum of the failure-rate rule, and those per-minute
    outcome counts live in memory only. UNVERIFIED: that `count_failure` is memory only in the Rust store as it was
    `local_only=True` on main; see `router-core/src/health.rs:262`.
    """
    results: Final = statuses(tuple(chat(gateway, "state-rate") for gateway in pair.instances for _ in range(3)))
    assert results == (500,) * 6
    held(lambda: _cooldown(pair, "dec1-status-500-state-rate"), lambda raw: raw is None, holding=2)


def test_a_cooldown_started_by_an_embeddings_call_on_one_instance_moves_the_peers_chat_traffic(pair: Fleet) -> None:
    """An embeddings request to `state-embed` through the first instance gets a 429 from the weighted deployment.

    After the read throttle, five chat requests to `state-embed` through the second instance must all return 200
    from `ok-state-embed`. The cooldown key is per deployment id, not per endpoint.
    """
    response: Final = pair.first.request("POST", "/v1/embeddings", {"model": "state-embed", "input": "embed me"})
    assert response.status_code == 429, response.text
    eventually(lambda: _cooldown(pair, "dec1-status-429-state-embed"), lambda raw: raw is not None, seconds=5)
    time.sleep(COOLDOWN_READ_SECONDS)
    assert statuses(tuple(chat(pair.second, "state-embed") for _ in range(5))) == (200,) * 5


def _allowed_fails_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("fails-router", "status-500-fails-router", api_base, weight=1),
        deployment("fails-router", "ok-fails-router", api_base),
        deployment(
            "fails-deployment",
            "status-500-fails-deployment",
            api_base,
            weight=1,
            model_info={"id": "dec1-status-500-fails-deployment", "allowed_fails": 1},
        ),
        deployment("fails-deployment", "ok-fails-deployment", api_base),
    )


@pytest.fixture(scope="module")
def fails_pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-fails"),
        models=_allowed_fails_models,
        router_settings={"num_retries": 0, "cooldown_time": COOLDOWN_SECONDS, "allowed_fails": 1},
    ) as value:
        yield value


def test_one_failure_writes_the_shared_allowed_fails_counter_with_the_cooldown_time_as_ttl(fails_pair: Fleet) -> None:
    """`router_settings.allowed_fails: 1` and `cooldown_time: 30`. One request through the first instance gets a 500
    from the weighted deployment of `fails-router`.

    Redis must then hold `deployment:dec1-status-500-fails-router:allowed_fails` with the value `1` and a TTL
    between 1 and 30 seconds (`fails_window()`, the router's cooldown time), and no cooldown key yet.
    """
    assert chat(fails_pair.first, "fails-router").status_code == 500
    key: Final = "deployment:dec1-status-500-fails-router:allowed_fails"
    raw: Final = eventually(lambda: fails_pair.raw(key), lambda value: value is not None, seconds=5)
    assert float(raw.decode()) == 1
    assert 0 < fails_pair.ttl(key) <= COOLDOWN_SECONDS
    assert fails_pair.raw("deployment:dec1-status-500-fails-router:cooldown") is None


def test_failures_on_two_instances_add_up_against_allowed_fails(fails_pair: Fleet) -> None:
    """With `allowed_fails: 1`, the counter already holds one failure from the first instance. One more request to
    `fails-router`, this time through the second instance, gets a 500.

    The fleet-wide count is now 2, above the allowance, so `deployment:dec1-status-500-fails-router:cooldown` must
    appear in Redis, and requests through either instance must then be served by `ok-fails-router`. A rewrite that
    counted per instance would need two failures on each pod.
    """
    assert chat(fails_pair.second, "fails-router").status_code == 500
    eventually(
        lambda: fails_pair.raw("deployment:dec1-status-500-fails-router:cooldown"),
        lambda raw: raw is not None,
        seconds=5,
    )
    time.sleep(COOLDOWN_READ_SECONDS)
    assert {served(chat(gateway, "fails-router")) for gateway in fails_pair.instances} == {"ok-fails-router"}


def test_a_deployment_level_allowed_fails_counts_under_the_generic_suffix(fails_pair: Fleet) -> None:
    """The weighted deployment of `fails-deployment` sets `model_info.allowed_fails: 1`. One request through the
    first instance gets its 500.

    Redis must hold `deployment:dec1-status-500-fails-deployment:allowed_fails:generic` with the value `1`.
    The suffix comes from `health.rs:420-425`; a rewrite that drops it would share one counter between the
    router-wide and deployment-wide rules.
    """
    assert chat(fails_pair.first, "fails-deployment").status_code == 500
    key: Final = "deployment:dec1-status-500-fails-deployment:allowed_fails:generic"
    raw: Final = eventually(lambda: fails_pair.raw(key), lambda value: value is not None, seconds=5)
    assert float(raw.decode()) == 1


def _usage_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("usage-capped", "ok-usage-capped", api_base, rpm=3),
        deployment("usage-counted", "ok-usage-counted", api_base, rpm=1000),
    )


@pytest.fixture(scope="module")
def usage_pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-usage"),
        models=_usage_models,
        router_settings={"num_retries": 0, "disable_cooldowns": True, "routing_strategy": "usage-based-routing-v2"},
        litellm_settings={"default_redis_batch_cache_expiry": 1},
    ) as value:
        yield value


def _in_one_minute(action: Callable[[], T]) -> tuple[str, T]:
    """Runs `action` until a run starts and ends in the same UTC minute, so per-minute counters are comparable.

    Every run starts at the beginning of a fresh minute when fewer than 15 seconds are left, and a retry always
    waits for the next minute, so a run never inherits counts that an earlier run left in its minute (CI run
    44176682961 retried straight into a minute the first run had already used two requests of)."""
    while True:
        if datetime.now(UTC).second >= 45:
            started = utc_minute()
            eventually(utc_minute, lambda minute, seen=started: minute != seen, seconds=20)
        before = utc_minute()
        result = action()
        after = utc_minute()
        if after == before:
            return (before, result)
        eventually(utc_minute, lambda minute, seen=after: minute != seen, seconds=65)


def test_usage_routing_writes_an_rpm_counter_per_deployment_and_minute(usage_pair: Fleet) -> None:
    """`routing_strategy: usage-based-routing-v2`. Two requests to `usage-counted` go through the first instance and
    one through the second, within one UTC minute.

    Redis must hold `dec1-ok-usage-counted:openai/ok-usage-counted:rpm:<HH-MM>` for that minute with the value 3 and
    a TTL between 1 and 60 seconds: both instances increment the one key. The key layout is
    `{id}:{litellm_params.model}:rpm:{HH-MM}` (`redis-store/src/router_usage/keys.rs`).
    """

    def send() -> None:
        for gateway in (usage_pair.first, usage_pair.first, usage_pair.second):
            assert chat(gateway, "usage-counted").status_code == 200

    minute, _ = _in_one_minute(send)
    key: Final = f"dec1-ok-usage-counted:openai/ok-usage-counted:rpm:{minute}"
    raw: Final = usage_pair.raw(key)
    if raw is None:
        pytest.skip("the UTC minute rolled over between the requests and the read")
    assert float(raw.decode()) >= 3
    assert 0 < usage_pair.ttl(key) <= 60


def test_usage_routing_writes_a_tpm_counter_that_grows_by_the_answers_total_tokens(usage_pair: Fleet) -> None:
    """One request to `usage-counted` through the second instance; the scripted upstream reports
    `usage.total_tokens` 8.

    Within 10 seconds Redis must hold a key matching `dec1-ok-usage-counted:*:tpm:<HH-MM>` whose value is a
    multiple of 8 and at least 8, with a TTL between 1 and 60 seconds. UNVERIFIED: the middle segment, which is
    `hidden_params.litellm_model_name` on the success path (`lowest_tpm_rpm_v2.py:362`) and may differ from the
    `openai/...` spelling the rpm key uses.
    """
    minute: Final = utc_minute()
    assert chat(usage_pair.second, "usage-counted").status_code == 200
    keys: Final = eventually(
        lambda: usage_pair.keys(f"dec1-ok-usage-counted:*:tpm:{minute}"), lambda found: len(found) >= 1, seconds=10
    )
    raw: Final = usage_pair.raw(min(keys))
    assert raw is not None
    assert float(raw.decode()) >= TOKENS_PER_ANSWER
    assert float(raw.decode()) % TOKENS_PER_ANSWER == 0
    assert 0 < usage_pair.ttl(min(keys)) <= 60


def test_rpm_used_on_one_instance_counts_against_the_deployment_on_the_peer(usage_pair: Fleet) -> None:
    """`usage-capped` has one deployment with `rpm: 3` and `default_redis_batch_cache_expiry: 1`. Three requests go
    through the first instance within one UTC minute and return 200.

    A fourth request in the same minute through the second instance, which never served the group, must be refused
    with 429 and a message containing `No deployments available for selected model`, and the upstream must have
    received only three requests. UNVERIFIED: the exact refusal text for a group whose only deployment is over its
    rpm; see `_return_potential_deployments` and the raise near `lowest_tpm_rpm_v2.py:503`.
    """

    def send() -> tuple[int, ...]:
        first_three = statuses(tuple(chat(usage_pair.first, "usage-capped") for _ in range(3)))
        time.sleep(1.5)  # the peer's usage read throttle, `default_redis_batch_cache_expiry: 1`
        return (*first_three, chat(usage_pair.second, "usage-capped").status_code)

    before: Final = usage_pair.upstream.count("ok-usage-capped")
    minute, results = _in_one_minute(send)
    assert results[:3] == (200, 200, 200), (minute, results)
    assert results[3] == 429, (minute, results)
    assert usage_pair.upstream.count("ok-usage-capped") - before == 3
    refused: Final = chat(usage_pair.second, "usage-capped")
    if refused.status_code == 429 and utc_minute() == minute:
        assert NO_DEPLOYMENTS in error_message(refused)


def _latency_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("latency-shared", "ok-latency-fast", api_base),
        deployment("latency-shared", "mid-latency-slow", api_base),
        deployment("latency-ttl", "ok-latency-ttl", api_base),
    )


@pytest.fixture(scope="module")
def latency_pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-latency"),
        models=_latency_models,
        router_settings={
            "num_retries": 0,
            "disable_cooldowns": True,
            "routing_strategy": "latency-based-routing",
            "routing_strategy_args": {"ttl": 120},
        },
    ) as value:
        yield value


def test_latency_routing_stores_its_map_under_the_group_name_with_the_routing_ttl(latency_pair: Fleet) -> None:
    """`routing_strategy: latency-based-routing` with `routing_strategy_args.ttl: 120`. Three requests to
    `latency-ttl` go through the first instance.

    Redis must then hold `latency-ttl_map`. Decoded (it is written with `str(value)`, a Python literal), it must be
    an object with the key `dec1-ok-latency-ttl` whose `latency_normalized` is a non-empty list of numbers, and the
    key's TTL must be between 1 and 120 seconds.
    """
    for _ in range(3):
        assert chat(latency_pair.first, "latency-ttl").status_code == 200
    raw: Final = eventually(lambda: latency_pair.raw("latency-ttl_map"), lambda value: value is not None, seconds=10)
    stats: Final = object_value(object_value(decoded(raw))["dec1-ok-latency-ttl"])
    samples: Final = stats["latency_normalized"]
    assert isinstance(samples, list) and samples
    assert all(isinstance(sample, int | float) for sample in samples)
    assert 0 < latency_pair.ttl("latency-ttl_map") <= 120


def test_a_fresh_peer_routes_by_latency_it_read_from_the_shared_map(latency_pair: Fleet) -> None:
    """The first instance sends twelve warm-up requests to `latency-shared`, whose deployments answer at once
    (`ok-latency-fast`) and after 1.5 seconds (`mid-latency-slow`), until both appear in `latency-shared_map`.

    Eight requests through the second instance, which never served the group, must then all be served by
    `ok-latency-fast`. UNVERIFIED: that the peer's selection reads the map from Redis on a memory miss rather than
    starting from an empty map; see `LowestLatencyLoggingHandler.async_get_available_deployments`.
    """
    for _ in range(12):
        assert chat(latency_pair.first, "latency-shared").status_code == 200
    eventually(
        lambda: latency_pair.raw("latency-shared_map"),
        lambda raw: (
            raw is not None and {"dec1-ok-latency-fast", "dec1-mid-latency-slow"} <= set(object_value(decoded(raw)))
        ),
        seconds=20,
    )
    names: Final = tuple(served(chat(latency_pair.second, "latency-shared")) for _ in range(8))
    assert names == ("ok-latency-fast",) * 8


def test_a_deployment_with_an_rpm_limit_writes_the_router_cores_global_counter(tmp_path: Path) -> None:
    """Default routing; the only deployment of `global-counted` sets `rpm: 1000`. One request goes through the first
    instance.

    Within 10 seconds Redis must hold a key matching `global_router:dec1-ok-global-counted:*:rpm:<HH-MM>` with a
    value of at least 1. These are the router core's own counters (`Scope::GlobalRouter`,
    `redis-store/src/router_usage/keys.rs`). UNVERIFIED: that the core tracks usage for a deployment that only sets
    `rpm` (`record_success` counts "when `tracks` says it has limits to track").
    """
    with fleet(
        tmp_path,
        models=lambda api_base: (deployment("global-counted", "ok-global-counted", api_base, rpm=1000),),
        router_settings={"num_retries": 0},
    ) as single:
        minute: Final = utc_minute()
        assert chat(single.first, "global-counted").status_code == 200
        keys: Final = eventually(
            lambda: single.keys(f"global_router:dec1-ok-global-counted:*:rpm:{minute}"),
            lambda found: len(found) >= 1,
            seconds=10,
        )
        raw: Final = single.raw(min(keys))
        assert raw is not None and float(raw.decode()) >= 1
