"""Priority names, pool sizing edges, capacity headers by route, multi-instance class shares and a Redis outage.

Priority names. Admission reads the priority from the team's metadata first and the key's metadata second
(``_get_priority_from_user_api_key_dict``, dynamic_rate_limiter_v3.py:484); the request body is never read. A name
missing from ``litellm.priority_reservation`` counts in the shared ``<m>:default_pool`` (``_get_priority_allocation``,
:532) but is still echoed in ``x-litellm-priority`` and in refusal texts. A reservation typed ``rpm`` on a model with
no ``rpm`` is read as a plain fraction and then normalized with the others (``convert_priority_to_percent``,
rate_limiter_utils.py:141-173; ``_normalize_priority_weights``, :508).

Headers. Every served response the post-call hook sees gets ``x-litellm-priority`` and ``x-litellm-rate-limiter-version:
v3`` (:1766-1776) plus one ``x-ratelimit-<descriptor>-<limit|remaining>-<requests|tokens>`` pair per tracked counter
(parallel_request_limiter_v3.py:4750).

Multi-instance and Redis. The active caller registry behind class share slices is one Redis sorted set per model
(priority_class_share.py:96), so callers seen by one instance shrink the slice on the other. When the registry cannot
be read the caller is held to its class pool instead (:663-674).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually
from tests.integration._support.redis_process import owned_redis
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap4_support import counter, counter_settles
from tests.integration.authorization.dec1_limits_support import complete
from tests.integration.routing.dec1_capacity3_support import CLASS_SHARE_SCOPE, class_share_value
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    PRIORITY_SCOPE,
    capacity_proxy,
    chat,
    embed,
    entry,
    error_message,
    group_name,
    priority_message,
    provider,
    scope_headers,
    statuses,
    wait_for_model,
)


@contextmanager
def _rig(
    gateway: Gateway,
    directory: Path,
    *,
    limits: Mapping[str, JsonValue],
    reservation: Mapping[str, JsonValue],
    priority_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[tuple[Gateway, str, Scenario]]:
    group: Final = group_name("c3prio")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            directory,
            model_list=[entry(group, wire.url, **limits)],
            priority_reservation=reservation,
            priority_settings=priority_settings,
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield candidate, group, scenario


def test_a_priority_in_the_request_body_metadata_is_ignored(gateway: Gateway, tmp_path: Path) -> None:
    """``{"premium": 0.5}`` on ``rpm: 100``. A key without a priority sends ``metadata: {"priority": "premium"}`` in
    the body. The answer reports ``x-litellm-priority: default`` and the request counts on
    ``{priority_model:<m>:default_pool}:requests`` (1), never on ``premium``: callers cannot pick their own pool."""
    with _rig(gateway, tmp_path, limits={"rpm": 100}, reservation={"premium": 0.5}) as (candidate, group, scenario):
        served: Final = chat(candidate, group, scenario.key(), metadata={"priority": "premium"})
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-priority"] == "default"
        assert counter_settles(PRIORITY_SCOPE, f"{group}:default_pool", "requests", 1) == 1
        assert counter(PRIORITY_SCOPE, f"{group}:premium", "requests") is None


def test_a_teams_unreserved_priority_wins_over_its_keys_reserved_one(gateway: Gateway, tmp_path: Path) -> None:
    """The team has ``metadata.priority: mystery`` (not reserved) and its key ``premium`` (reserved). The team wins
    (:498-504), so the answer reports ``x-litellm-priority: mystery`` and the request counts on ``default_pool``, not
    on ``premium``: an unreserved team name cannot be escaped by a key-level priority."""
    with _rig(gateway, tmp_path, limits={"rpm": 100}, reservation={"premium": 0.5}) as (candidate, group, scenario):
        team: Final = scenario.team(metadata={"priority": "mystery"})
        served: Final = chat(candidate, group, scenario.key(team_id=team, metadata={"priority": "premium"}))
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-priority"] == "mystery"
        assert counter_settles(PRIORITY_SCOPE, f"{group}:default_pool", "requests", 1) == 1
        assert counter(PRIORITY_SCOPE, f"{group}:premium", "requests") is None


def test_an_unreserved_priority_is_refused_on_the_default_pool_under_its_own_name(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``{"premium": 0.5}`` on ``rpm: 4`` with ``saturation_threshold: 0.0``, so the default pool is ``int(4 x 0.25) =
    1``. A key with ``priority: mystery`` is served once; its second request is refused with exactly
    ``Priority-based rate limit exceeded. Model: <m>, Priority: mystery, Rate limit type: requests, Model TPM: not
    configured, Model RPM: 4, Remaining: 0, Model saturation: 25.0%`` and ``x-litellm-priority: mystery`` (:1329-1349):
    the pool is the default one but the text names what the key asked for."""
    with _rig(
        gateway,
        tmp_path,
        limits={"rpm": 4},
        reservation={"premium": 0.5},
        priority_settings={"saturation_threshold": 0.0, "saturation_check_cache_ttl": 0},
    ) as (candidate, group, scenario):
        key: Final = scenario.key(metadata={"priority": "mystery"})
        assert chat(candidate, group, key).status_code == 200
        refused: Final = chat(candidate, group, key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == priority_message(group, "mystery", "requests", "25.0%", rpm=4)
        assert refused.headers["x-litellm-priority"] == "mystery"


def test_an_unreserved_priority_and_no_priority_share_one_default_pool_counter(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key with ``priority: mystery`` and a key without a priority each send one request on an idle model.
    ``{priority_model:<m>:default_pool}:requests`` holds 2: both land in the one shared pool (:566-570)."""
    with _rig(gateway, tmp_path, limits={"rpm": 100}, reservation={"premium": 0.5}) as (candidate, group, scenario):
        responses: Final = [
            chat(candidate, group, scenario.key(metadata={"priority": "mystery"})),
            chat(candidate, group, scenario.key()),
        ]
        assert statuses(responses) == [200, 200]
        assert counter_settles(PRIORITY_SCOPE, f"{group}:default_pool", "requests", 2) == 2


def test_an_rpm_typed_reservation_on_a_model_without_rpm_is_read_as_a_fraction_and_normalized(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``{"bulk": {"type": "rpm", "value": 5}, "lite": 0.5}`` on a ``tpm: 1000`` model with no ``rpm``. ``bulk`` cannot
    be converted, so it is read as the fraction 5.0, and the weights 5.0 and 0.5 are normalized by their sum 5.5. A
    ``lite`` key's served response reports ``x-ratelimit-priority_model-limit-tokens: 90`` (``int(1000 x 0.5 / 5.5)``)
    and ``remaining-tokens: 90`` (tracking adds no tokens), with no request headers for the pool."""
    with _rig(
        gateway, tmp_path, limits={"tpm": 1000}, reservation={"bulk": {"type": "rpm", "value": 5}, "lite": 0.5}
    ) as (candidate, group, scenario):
        served: Final = chat(candidate, group, scenario.key(metadata={"priority": "lite"}))
        assert served.status_code == 200, served.text
        assert scope_headers(served, PRIORITY_SCOPE) == {"limit-tokens": "90", "remaining-tokens": "90"}


def _completion(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return complete(gateway, model, key)


@pytest.mark.parametrize(
    "send",
    (
        pytest.param(chat, id="chat_completions"),
        pytest.param(embed, id="embeddings"),
        pytest.param(_completion, id="text_completions"),
    ),
)
def test_every_served_route_reports_the_limiter_version_priority_and_model_window(
    gateway: Gateway, tmp_path: Path, send: Callable[[Gateway, str, str], httpx.Response]
) -> None:
    """One first request on an ``rpm: 10`` model through each route is served with ``x-litellm-rate-limiter-version:
    v3``, ``x-litellm-priority: default`` and exactly ``x-ratelimit-model_saturation_check-limit-requests: 10`` and
    ``...-remaining-requests: 9`` (:1766-1776). A port that adds headers only on chat would drop them silently on the
    other routes."""
    with _rig(gateway, tmp_path, limits={"rpm": 10}, reservation={"premium": 0.5}) as (candidate, group, _scenario):
        served: Final = send(candidate, group, candidate.key)
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-rate-limiter-version"] == "v3"
        assert served.headers["x-litellm-priority"] == "default"
        assert scope_headers(served, MODEL_SCOPE) == {"limit-requests": "10", "remaining-requests": "9"}


def test_a_streamed_chat_completion_carries_no_model_window_headers(gateway: Gateway, tmp_path: Path) -> None:
    """A first ``stream: true`` chat completion on an ``rpm: 10`` model is answered 200 without any
    ``x-ratelimit-model_saturation_check-*`` header. The limiter adds its headers in ``async_post_call_success_hook``
    (litellm/proxy/hooks/dynamic_rate_limiter_v3.py:1769-1795), which a streamed answer's headers are sent before;
    run 44176682961 saw none. The served request is still counted, so a second non-streamed request reports
    ``remaining-requests: 8``."""
    with _rig(gateway, tmp_path, limits={"rpm": 10}, reservation={"premium": 0.5}) as (candidate, group, _scenario):
        streamed: Final = chat(candidate, group, candidate.key, stream=True)
        assert streamed.status_code == 200, streamed.text
        assert scope_headers(streamed, MODEL_SCOPE) == {}, dict(streamed.headers)
        unary: Final = chat(candidate, group, candidate.key)
        assert unary.status_code == 200, unary.text
        assert scope_headers(unary, MODEL_SCOPE) == {"limit-requests": "10", "remaining-requests": "8"}


@contextmanager
def _split_pair(gateway: Gateway, directory: Path) -> Iterator[tuple[Gateway, Gateway, str, Scenario]]:
    """Two instances on the shared database and Redis, teams splitting ``{"interactive": 1.0}`` on an idle ``rpm: 10``
    model."""
    group: Final = group_name("c3pair")
    settings: Final[dict[str, JsonValue]] = {
        "saturation_threshold": 0.9,
        "priority_class_share_config": {"split_equally_among_active": ["team"]},
    }
    for name in ("first", "second"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    with wire_server(provider()) as wire:
        model_list: Final = [entry(group, wire.url, rpm=10)]
        with (
            capacity_proxy(
                gateway,
                directory / "first",
                model_list=model_list,
                priority_reservation={"interactive": 1.0},
                priority_settings=settings,
                shared=True,
            ) as first,
            capacity_proxy(
                gateway,
                directory / "second",
                model_list=model_list,
                priority_reservation={"interactive": 1.0},
                priority_settings=settings,
                shared=True,
            ) as second,
            first.scenario() as scenario,
        ):
            wait_for_model(second, group)
            yield first, second, group, scenario


def test_a_team_active_on_one_instance_halves_the_slice_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """Two instances. Team A's request through the first reports ``priority_class_share_saturated`` ``limit-requests:
    10``; team B's first request through the second then reports ``limit-requests: 5``, because the active caller set
    is the shared Redis sorted set ``dynamic_rate_limiter:active_teams:<m>`` (priority_class_share.py:96-111). A port
    that kept the set per process would give each instance's caller the whole class."""
    with _split_pair(gateway, tmp_path) as (first, second, group, scenario):
        team_a: Final = scenario.key(team_id=scenario.team(metadata={"priority": "interactive"}))
        team_b: Final = scenario.key(team_id=scenario.team(metadata={"priority": "interactive"}))
        assert scope_headers(chat(first, group, team_a), CLASS_SHARE_SCOPE)["limit-requests"] == "10"
        served: Final = eventually(
            lambda: chat(second, group, team_b), lambda response: response.status_code == 200, seconds=30
        )
        assert scope_headers(served, CLASS_SHARE_SCOPE)["limit-requests"] == "5", dict(served.headers)


def test_a_teams_slice_counts_its_requests_from_both_instances(gateway: Gateway, tmp_path: Path) -> None:
    """Two instances. One team sends one request through each, and its slice counter
    ``{priority_class_share_saturated:<len>:<m>:<team_id>}:requests`` holds 2: idle tracking (#51952) writes the shared
    counter, so a slice used on one instance is already spent when the model saturates on the other."""
    with _split_pair(gateway, tmp_path) as (first, second, group, scenario):
        team: Final = scenario.team(metadata={"priority": "interactive"})
        key: Final = scenario.key(team_id=team)
        assert chat(first, group, key).status_code == 200
        eventually(lambda: chat(second, group, key), lambda response: response.status_code == 200, seconds=30)
        assert counter_settles(CLASS_SHARE_SCOPE, class_share_value(group, team), "requests", 2) == 2


def test_when_redis_stops_a_class_share_team_is_held_to_its_class_pool(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy on a Redis this test owns, teams splitting ``{"interactive": 0.5}`` on a saturated (``saturation_threshold:
    0.0``) ``rpm: 10`` model. Redis is stopped, so the active caller registry cannot be read and the team is held to
    the class pool (:663-674): after five served requests the sixth is refused with a text starting ``Priority-based
    rate limit exceeded. Model: <m>, Priority: interactive`` (the ``priority_model`` pool of ``int(10 x 0.5) = 5``),
    not the class share text. UNVERIFIED end to end: assumes counters fail open onto memory as
    ``test_dec1_capacity_shared_state.py`` assumes, and that the registry raises a ``RedisError``."""
    group: Final = group_name("c3outage")
    with (
        wire_server(provider()) as wire,
        owned_redis(tmp_path / "redis") as redis,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, rpm=10)],
            priority_reservation={"interactive": 0.5},
            priority_settings={
                "saturation_threshold": 0.0,
                "priority_class_share_config": {"split_equally_among_active": ["team"]},
            },
            environment={"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)},
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(team_id=scenario.team(metadata={"priority": "interactive"}))
        redis.stop()
        first: Final = eventually(
            lambda: chat(candidate, group, key), lambda response: response.status_code < 500, seconds=30
        )
        assert first.status_code == 200, first.text
        served: Final = [chat(candidate, group, key) for _ in range(4)]
        assert statuses(served) == [200] * 4, [response.text for response in served]
        refused: Final = chat(candidate, group, key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused).startswith(
            f"Priority-based rate limit exceeded. Model: {group}, Priority: interactive"
        ), error_message(refused)
        redis.start()
