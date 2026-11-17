"""Model capacity under router retries and provider-error fallbacks, and capacity changes through /model/update.

Retries. The dynamic rate limiter v3 admits in the proxy pre-call hook (``async_pre_call_hook``,
``litellm/proxy/hooks/dynamic_rate_limiter_v3.py:1638``), once per proxy request. Router retries inside one model
group are not admitted again, so a retried request takes one request of the group's window, and settlement
(``async_log_success_event``, :1785) charges the usage of the attempt that answered.

Fallbacks after a provider error. The router asks ``RouterFallbackRateLimitCheck`` (``litellm/proxy/auth/
fallback_rate_limit.py:38``) before each cross-group fallback. The dynamic limiter's ``async_fallback_target_allowed``
(:1503) returns early when no borrower has a share on the target (:1516), so without borrowing the fallback group's
model-wide window is neither checked nor counted, while its success still settles tokens on that group. With a
borrowing team that has a limit on the target, the attempt is committed and counted on the target's window
(:1542). Capacity refusals (not provider errors) fall back through the proxy instead and are pinned by
``test_dec1_capacity_fallbacks.py``.

Updates. A group's capacity is the sum of its deployments' ``rpm``/``tpm`` read from the live router on every
request, so ``PATCH /model/{id}/update`` changes it without a restart, and a peer that synced the model change over
Redis enforces the new number against the same shared counter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap4_support import (
    PEER_PICKUP_SECONDS,
    PROVIDER_TOTAL,
    FlakyProvider,
    counter,
    counter_settles,
    model_with_id,
    update_model,
)
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    capacity_message,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    posts,
    provider,
    scope_headers,
    statuses,
    wait_for_model,
)

RETRIES: Final = {"num_retries": 2}
BORROWING: Final = {"model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]}}


def test_a_retried_request_counts_once_in_the_model_capacity_window(gateway: Gateway, tmp_path: Path) -> None:
    """On a proxy with ``num_retries: 2`` the provider fails the first attempt and answers the retry. Redis holds 1 on
    ``{model_saturation_check:<group>}:requests`` and 18 on ``{model_saturation_check:<group>}:tokens``: one
    admission, and the answering attempt's usage settled once."""
    group: Final = group_name("retrycount")
    flaky: Final = FlakyProvider(failures=1)
    with (
        wire_server(flaky) as wire,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, wire.url, rpm=10)], router_settings=RETRIES, shared=True
        ) as candidate,
    ):
        assert chat(candidate, group, candidate.key).status_code == 200
        assert flaky.attempts >= 2
        assert counter_settles(MODEL_SCOPE, group, "tokens", PROVIDER_TOTAL) == PROVIDER_TOTAL
        assert counter(MODEL_SCOPE, group, "requests") == 1


def test_retries_do_not_use_up_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """A group with ``rpm: 2``. The first request is retried once and served, the second is served, and the third is
    refused with ``Model capacity reached for <group>. ... Model RPM: 2, Remaining: 0``: the retry did not take the
    second request of the window."""
    group: Final = group_name("retryroom")
    with (
        wire_server(FlakyProvider(failures=1)) as wire,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, wire.url, rpm=2)], router_settings=RETRIES
        ) as candidate,
    ):
        assert statuses([chat(candidate, group, candidate.key) for _ in range(2)]) == [200, 200]
        refused: Final = chat(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == capacity_message(group, None, "requests", rpm=2)


def test_a_request_that_fails_every_retry_keeps_its_request_and_settles_no_tokens(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The provider fails all three attempts of a request on a proxy with ``num_retries: 2``. The model window keeps
    the one admitted request (``requests`` = 1) and has no token counter: failures settle nothing on the model-wide
    window."""
    group: Final = group_name("retryfail")
    flaky: Final = FlakyProvider(failures=-1)
    with (
        wire_server(flaky) as wire,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, wire.url, rpm=10)], router_settings=RETRIES, shared=True
        ) as candidate,
    ):
        assert chat(candidate, group, candidate.key).status_code >= 500
        assert flaky.attempts >= 3
        assert counter_settles(MODEL_SCOPE, group, "requests", 1) == 1
        assert counter(MODEL_SCOPE, group, "tokens") is None


def test_a_provider_error_fallback_settles_its_tokens_on_the_fallback_group(gateway: Gateway, tmp_path: Path) -> None:
    """``primary`` always fails and falls back to ``spare`` (``num_retries: 0``). The caller gets 200 from ``spare``.
    ``{model_saturation_check:<spare>}:tokens`` settles to 18, because settlement uses the model group of the attempt
    that answered, and ``primary``'s window keeps its one admitted request and no tokens."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(FlakyProvider(failures=-1)) as broken,
        wire_server(provider()) as healthy,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, broken.url, rpm=10), entry(spare, healthy.url, rpm=10)],
            router_settings={"fallbacks": [{primary: [spare]}]},
            shared=True,
        ) as candidate,
    ):
        assert chat(candidate, primary, candidate.key).status_code == 200
        assert counter_settles(MODEL_SCOPE, spare, "tokens", PROVIDER_TOTAL) == PROVIDER_TOTAL
        assert counter(MODEL_SCOPE, primary, "requests") == 1
        assert counter(MODEL_SCOPE, primary, "tokens") is None


def test_without_borrowing_a_provider_error_fallback_takes_no_request_of_the_fallback_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Same fallback without any borrowing config. After the request ``{model_saturation_check:<spare>}:requests``
    is still absent: ``async_fallback_target_allowed`` returns before counting when no borrower has a share on the
    target (:1516). Pinned so the Rust port changes it on purpose; see the skipped test below for the capacity it
    lets a fallback exceed."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(FlakyProvider(failures=-1)) as broken,
        wire_server(provider()) as healthy,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, broken.url, rpm=10), entry(spare, healthy.url, rpm=10)],
            router_settings={"fallbacks": [{primary: [spare]}]},
            shared=True,
        ) as candidate,
    ):
        assert chat(candidate, primary, candidate.key).status_code == 200
        assert counter_settles(MODEL_SCOPE, spare, "tokens", PROVIDER_TOTAL) == PROVIDER_TOTAL
        assert counter(MODEL_SCOPE, spare, "requests") is None


def test_a_provider_error_fallback_is_held_to_the_fallback_groups_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """``spare`` has ``rpm: 1`` and a direct request already used it. A request for ``primary`` fails at the provider
    and the router considers ``spare``. ``spare`` is full, so it should not be attempted: the caller gets
    ``primary``'s provider error and ``spare``'s provider sees nothing more."""
    pytest.skip(
        "BUG: a provider-error fallback is not checked against the fallback group's model capacity when no "
        "borrower has a share there (dynamic_rate_limiter_v3.py:1516), so a full group still serves fallbacks"
    )
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(FlakyProvider(failures=-1)) as broken,
        wire_server(provider()) as healthy,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, broken.url, rpm=10), entry(spare, healthy.url, rpm=1)],
            router_settings={"fallbacks": [{primary: [spare]}]},
        ) as candidate,
    ):
        assert chat(candidate, spare, candidate.key).status_code == 200
        healthy.drain()
        failed: Final = chat(candidate, primary, candidate.key)
        assert failed.status_code >= 500, failed.text
        assert posts(healthy) == ()


def test_with_a_borrowing_team_a_provider_error_fallback_counts_on_the_fallback_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Teams borrow, and the caller's team has ``model_tpm_limit`` on ``spare``. When ``primary`` fails and the router
    falls back, the attempt is committed on ``spare``: ``{model_saturation_check:<spare>}:requests`` holds 1
    (``async_fallback_target_allowed`` with ``commit``, :1542)."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(FlakyProvider(failures=-1)) as broken,
        wire_server(provider()) as healthy,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, broken.url, rpm=10), entry(spare, healthy.url, rpm=10, tpm=100000)],
            priority_settings=BORROWING,
            router_settings={"fallbacks": [{primary: [spare]}]},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(metadata={"model_tpm_limit": {spare: 1000}})
        key: Final = scenario.key(team_id=team)
        assert chat(candidate, primary, key).status_code == 200
        assert counter_settles(MODEL_SCOPE, spare, "requests", 1) == 1


def test_raising_a_deployments_rpm_through_model_update_raises_the_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """A DB model has one deployment with ``rpm: 1``; its window is full after one request. ``PATCH
    /model/{id}/update`` with ``litellm_params.rpm: 3`` takes effect without a restart: a later request is served and
    reports model-wide ``limit-requests: 3``."""
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        group, identity = model_with_id(scenario, f"{wire.url}/v1", rpm=1)
        eventually(
            lambda: chat(candidate, group, candidate.key), lambda response: response.status_code == 200, seconds=60
        )
        assert chat(candidate, group, candidate.key).status_code == 429
        update_model(candidate, identity, rpm=3)
        served: Final = eventually(
            lambda: chat(candidate, group, candidate.key), lambda response: response.status_code == 200, seconds=60
        )
        assert scope_headers(served, MODEL_SCOPE)["limit-requests"] == "3", dict(served.headers)


def test_lowering_a_deployments_rpm_through_model_update_refuses_sooner(gateway: Gateway, tmp_path: Path) -> None:
    """A DB model with ``rpm: 10`` serves two requests. ``PATCH /model/{id}/update`` lowers ``rpm`` to 2; once the
    proxy applied it the next request is refused with ``Model capacity reached for <group>. ... Model RPM: 2,
    Remaining: 0`` (the two earlier requests are still in the window)."""
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        group, identity = model_with_id(scenario, f"{wire.url}/v1", rpm=10)
        eventually(
            lambda: chat(candidate, group, candidate.key), lambda response: response.status_code == 200, seconds=60
        )
        assert chat(candidate, group, candidate.key).status_code == 200
        update_model(candidate, identity, rpm=2)
        refused: Final = eventually(
            lambda: chat(candidate, group, candidate.key), lambda response: response.status_code == 429, seconds=60
        )
        assert error_message(refused) == capacity_message(group, None, "requests", rpm=2)


def test_a_capacity_raised_on_one_instance_is_enforced_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """Two proxies with the dynamic limiter share the database and Redis. A DB model with ``rpm: 1`` is created on
    the first; the second fills its window with one request and is refused. ``PATCH /model/{id}/update`` on the
    first raises ``rpm`` to 3, and within seconds the second serves the model again and reports ``limit-requests:
    3``: the model change reached the peer over Redis and the shared counter kept the first request."""
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path / "first", shared=True) as first,
        capacity_proxy(gateway, tmp_path / "second", shared=True) as second,
        first.scenario() as scenario,
    ):
        group, identity = model_with_id(scenario, f"{wire.url}/v1", rpm=1)
        wait_for_model(second, group)
        assert chat(second, group, second.key).status_code == 200
        assert chat(second, group, second.key).status_code == 429
        update_model(first, identity, rpm=3)
        served: Final = eventually(
            lambda: chat(second, group, second.key),
            lambda response: response.status_code == 200,
            seconds=PEER_PICKUP_SECONDS,
        )
        assert scope_headers(served, MODEL_SCOPE)["limit-requests"] == "3", dict(served.headers)


def test_a_model_window_used_on_one_instance_refuses_after_a_capacity_cut_on_the_other(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Two proxies share the database and Redis. A DB model with ``rpm: 10`` serves two requests on the first. The
    second lowers ``rpm`` to 2 through ``PATCH /model/{id}/update``, and the first then refuses the model with
    ``Model RPM: 2``: the cut reached the instance that used the window."""
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path / "first", shared=True) as first,
        capacity_proxy(gateway, tmp_path / "second", shared=True) as second,
        first.scenario() as scenario,
    ):
        group, identity = model_with_id(scenario, f"{wire.url}/v1", rpm=10)
        eventually(lambda: chat(first, group, first.key), lambda response: response.status_code == 200, seconds=60)
        assert chat(first, group, first.key).status_code == 200
        wait_for_model(second, group)
        update_model(second, identity, rpm=2)
        refused: Final = eventually(
            lambda: chat(first, group, first.key),
            lambda response: response.status_code == 429,
            seconds=PEER_PICKUP_SECONDS,
        )
        assert error_message(refused) == capacity_message(group, None, "requests", rpm=2)
