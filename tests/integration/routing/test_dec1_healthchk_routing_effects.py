"""What background health check results do to routing and to the model listings.

With `enable_health_check_routing` the background loop writes each deployment's verdict into the router's
health state (`_write_health_state_to_router_cache`, litellm/proxy/proxy_server.py:5096) and the Rust router
core leaves unhealthy deployments out of selection while a healthy sibling exists. Two knobs shape that:
`health_check_ignore_transient_errors` keeps a deployment whose probe failed with 429 or 408 (a timeout) in
rotation, and the flag can be set under `general_settings` as well as `router_settings` (the proxy forwards
it, proxy_server.py:7747-7775).

Separately, `general_settings.model_list_healthy_only: true` (or `?healthy_only=true` per request) hides a
model from `/v1/models` and `/v1/model/info` when every deployment behind it failed its health check
(litellm/proxy/common_utils/healthy_model_filter.py). Hiding is presentation only: the model stays callable
and routing is untouched, and the filter fails open (quoted `"true"`, an `allowed_fails_policy`, or no health
state hide nothing).

Every deployment's upstream fails only the health probes (`probefail-<status>-*`) or every call
(`status-<status>-*`), so a client request that lands on the probed deployment proves it was kept in
rotation. Cooldowns are disabled and retries are off, so only health state moves traffic. The failing
deployment of each pair carries `weight: 1` and its sibling none, so by weight alone it would get every
request.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, held, list_value, object_value
from tests.integration.routing.dec1_healthchk_support import (
    Rig,
    chat,
    deployment,
    health_rig,
    listed_ids,
    report,
    served_text,
)

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:6]
CONVERGENCE_SECONDS: Final = 90
BATCH: Final = 6
PROBE_TIMEOUT_SECONDS: Final = 2
BACKGROUND: Final[dict[str, JsonValue]] = {"background_health_checks": True, "health_check_interval": 2}


def _pair(label: str, failing: str, api_base: str, **model_info: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    group: Final = f"route-{label}-{RUN}"
    return (
        deployment(group, failing, api_base, model_info=dict(model_info), weight=1),
        deployment(group, f"ok-{label}-sibling-{RUN}", api_base),
    )


T429: Final = f"probefail-429-t429-{RUN}"
T408: Final = f"probeslow-t408-{RUN}"
T401: Final = f"probefail-401-t401-{RUN}"
T500: Final = f"probefail-500-t500-{RUN}"
PLAIN429: Final = f"probefail-429-plain-{RUN}"
DOWN_GROUP: Final = f"route-down-{RUN}"
UP_GROUP: Final = f"route-up-{RUN}"
MIXED_GROUP: Final = f"route-mixed-{RUN}"
DOWN_MODEL: Final = f"status-500-down-{RUN}"


def _transient_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        *_pair("t429", T429, api_base),
        *_pair("t408", T408, api_base, health_check_timeout=PROBE_TIMEOUT_SECONDS),
        *_pair("t401", T401, api_base),
        *_pair("t500", T500, api_base),
        deployment(DOWN_GROUP, DOWN_MODEL, api_base),
        deployment(UP_GROUP, f"ok-up-{RUN}", api_base),
    )


def _plain_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (*_pair("plain", PLAIN429, api_base), deployment(DOWN_GROUP, DOWN_MODEL, api_base))


def _listing_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment(DOWN_GROUP, DOWN_MODEL, api_base),
        deployment(UP_GROUP, f"ok-up-{RUN}", api_base),
        deployment(MIXED_GROUP, f"status-500-mixed-{RUN}", api_base, weight=1),
        deployment(MIXED_GROUP, f"ok-mixed-{RUN}", api_base),
    )


@pytest.fixture(scope="module")
def transient(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-transient"),
        _transient_models,
        router_settings={"enable_health_check_routing": True, "health_check_ignore_transient_errors": True},
        general_settings=BACKGROUND,
    ) as value:
        yield value


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-plain"),
        _plain_models,
        general_settings={**BACKGROUND, "enable_health_check_routing": True, "model_list_healthy_only": "true"},
    ) as value:
        yield value


@pytest.fixture(scope="module")
def listing(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-listing"),
        _listing_models,
        general_settings={**BACKGROUND, "model_list_healthy_only": True},
    ) as value:
        yield value


@pytest.fixture(scope="module")
def policy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-policy"),
        _listing_models,
        router_settings={"allowed_fails_policy": {"RateLimitErrorAllowedFails": 100}},
        general_settings={**BACKGROUND, "model_list_healthy_only": True},
    ) as value:
        yield value


def _served_batch(rig: Rig, group: str) -> frozenset[str]:
    responses: Final = tuple(chat(rig.gateway, group) for _ in range(BATCH))
    return frozenset(
        served_text(response) if response.status_code == 200 else f"status {response.status_code}"
        for response in responses
    )


def _probed(rig: Rig, model: str, count: int = 2) -> None:
    eventually(lambda: len(rig.upstream.probes(model)), lambda seen: seen >= count, seconds=CONVERGENCE_SECONDS)


def test_with_ignore_transient_errors_a_probe_429_keeps_the_deployment_in_rotation(transient: Rig) -> None:
    """Group `route-t429` has a weighted deployment whose upstream answers 429 to every health probe and 200
    to client calls; `health_check_ignore_transient_errors: true`.

    After at least two failed probes, six client requests must all be served by that weighted deployment
    (content `served probefail-429-t429-<run>`): a rate limited probe is not an outage.
    """
    _probed(transient, T429)
    assert held(
        lambda: _served_batch(transient, f"route-t429-{RUN}"),
        lambda seen: seen == {f"served {T429}"},
        holding=4,
        seconds=30,
    ) == {f"served {T429}"}


def test_with_ignore_transient_errors_a_probe_timeout_keeps_the_deployment_in_rotation(transient: Rig) -> None:
    """Group `route-t408` has a weighted deployment whose upstream stalls 8 seconds on health probes only, with
    `model_info.health_check_timeout: 2`.

    The probe is cut off as a 408 `Timeout exceeded`, which the transient filter ignores, so six client
    requests must all be served by the weighted deployment (`served probeslow-t408-<run>`).
    """
    _probed(transient, T408)
    assert held(
        lambda: _served_batch(transient, f"route-t408-{RUN}"),
        lambda seen: seen == {f"served {T408}"},
        holding=4,
        seconds=60,
    ) == {f"served {T408}"}


@pytest.mark.parametrize(("label", "failing"), [("t401", T401), ("t500", T500)], ids=["t401", "t500"])
def test_with_ignore_transient_errors_a_probe_401_or_500_still_removes_the_deployment(
    transient: Rig, label: str, failing: str
) -> None:
    """Groups `route-t401` and `route-t500` have a weighted deployment whose upstream answers 401 or 500 to
    health probes and 200 to clients, with `health_check_ignore_transient_errors: true`.

    Within 90 seconds, six client requests must all be served by the unweighted sibling
    (`served ok-<label>-sibling-<run>`): only 429 and 408 are transient.
    """
    sibling: Final = f"served ok-{label}-sibling-{RUN}"
    eventually(
        lambda: _served_batch(transient, f"route-{label}-{RUN}"),
        lambda seen: seen == {sibling},
        seconds=CONVERGENCE_SECONDS,
    )
    assert transient.upstream.probes(failing)


def test_per_request_healthy_only_hides_a_fully_unhealthy_model_from_the_listing(transient: Rig) -> None:
    """The `transient` proxy has `enable_health_check_routing` on (so health state is kept) but no
    `model_list_healthy_only`. Group `route-down` has one deployment that fails every probe.

    After it has failed a probe, `GET /v1/models` must still list `route-down-<run>`, while
    `GET /v1/models?healthy_only=true` must not, and must still list `route-up-<run>`.
    """
    _probed(transient, DOWN_MODEL)
    assert DOWN_GROUP in listed_ids(transient.gateway)
    hidden: Final = eventually(
        lambda: listed_ids(transient.gateway, params={"healthy_only": "true"}),
        lambda ids: DOWN_GROUP not in ids,
        seconds=CONVERGENCE_SECONDS,
    )
    assert UP_GROUP in hidden


def test_without_ignore_transient_errors_a_probe_429_removes_the_deployment(plain: Rig) -> None:
    """The `plain` proxy turns routing on through `general_settings.enable_health_check_routing: true` and
    leaves `health_check_ignore_transient_errors` unset. Group `route-plain` has a weighted deployment whose
    probes get 429.

    Within 90 seconds six client requests must all be served by the sibling (`served ok-plain-sibling-<run>`).
    This pins both the default (429 probes count) and that the general_settings spelling of the routing flag
    reaches the router.
    """
    sibling: Final = f"served ok-plain-sibling-{RUN}"
    assert eventually(
        lambda: _served_batch(plain, f"route-plain-{RUN}"), lambda seen: seen == {sibling}, seconds=CONVERGENCE_SECONDS
    ) == {sibling}


def test_a_quoted_true_for_model_list_healthy_only_hides_nothing(plain: Rig) -> None:
    """The `plain` proxy sets `model_list_healthy_only: "true"` (a string) and keeps health state through the
    routing flag. Group `route-down` fails every probe.

    After it has failed two probes, `GET /v1/models` must still list `route-down-<run>`: only a real boolean
    `true` turns the proxy wide filter on (`is_healthy_only_listing_default`).
    """
    _probed(plain, DOWN_MODEL)
    assert DOWN_GROUP in held(lambda: listed_ids(plain.gateway), lambda ids: DOWN_GROUP in ids, holding=4, seconds=20)


def test_model_list_healthy_only_hides_a_model_whose_every_deployment_is_unhealthy(listing: Rig) -> None:
    """The `listing` proxy sets `model_list_healthy_only: true` with background checks and no routing flag.

    Within 90 seconds, `GET /v1/models` must stop listing `route-down-<run>` (its one deployment fails every
    probe) while still listing `route-up-<run>` and `route-mixed-<run>` (one healthy deployment of two is enough).
    """
    ids: Final = eventually(
        lambda: listed_ids(listing.gateway), lambda seen: DOWN_GROUP not in seen, seconds=CONVERGENCE_SECONDS
    )
    assert {UP_GROUP, MIXED_GROUP} <= ids


def test_a_request_cannot_turn_the_proxy_wide_filter_back_off(listing: Rig) -> None:
    """With `model_list_healthy_only: true`, a client sends `GET /v1/models?healthy_only=false`.

    `route-down-<run>` must still be hidden: `healthy_only=false` is indistinguishable from absent
    (`is_healthy_only_enabled`).
    """
    eventually(lambda: listed_ids(listing.gateway), lambda seen: DOWN_GROUP not in seen, seconds=CONVERGENCE_SECONDS)
    assert DOWN_GROUP not in listed_ids(listing.gateway, params={"healthy_only": "false"})


def test_model_info_hides_the_fully_unhealthy_model_too(listing: Rig) -> None:
    """With `model_list_healthy_only: true`, once `/v1/models` hides `route-down-<run>`, the master key calls
    `GET /v1/model/info`.

    No entry may have `model_name` `route-down-<run>`, and `route-up-<run>` must be present: both listings
    share the filter.
    """
    eventually(lambda: listed_ids(listing.gateway), lambda seen: DOWN_GROUP not in seen, seconds=CONVERGENCE_SECONDS)
    response: Final = listing.gateway.request("GET", "/v1/model/info")
    assert response.status_code == 200, response.text
    names: Final = {object_value(entry).get("model_name") for entry in list_value(report(response)["data"])}
    assert DOWN_GROUP not in names and UP_GROUP in names


def test_a_hidden_model_is_still_callable_and_returns_the_providers_error(listing: Rig) -> None:
    """`route-down-<run>` is hidden from `/v1/models` because its one deployment fails every probe.

    A chat request for it must still reach the upstream and return the provider's 500, not a 400 `Invalid
    model name` or a 404: hiding is presentation only.
    """
    eventually(lambda: listed_ids(listing.gateway), lambda seen: DOWN_GROUP not in seen, seconds=CONVERGENCE_SECONDS)
    before: Final = len(listing.upstream.client_calls(DOWN_MODEL))
    response: Final = chat(listing.gateway, DOWN_GROUP)
    assert response.status_code == 500, response.text
    assert len(listing.upstream.client_calls(DOWN_MODEL)) == before + 1


def test_the_listing_filter_alone_does_not_change_routing(listing: Rig) -> None:
    """The `listing` proxy keeps health state for the filter but has no `enable_health_check_routing`.
    Group `route-mixed` has a weighted deployment that fails everything and a healthy sibling.

    After the failing deployment has failed two probes, six client requests must all still go to it and
    return 500: the listing setting writes the health cache but never filters routing
    (`_write_health_state_to_router_cache` returns before the routing bookkeeping).
    """
    failing: Final = f"status-500-mixed-{RUN}"
    _probed(listing, failing)
    assert _served_batch(listing, MIXED_GROUP) == {"status 500"}


def test_an_allowed_fails_policy_makes_the_listing_filter_hide_nothing(policy: Rig) -> None:
    """The `policy` proxy sets `model_list_healthy_only: true` and a `router_settings.allowed_fails_policy`.

    After `route-down-<run>` has failed two probes, `GET /v1/models` must keep listing it for another 6 seconds:
    `fully_unhealthy_model_names` returns nothing under an allowed fails policy
    (litellm/router_rust/surface/health_routing.py:129). Read from code; the reason is not documented.
    """
    _probed(policy, DOWN_MODEL)
    assert DOWN_GROUP in held(lambda: listed_ids(policy.gateway), lambda ids: DOWN_GROUP in ids, holding=6, seconds=20)
