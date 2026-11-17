"""`GET /health` and its sibling routes, run on demand: targeting, error reporting, timeouts and who may ask.

Without `background_health_checks` every `GET /health` probes the deployments it targets right then and
answers with `healthy_endpoints`, `unhealthy_endpoints`, `healthy_count` and `unhealthy_count`
(`health_endpoint`, litellm/proxy/health_endpoints/_health_endpoints.py:1563). `model` targets a group (or,
failing that, a `litellm_params.model` string), `model_id` targets one deployment and wins over `model`. A
targeted check with no healthy endpoint answers 503. Every unhealthy entry carries the provider error and
the `exception_status` the health routing uses. A probe that runs past `model_info.health_check_timeout`
is cut off and reported as `Timeout exceeded`.

Auth: `/health` is an info route any key may call, but a key only sees and only probes the deployments it
may call; `/health/latest`, `/health/history` and `/health/shared-status` are in no non-admin route list
(litellm/proxy/_types.py), so only admins read them.

Two owned proxies: the main one with default `health_check_details`, and a second one with
`health_check_details: false` and `health_check_skip_disabled_background_models: true`. The Rust front door
and auth port must keep all of this, because monitoring systems key off the 503 and the field names.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import object_value, string_value
from tests.integration.routing.dec1_healthchk_support import (
    SLOW_SECONDS,
    UNREACHABLE_API_BASE,
    Rig,
    deployment,
    endpoint_for,
    endpoints,
    health,
    health_rig,
    report,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:6]
HEALTH_TIMEOUT_SECONDS: Final = 2
OK_GROUP: Final = f"hc-ok-{RUN}"
MIXED_GROUP: Final = f"hc-mixed-{RUN}"
AUTH_GROUP: Final = f"hc-auth-{RUN}"
DEAD_GROUP: Final = f"hc-dead-{RUN}"
SLOW_GROUP: Final = f"hc-slow-{RUN}"
OPTOUT_GROUP: Final = f"hc-optout-{RUN}"
OTHER_GROUP: Final = f"hc-other-{RUN}"
OK_MODEL: Final = f"ok-endpoint-{RUN}"
MIXED_OK: Final = f"ok-mixed-{RUN}"
MIXED_BAD: Final = f"status-500-mixed-{RUN}"
AUTH_BAD: Final = f"status-401-auth-{RUN}"
SLOW_MODEL: Final = f"slow-endpoint-{RUN}"
OPTOUT_MODEL: Final = f"ok-optout-{RUN}"
OTHER_MODEL: Final = f"ok-other-{RUN}"
DEAD_ID: Final = f"hc-dead-{RUN}"


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment(OK_GROUP, OK_MODEL, api_base),
        deployment(MIXED_GROUP, MIXED_OK, api_base),
        deployment(MIXED_GROUP, MIXED_BAD, api_base),
        deployment(AUTH_GROUP, AUTH_BAD, api_base),
        deployment(DEAD_GROUP, f"ok-dead-{RUN}", UNREACHABLE_API_BASE, model_info={"id": DEAD_ID}),
        deployment(SLOW_GROUP, SLOW_MODEL, api_base, model_info={"health_check_timeout": HEALTH_TIMEOUT_SECONDS}),
        deployment(OPTOUT_GROUP, OPTOUT_MODEL, api_base, model_info={"disable_background_health_check": True}),
        deployment(OTHER_GROUP, OTHER_MODEL, api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(tmp_path_factory.mktemp("dec1-healthchk-endpoint"), _models) as value:
        yield value


@pytest.fixture(scope="module")
def terse(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-terse"),
        _models,
        general_settings={"health_check_details": False, "health_check_skip_disabled_background_models": True},
    ) as value:
        yield value


def test_a_group_with_one_healthy_and_one_failing_deployment_answers_200_with_both_counts(rig: Rig) -> None:
    """Group `hc-mixed` has a deployment whose upstream answers 200 and one whose upstream answers 500.

    `GET /health?model=hc-mixed` from the master key must answer 200 (one healthy endpoint is enough) with
    `healthy_count: 1` and `unhealthy_count: 1`, and the entries must carry `model_id` values
    `hc-ok-mixed-<run>` (healthy) and `hc-status-500-mixed-<run>` (unhealthy).
    """
    response: Final = health(rig.gateway, model=MIXED_GROUP)
    assert response.status_code == 200, response.text
    body: Final = report(response)
    assert (body["healthy_count"], body["unhealthy_count"]) == (1, 1)
    assert endpoint_for(response, f"hc-{MIXED_OK}")[0] == "healthy"
    assert endpoint_for(response, f"hc-{MIXED_BAD}")[0] == "unhealthy"


def test_an_unhealthy_entry_carries_the_provider_error_text(rig: Rig) -> None:
    """The failing deployment of `hc-mixed` answers 500 with message `scripted 500 from status-500-mixed-<run>`.

    Its unhealthy entry must have an `error` string that contains that provider message, so an operator can
    tell a bad key from an outage without reading logs.
    """
    _, entry = endpoint_for(health(rig.gateway, model=MIXED_GROUP), f"hc-{MIXED_BAD}")
    assert f"scripted 500 from {MIXED_BAD}" in string_value(entry["error"])


@pytest.mark.parametrize(
    ("group", "deployment_id", "status"),
    [(MIXED_GROUP, f"hc-{MIXED_BAD}", 500), (AUTH_GROUP, f"hc-{AUTH_BAD}", 401)],
    ids=["mixed-500", "auth-401"],
)
def test_an_unhealthy_entry_carries_the_provider_status_as_exception_status(
    rig: Rig, group: str, deployment_id: str, status: int
) -> None:
    """A deployment's upstream answers 500, or 401, to every probe.

    Its unhealthy entry must carry `exception_status` equal to that status (500 or 401). The field is what
    `health_check_ignore_transient_errors` and the shared Redis cache read to rebuild the transient filter
    without the exception object (health_check.py:715-731).
    """
    _, entry = endpoint_for(health(rig.gateway, model=group), deployment_id)
    assert entry["exception_status"] == status


def test_an_unreachable_deployment_is_unhealthy_with_a_connection_error(rig: Rig) -> None:
    """Group `hc-dead` has one deployment whose `api_base` is a port nothing listens on.

    `GET /health?model=hc-dead` must answer 503 with the deployment unhealthy and an `error` that names a
    connection failure (contains `Connection` in any case). UNVERIFIED: the exact wording of the
    APIConnectionError text.
    """
    response: Final = health(rig.gateway, model=DEAD_GROUP)
    assert response.status_code == 503, response.text
    kind, entry = endpoint_for(response, DEAD_ID)
    assert kind == "unhealthy"
    assert "connection" in string_value(entry["error"]).lower()


def test_model_targets_a_litellm_params_model_string_when_no_group_has_that_name(rig: Rig) -> None:
    """No group is named `openai/ok-other-<run>`, but deployment `hc-ok-other-<run>` has that
    `litellm_params.model`.

    `GET /health?model=openai/ok-other-<run>` must answer 200 with exactly that deployment healthy and
    `healthy_count: 1`, `unhealthy_count: 0` (`deployments_targeted_by_name` falls back to the provider
    model string).
    """
    response: Final = health(rig.gateway, model=f"openai/{OTHER_MODEL}")
    assert response.status_code == 200, response.text
    assert (report(response)["healthy_count"], report(response)["unhealthy_count"]) == (1, 0)
    assert endpoint_for(response, f"hc-{OTHER_MODEL}")[0] == "healthy"


def test_model_id_checks_one_deployment_of_a_two_deployment_group(rig: Rig) -> None:
    """`GET /health?model_id=hc-status-500-mixed-<run>` targets only the failing deployment of `hc-mixed`.

    It must answer 503 with `healthy_count: 0` and `unhealthy_count: 1`, and the healthy sibling's upstream
    must receive no probe during the call: deployments that share a group are checked separately by id.
    """
    before: Final = rig.upstream.count(MIXED_OK)
    response: Final = health(rig.gateway, model_id=f"hc-{MIXED_BAD}")
    assert response.status_code == 503, response.text
    assert (report(response)["healthy_count"], report(response)["unhealthy_count"]) == (0, 1)
    assert rig.upstream.count(MIXED_OK) == before


def test_an_unknown_model_id_is_404_even_when_paired_with_a_known_model_name(rig: Rig) -> None:
    """The master key sends `GET /health?model=hc-ok-<run>&model_id=dec1-healthchk-no-such-id`.

    It must answer 404 naming `Model with ID dec1-healthchk-no-such-id not found`: `model_id` wins over
    `model` (`_health_endpoint_resolve_target_model_name`), so a typo in the id is never silently answered
    with the group's health.
    """
    response: Final = health(rig.gateway, model=OK_GROUP, model_id="dec1-healthchk-no-such-id")
    assert response.status_code == 404, response.text
    assert "Model with ID dec1-healthchk-no-such-id not found" in response.text


def test_an_unknown_model_name_answers_503_with_zero_counts_and_probes_nothing(rig: Rig) -> None:
    """The master key asks `GET /health?model=dec1-healthchk-no-such-group-<run>`, a name nothing serves.

    It must answer 503 with `healthy_count: 0` and `unhealthy_count: 0`, and the upstream must receive no
    request during the call. UNVERIFIED that a 404 is not preferred here; read from `perform_health_check`
    returning empty lists and the targeted-503 rule.
    """
    before: Final = len(rig.upstream.log)
    response: Final = health(rig.gateway, model=f"dec1-healthchk-no-such-group-{RUN}")
    assert response.status_code == 503, response.text
    assert (report(response)["healthy_count"], report(response)["unhealthy_count"]) == (0, 0)
    assert len(rig.upstream.log) == before


def test_a_probe_past_its_health_check_timeout_is_reported_as_timeout_exceeded(rig: Rig) -> None:
    """Deployment `hc-slow` sets `model_info.health_check_timeout: 2` and its upstream answers after 8 seconds.

    `GET /health?model=hc-slow` must answer 503 in well under 8 seconds with the deployment unhealthy, its
    `error` exactly `Timeout exceeded` and `exception_status: 408` (`run_with_timeout`, health_check.py:297).
    """
    started: Final = time.monotonic()
    response: Final = health(rig.gateway, model=SLOW_GROUP)
    elapsed: Final = time.monotonic() - started
    assert response.status_code == 503, response.text
    assert elapsed < SLOW_SECONDS - 1
    _, entry = endpoint_for(response, f"hc-{SLOW_MODEL}")
    assert entry["error"] == "Timeout exceeded"
    assert entry["exception_status"] == 408


def test_a_slow_deployment_does_not_hold_back_the_report_of_the_others(rig: Rig) -> None:
    """`GET /health` with no model checks every deployment, including the slow one cut off at 2 seconds.

    The answer must arrive in under 8 seconds and list `hc-ok-endpoint-<run>` as healthy: probes run
    concurrently and each has its own timeout, so one hung provider cannot make the whole report time out.
    """
    started: Final = time.monotonic()
    response: Final = health(rig.gateway)
    assert time.monotonic() - started < SLOW_SECONDS - 1
    assert response.status_code == 200, response.text
    assert endpoint_for(response, f"hc-{OK_MODEL}")[0] == "healthy"


def test_an_opted_out_deployment_is_still_probed_on_demand_by_default(rig: Rig) -> None:
    """Deployment `hc-optout` sets `model_info.disable_background_health_check: true`, and the proxy leaves
    `health_check_skip_disabled_background_models` at its default `false`.

    `GET /health?model=hc-optout` must still probe it (the upstream receives a probe) and answer 200 with it
    healthy: the opt-out only covers the background loop unless the operator widens it.
    """
    before: Final = rig.upstream.count(OPTOUT_MODEL)
    response: Final = health(rig.gateway, model=OPTOUT_GROUP)
    assert response.status_code == 200, response.text
    assert rig.upstream.count(OPTOUT_MODEL) > before


def test_a_key_sees_and_probes_only_the_deployments_it_may_call(rig: Rig) -> None:
    """A virtual key is generated with `models: [hc-ok-<run>]` and calls `GET /health` with no model.

    Every entry in the answer must have `model_id` `hc-ok-endpoint-<run>`, and the upstream of `hc-other`
    must receive no probe during the call: a key must not make the proxy spend provider quota on models it
    cannot use.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[OK_GROUP])
        before: Final = rig.upstream.count(OTHER_MODEL)
        response: Final = health(rig.gateway, key=key)
    assert response.status_code == 200, response.text
    ids: Final = {entry.get("model_id") for kind in ("healthy", "unhealthy") for entry in endpoints(response, kind)}
    assert ids == {f"hc-{OK_MODEL}"}
    assert rig.upstream.count(OTHER_MODEL) == before


def test_a_key_asking_for_a_model_it_may_not_call_is_403(rig: Rig) -> None:
    """The same kind of key, limited to `hc-ok-<run>`, asks `GET /health?model=hc-other-<run>`.

    It must answer 403 and probe nothing. The refusal comes from the key's model check in auth, before the
    health endpoint's own `key not allowed to health-check model` branch runs, so the body is the generic
    `key_model_access_denied` error (`litellm/proxy/auth/model_access_denied.py:6`).
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[OK_GROUP])
        before: Final = rig.upstream.count(OTHER_MODEL)
        response: Final = health(rig.gateway, key=key, model=OTHER_GROUP)
    assert response.status_code == 403, response.text
    error: Final = object_value(response.json()["error"])
    assert error.get("type") == "key_model_access_denied", response.text
    assert f"The requested model '{OTHER_GROUP}' is not available for this API key" in response.text
    assert rig.upstream.count(OTHER_MODEL) == before


def test_a_key_asking_for_a_deployment_id_it_may_not_call_is_403(rig: Rig) -> None:
    """The key limited to `hc-ok-<run>` asks `GET /health?model_id=hc-ok-other-<run>`.

    It must answer 403 with `key not allowed to health-check model_id hc-ok-other-<run>`: ids outside the
    caller's scope resolve to nothing instead of being probed.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[OK_GROUP])
        response: Final = health(rig.gateway, key=key, model_id=f"hc-{OTHER_MODEL}")
    assert response.status_code == 403, response.text
    assert f"key not allowed to health-check model_id hc-{OTHER_MODEL}" in response.text


def test_a_view_only_admin_sees_the_api_base_and_gets_no_field_notice(rig: Rig) -> None:
    """A user with role `proxy_admin_viewer` gets a key and calls `GET /health?model=hc-ok-<run>`.

    The answer must not carry `Litellm-Health-Field-Notice`, and the healthy entry must keep `api_base`
    equal to the deployment's: view-only admins are operators who need the routing fields
    (`_PROXY_ADMIN_ROLES`, _health_endpoints.py:1052).
    """
    with rig.gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="proxy_admin_viewer")
        key: Final = scenario.key(user_id=user)
        response: Final = health(rig.gateway, key=key, model=OK_GROUP)
    assert response.status_code == 200, response.text
    assert "litellm-health-field-notice" not in response.headers
    _, entry = endpoint_for(response, f"hc-{OK_MODEL}")
    assert entry["api_base"] == f"{rig.upstream.api_base}"


@pytest.mark.parametrize("path", ["/health/latest", "/health/history", "/health/shared-status"])
def test_health_records_cannot_be_read_without_a_key(rig: Rig, path: str) -> None:
    """`GET` on `/health/latest`, `/health/history` or `/health/shared-status` with no `Authorization` header.

    Each must answer 401: the stored history names every model and its provider errors.
    """
    response: Final = rig.gateway.client.get(path)
    assert response.status_code == 401, response.text


@pytest.mark.parametrize("path", ["/health/latest", "/health/history"])
def test_health_records_are_admin_only_for_a_plain_virtual_key(rig: Rig, path: str) -> None:
    """A virtual key with no user and no role calls `/health/latest` or `/health/history`.

    It must be refused with 401 and the message `Only proxy admin can be used`: neither route is in the info
    or LLM route lists (litellm/proxy/_types.py:609), unlike `/health` itself, so the auth port must not
    widen them. UNVERIFIED that the status is 401 rather than 403 (`_raise_admin_only_route_exception`).
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = rig.gateway.request("GET", path, key=key)
    assert response.status_code == 401, response.text
    assert "Only proxy admin can be used" in response.text


def test_shared_status_says_sharing_is_off_when_use_shared_health_check_is_unset(rig: Rig) -> None:
    """The proxy runs without `general_settings.use_shared_health_check`.

    The master key's `GET /health/shared-status` must answer 200 with exactly
    `{"shared_health_check_enabled": false, "message": "Shared health check is not enabled"}`.
    """
    response: Final = rig.gateway.request("GET", "/health/shared-status")
    assert response.status_code == 200, response.text
    assert report(response) == {"shared_health_check_enabled": False, "message": "Shared health check is not enabled"}


@pytest.mark.parametrize("params", [{"limit": "0"}, {"limit": "1001"}, {"offset": "-1"}])
def test_health_history_rejects_out_of_range_paging(rig: Rig, params: dict[str, str]) -> None:
    """The master key calls `GET /health/history` with `limit=0`, `limit=1001` or `offset=-1`.

    Each must answer 422: `limit` is bounded to 1..1000 and `offset` to 0 or more by the route signature, so a
    client cannot ask the database for an unbounded scan.
    """
    response: Final = rig.gateway.request("GET", "/health/history", params=params)
    assert response.status_code == 422, response.text


def test_without_details_a_healthy_entry_carries_only_the_model_and_its_id(terse: Rig) -> None:
    """The second proxy sets `general_settings.health_check_details: false`.

    The healthy entry of `GET /health?model=hc-ok-<run>` must have exactly the keys `model` and `model_id`:
    `clean_endpoint_data` keeps only `MINIMAL_DISPLAY_PARAMS` (health_check.py:217), so no `api_base` and no
    rate limit headers.
    """
    response: Final = health(terse.gateway, model=OK_GROUP)
    assert response.status_code == 200, response.text
    (entry,) = endpoints(response, "healthy")
    assert set(entry) == {"model", "model_id"}


def test_without_details_an_unhealthy_entry_drops_the_error_text_but_keeps_the_status(terse: Rig) -> None:
    """With `health_check_details: false`, the failing deployment of `hc-mixed` is checked.

    Its unhealthy entry must have no `error` key and must carry `exception_status: 500`, with keys exactly
    `model`, `model_id` and `exception_status`. Read from code: the status is added after cleaning
    (health_check.py:718-724), the error is not.
    """
    _, entry = endpoint_for(health(terse.gateway, model=MIXED_GROUP), f"hc-{MIXED_BAD}")
    assert set(entry) == {"model", "model_id", "exception_status"}
    assert entry["exception_status"] == 500


def test_skip_disabled_models_leaves_an_opted_out_deployment_out_of_on_demand_checks(terse: Rig) -> None:
    """The second proxy sets `health_check_skip_disabled_background_models: true`; deployment `hc-optout`
    sets `disable_background_health_check: true`.

    `GET /health?model=hc-optout` must answer 503 with `healthy_count: 0` and `unhealthy_count: 0`, and its
    upstream must receive no probe: the opt-out now covers `GET /health` too.
    """
    before: Final = terse.upstream.count(OPTOUT_MODEL)
    response: Final = health(terse.gateway, model=OPTOUT_GROUP)
    assert response.status_code == 503, response.text
    assert (report(response)["healthy_count"], report(response)["unhealthy_count"]) == (0, 0)
    assert terse.upstream.count(OPTOUT_MODEL) == before


def test_skip_disabled_models_also_drops_the_opted_out_deployment_from_a_full_report(terse: Rig) -> None:
    """On the same proxy the master key calls `GET /health` with no model.

    No entry may have `model_id` `hc-ok-optout-<run>`, while `hc-ok-endpoint-<run>` is listed healthy.
    """
    response: Final = health(terse.gateway)
    ids: Final = {entry.get("model_id") for kind in ("healthy", "unhealthy") for entry in endpoints(response, kind)}
    assert f"hc-{OPTOUT_MODEL}" not in ids
    assert f"hc-{OK_MODEL}" in ids
