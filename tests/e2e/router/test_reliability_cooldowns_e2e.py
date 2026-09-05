"""Live e2e: a deployment that fails is benched for its cooldown and comes back
once the cooldown lapses.

Every model group is the same pair: a deployment that always fails in one specific
way (a 500, a 429, a 401, or a timeout) holding all of the group's shuffle weight,
with an `allowed_fails_policy` of zero for that error class and a short
`cooldown_time`, plus a healthy backup at weight 0. The first call, retries off,
surfaces the failure to the customer as-is and benches the deployment. The proxy
records the bench off the request path, and a sibling replica that checked Redis
for that deployment just before the bench landed keeps sending it traffic until
it looks again, which it does at most every 10s
(litellm.default_redis_batch_cache_expiry). So for REPLICA_PROPAGATION_SECONDS
after the trip every answer has to be either the deployment's own failure or a
200 from the backup, which the proxy names in x-litellm-model-id, and at least
one replica has to have served from the backup by then. From then until shortly
before the cooldown can lapse, every call has to land on the backup whichever
replica takes it. Then the test polls until the weighted shuffle opens on the
failing deployment again and the same failure comes back (or, for the 429 pair,
its own 200 once the key's rpm window has reset): that is the recovery, since a
benched deployment is one the router will try again, not one it forgot. Its
deadline counts from the last failure a stale replica caused, because every
failure re-arms the cooldown.

The failures are the same real ones the retry tests use: a 1ms deadline and a
bogus key on the real backend, and this proxy standing in as the upstream for
the 500 (fronting a group whose only deployment is unreachable) and the 429
(fronting a healthy group with a key whose one request per minute is spent right
before the trip, so its window outlasts the bench).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import KeyGenerateBody, RouterSettingsOverride
from reliability_support import (
    COOLDOWN_SECONDS,
    chat_override,
    create_always_5xx_deployment,
    create_always_rate_limited_deployment,
    create_always_timing_out_deployment,
    create_always_unauthorized_deployment,
    create_bad_base_deployment,
    create_zero_weight_backup_deployment,
    model_id_of,
    spend_only_request_of,
)

pytestmark = pytest.mark.e2e

RECOVERY_GRACE_SECONDS = 10
REPLICA_PROPAGATION_SECONDS = 12.0
PROPAGATION_POLL_SECONDS = 0.25
BENCH_MARGIN_SECONDS = 4.0


def _call_without_retries(client: ComplexityRouterClient, key: str, group: str) -> StreamingResponse:
    return chat_override(
        client.proxy, key, group, f"say hi {unique_marker()}", override=RouterSettingsOverride(num_retries=0)
    )


def _assert_served_by_backup(resp: StreamingResponse, backup: str, when: str) -> None:
    assert resp.status_code == 200, (
        f"{when} the group should have served from the backup, got {resp.status_code}: {resp.body[:300]}"
    )
    assert model_id_of(resp) == backup, (
        f"{when} the proxy should have named the backup {backup} in x-litellm-model-id, got {model_id_of(resp)!r}"
    )


def _answers_while_replicas_catch_up(
    client: ComplexityRouterClient, key: str, group: str, tripped_at: float
) -> Iterator[tuple[float, StreamingResponse]]:
    while time.monotonic() < tripped_at + REPLICA_PROPAGATION_SECONDS:
        resp = _call_without_retries(client, key, group)
        yield time.monotonic() - tripped_at, resp
        time.sleep(PROPAGATION_POLL_SECONDS)


def _backup_sighting(resp: StreamingResponse, elapsed: float, backup: str, failure_status: int) -> float | None:
    if resp.status_code == 200:
        _assert_served_by_backup(resp, backup, f"{elapsed:.1f}s after the trip")
        return elapsed
    assert resp.status_code == failure_status, (
        f"{elapsed:.1f}s after the trip the group answered {resp.status_code}, neither the deployment's own "
        f"{failure_status} nor a 200 from the backup: {resp.body[:300]}"
    )
    return None


@dataclass(frozen=True, slots=True)
class _Propagation:
    first_backup_at: float
    last_failure_at: float


def _propagation_of(
    client: ComplexityRouterClient, key: str, group: str, backup: str, failure_status: int, tripped_at: float
) -> _Propagation:
    sightings = tuple(
        (elapsed, _backup_sighting(resp, elapsed, backup, failure_status))
        for elapsed, resp in _answers_while_replicas_catch_up(client, key, group, tripped_at)
    )
    backups = tuple(elapsed for elapsed, backup_at in sightings if backup_at is not None)
    assert backups, (
        f"no replica served {group} from the backup within {REPLICA_PROPAGATION_SECONDS:.0f}s of the trip, so the "
        "cooldown never became visible"
    )
    return _Propagation(
        first_backup_at=backups[0],
        last_failure_at=max((elapsed for elapsed, backup_at in sightings if backup_at is None), default=0.0),
    )


def _reached_benched_deployment(resp: StreamingResponse, failing: str, failure_status: int) -> bool:
    return resp.status_code == failure_status or model_id_of(resp) == failing


def _assert_trips_then_recovers(
    client: ComplexityRouterClient, key: str, group: str, failing: str, backup: str, failure_status: int
) -> None:
    tripped_at = time.monotonic()
    tripped = _call_without_retries(client, key, group)
    assert tripped.status_code == failure_status, (
        f"the first call should have surfaced the deployment's own {failure_status}, got {tripped.status_code}: "
        f"{tripped.body[:300]}"
    )

    propagation = _propagation_of(client, key, group, backup, failure_status, tripped_at)

    bench_until = tripped_at + COOLDOWN_SECONDS - BENCH_MARGIN_SECONDS
    while time.monotonic() < bench_until:
        _assert_served_by_backup(
            _call_without_retries(client, key, group),
            backup,
            f"{time.monotonic() - tripped_at:.1f}s into a {COOLDOWN_SECONDS:.0f}s cooldown that became visible "
            f"after {propagation.first_backup_at:.1f}s,",
        )

    recovery_deadline = tripped_at + propagation.last_failure_at + COOLDOWN_SECONDS + RECOVERY_GRACE_SECONDS
    while time.monotonic() < recovery_deadline:
        time.sleep(1)
        if _reached_benched_deployment(_call_without_retries(client, key, group), failing, failure_status):
            return
    pytest.fail(
        f"{group} never sent traffic back to its benched deployment within "
        f"{COOLDOWN_SECONDS + RECOVERY_GRACE_SECONDS:.0f}s of its last failure, so the cooldown never lapsed"
    )


class TestReliabilityCooldowns:
    @pytest.mark.covers("reliability.cooldown.5xx.trips_then_recovers")
    def test_5xx_trips_cooldown_then_recovers(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        upstream = f"reliability-cooldown-5xx-upstream-{unique_marker()}"
        upstream_id = create_bad_base_deployment(client.proxy, upstream)
        resources.defer(lambda: client.proxy.delete_model(upstream_id))

        group = f"reliability-cooldown-5xx-{unique_marker()}"
        failing = create_always_5xx_deployment(
            client.proxy, group, upstream, scoped_key, cooldown_time=COOLDOWN_SECONDS
        )
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_trips_then_recovers(client, scoped_key, group, failing, backup, failure_status=500)

    @pytest.mark.covers("reliability.cooldown.429.trips_then_recovers")
    def test_429_trips_cooldown_then_recovers(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        spent_key = client.proxy.generate_key(
            KeyGenerateBody(models=[CHEAP_OPENAI_MODEL], rpm_limit=1, user_id="e2e-test-user")
        )
        resources.defer(lambda: client.proxy.delete_key(spent_key))

        group = f"reliability-cooldown-429-{unique_marker()}"
        failing = create_always_rate_limited_deployment(
            client.proxy, group, CHEAP_OPENAI_MODEL, spent_key, cooldown_time=COOLDOWN_SECONDS
        )
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        spend_only_request_of(client.proxy, spent_key)
        _assert_trips_then_recovers(client, scoped_key, group, failing, backup, failure_status=429)

    @pytest.mark.covers("reliability.cooldown.auth.trips_then_recovers")
    def test_auth_failure_trips_cooldown_then_recovers(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-cooldown-auth-{unique_marker()}"
        failing = create_always_unauthorized_deployment(client.proxy, group, cooldown_time=COOLDOWN_SECONDS)
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_trips_then_recovers(client, scoped_key, group, failing, backup, failure_status=401)

    @pytest.mark.covers("reliability.cooldown.timeout.trips_then_recovers")
    def test_timeout_trips_cooldown_then_recovers(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-cooldown-timeout-{unique_marker()}"
        failing = create_always_timing_out_deployment(client.proxy, group, cooldown_time=COOLDOWN_SECONDS)
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_trips_then_recovers(client, scoped_key, group, failing, backup, failure_status=408)
