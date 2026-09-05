"""Live e2e: a deployment that fails is benched for its cooldown and comes back
once the cooldown lapses.

Every model group is the same pair: a deployment that always fails in one specific
way (a 500, a 429, a 401, or a timeout) holding all of the group's shuffle weight,
with an `allowed_fails_policy` of zero for that error class and a short
`cooldown_time`, plus a healthy backup at weight 0. The first call, retries off,
surfaces the failure to the customer as-is and benches the deployment. The next
call, still inside the cooldown, lands on the backup, which the proxy names in
x-litellm-model-id. Then the test polls until the weighted shuffle opens on the
failing deployment again and the same failure comes back: that is the recovery,
since a benched deployment is one the router will try again, not one it forgot.

The failures are the same real ones the retry tests use: a 1ms deadline and a
bogus key on the real backend, and this proxy standing in as the upstream for
the 500 (fronting a group whose only deployment is unreachable) and the 429
(fronting a healthy group with a key that already spent its one request per
minute).
"""

from __future__ import annotations

import time

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
)

pytestmark = pytest.mark.e2e

RECOVERY_GRACE_SECONDS = 10


def _call_without_retries(client: ComplexityRouterClient, key: str, group: str) -> StreamingResponse:
    return chat_override(
        client.proxy, key, group, f"say hi {unique_marker()}", override=RouterSettingsOverride(num_retries=0)
    )


def _assert_trips_then_recovers(
    client: ComplexityRouterClient, key: str, group: str, backup: str, failure_status: int
) -> None:
    tripped = _call_without_retries(client, key, group)
    assert tripped.status_code == failure_status, (
        f"the first call should have surfaced the deployment's own {failure_status}, got {tripped.status_code}: "
        f"{tripped.body[:300]}"
    )

    benched = _call_without_retries(client, key, group)
    assert benched.status_code == 200, (
        f"inside the cooldown the group should have served from the backup, got {benched.status_code}: "
        f"{benched.body[:300]}"
    )
    assert model_id_of(benched) == backup, (
        f"inside the cooldown the proxy should have named the backup {backup} in x-litellm-model-id, "
        f"got {model_id_of(benched)!r}"
    )

    for _ in range(int(COOLDOWN_SECONDS) + RECOVERY_GRACE_SECONDS):
        time.sleep(1)
        if _call_without_retries(client, key, group).status_code == failure_status:
            return
    pytest.fail(
        f"{group} never sent traffic back to its benched deployment within "
        f"{COOLDOWN_SECONDS + RECOVERY_GRACE_SECONDS}s, so the cooldown never lapsed"
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

        _assert_trips_then_recovers(client, scoped_key, group, backup, failure_status=500)

    @pytest.mark.covers("reliability.cooldown.429.trips_then_recovers")
    def test_429_trips_cooldown_then_recovers(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        spent_key = client.proxy.generate_key(
            KeyGenerateBody(models=[CHEAP_OPENAI_MODEL], rpm_limit=1, user_id="e2e-test-user")
        )
        resources.defer(lambda: client.proxy.delete_key(spent_key))
        primed = chat_override(client.proxy, spent_key, CHEAP_OPENAI_MODEL, f"say hi {unique_marker()}")
        assert primed.status_code == 200, (
            f"the one request the rpm-limited key allows should have succeeded, got {primed.status_code}: "
            f"{primed.body[:300]}"
        )

        group = f"reliability-cooldown-429-{unique_marker()}"
        failing = create_always_rate_limited_deployment(
            client.proxy, group, CHEAP_OPENAI_MODEL, spent_key, cooldown_time=COOLDOWN_SECONDS
        )
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_trips_then_recovers(client, scoped_key, group, backup, failure_status=429)

    @pytest.mark.covers("reliability.cooldown.auth.trips_then_recovers")
    def test_auth_failure_trips_cooldown_then_recovers(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-cooldown-auth-{unique_marker()}"
        failing = create_always_unauthorized_deployment(client.proxy, group, cooldown_time=COOLDOWN_SECONDS)
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_trips_then_recovers(client, scoped_key, group, backup, failure_status=401)

    @pytest.mark.covers("reliability.cooldown.timeout.trips_then_recovers")
    def test_timeout_trips_cooldown_then_recovers(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-cooldown-timeout-{unique_marker()}"
        failing = create_always_timing_out_deployment(client.proxy, group, cooldown_time=COOLDOWN_SECONDS)
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_trips_then_recovers(client, scoped_key, group, backup, failure_status=408)
