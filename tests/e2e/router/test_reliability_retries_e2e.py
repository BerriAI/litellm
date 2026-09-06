"""Live e2e: a request that fails on its first deployment is retried inside its own
model group and still comes back a completion.

Every model group is a pair: a deployment that always fails in one specific way
and holds all of the group's shuffle weight, and a healthy backup at weight 0.
The weighted pick always opens on the failing one, its first failure benches it
(an `allowed_fails_policy` of zero for that error class), and the retry falls
through to the only deployment left. So the customer sees a completion and the
proxy reports that it took a retry to get there, with no random first pick in
the middle of it.

The failures are real. A timeout is a 1ms deadline on the real backend and a 401
is a bogus key on it. A 500 and a 429 come from this same proxy standing in as
the upstream: the failing deployment fronts a group of this proxy whose only
deployment is unreachable (a real 500), or a healthy group called with a key that
has already spent its one request per minute (a real 429), so the router sees the
same statuses a customer's provider would send.

The context-window retry cell has no test on purpose: the router refuses to
retry a 400-class error, and a context-window refusal is one, so the documented
`ContextWindowExceededErrorRetries` policy never fires. That row stays uncovered
until the product either retries it or drops it from the docs.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import KeyGenerateBody, RouterSettingsOverride
from reliability_support import (
    chat_override,
    completion_tokens_of,
    content_of,
    create_always_5xx_deployment,
    create_always_rate_limited_deployment,
    create_always_timing_out_deployment,
    create_always_unauthorized_deployment,
    create_bad_base_deployment,
    create_zero_weight_backup_deployment,
    finish_reason_of,
    spend_only_request_of,
)

pytestmark = pytest.mark.e2e


def _assert_served_after_retry(resp: StreamingResponse) -> None:
    assert resp.status_code == 200, (
        f"the retry should have landed on the healthy backup, got {resp.status_code}: {resp.body[:300]}"
    )

    attempted = resp.headers.get("x-litellm-attempted-retries")
    assert attempted is not None, "response is missing the x-litellm-attempted-retries header"
    assert int(attempted) >= 1, (
        f"x-litellm-attempted-retries is {attempted!r}; a 200 with no retry means the request never "
        "opened on the failing deployment, so this proves nothing about retries"
    )

    content = content_of(resp)
    finish_reason = finish_reason_of(resp)
    completion_tokens = completion_tokens_of(resp) or 0
    assert isinstance(content, str), (
        f"the retry should have returned a completion body, got content {content!r} (body={resp.body[:300]})"
    )
    assert content or (finish_reason == "length" and completion_tokens > 0), (
        f"the retry returned empty content with finish_reason={finish_reason!r}, "
        f"completion_tokens={completion_tokens}; empty content is only acceptable when the budget "
        f"was spent on non-visible reasoning (body={resp.body[:300]})"
    )


def _retry_once(client: ComplexityRouterClient, key: str, group: str) -> StreamingResponse:
    return chat_override(
        client.proxy, key, group, f"say hi {unique_marker()}", override=RouterSettingsOverride(num_retries=2)
    )


class TestReliabilityRetries:
    @pytest.mark.covers("reliability.retry.timeout.succeeds_within_retries")
    def test_timeout_on_first_deployment_succeeds_on_retry(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-retry-{unique_marker()}"
        timing_out = create_always_timing_out_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(timing_out))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_served_after_retry(_retry_once(client, scoped_key, group))

    @pytest.mark.covers("reliability.retry.5xx.succeeds_within_retries")
    def test_5xx_on_first_deployment_succeeds_on_retry(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        upstream = f"reliability-5xx-upstream-{unique_marker()}"
        upstream_id = create_bad_base_deployment(client.proxy, upstream)
        resources.defer(lambda: client.proxy.delete_model(upstream_id))

        group = f"reliability-retry-5xx-{unique_marker()}"
        failing = create_always_5xx_deployment(client.proxy, group, upstream, scoped_key)
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_served_after_retry(_retry_once(client, scoped_key, group))

    @pytest.mark.covers("reliability.retry.429.succeeds_within_retries")
    def test_429_on_first_deployment_succeeds_on_retry(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        spent_key = client.proxy.generate_key(
            KeyGenerateBody(models=[CHEAP_OPENAI_MODEL], rpm_limit=1, user_id="e2e-test-user")
        )
        resources.defer(lambda: client.proxy.delete_key(spent_key))

        group = f"reliability-retry-429-{unique_marker()}"
        failing = create_always_rate_limited_deployment(client.proxy, group, CHEAP_OPENAI_MODEL, spent_key)
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        spend_only_request_of(client.proxy, spent_key)
        _assert_served_after_retry(_retry_once(client, scoped_key, group))

    @pytest.mark.covers("reliability.retry.auth.succeeds_within_retries")
    def test_auth_failure_on_first_deployment_succeeds_on_retry(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-retry-auth-{unique_marker()}"
        failing = create_always_unauthorized_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(failing))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        _assert_served_after_retry(_retry_once(client, scoped_key, group))
