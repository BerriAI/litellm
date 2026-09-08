"""Live e2e: a request that fails on its first deployment is retried inside its own
model group and still comes back a completion.

The model group is a pair: an always-timing-out deployment that holds all of the
group's shuffle weight, and a healthy backup at weight 0. The weighted pick always
opens on the timing-out one, its first Timeout benches it (an
`allowed_fails_policy` of `TimeoutErrorAllowedFails: 0`), and the retry falls
through to the only deployment left. So the customer sees a completion and the
proxy reports that it took a retry to get there, with no random first pick in the
middle of it.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import RouterSettingsOverride
from reliability_support import (
    chat_override,
    completion_tokens_of,
    content_of,
    create_always_timing_out_deployment,
    create_zero_weight_backup_deployment,
    finish_reason_of,
)

pytestmark = pytest.mark.e2e


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

        resp = chat_override(
            client.proxy,
            scoped_key,
            group,
            f"say hi {unique_marker()}",
            override=RouterSettingsOverride(num_retries=2),
        )

        assert resp.status_code == 200, (
            f"the retry should have landed on the healthy backup, got {resp.status_code}: {resp.body[:300]}"
        )

        attempted = resp.headers.get("x-litellm-attempted-retries")
        assert attempted is not None, "response is missing the x-litellm-attempted-retries header"
        assert int(attempted) >= 1, (
            f"x-litellm-attempted-retries is {attempted!r}; a 200 with no retry means the request never "
            "opened on the timing-out deployment, so this proves nothing about retries"
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
