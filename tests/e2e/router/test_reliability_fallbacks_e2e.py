"""Live e2e: per-request fallbacks reroute a failing deployment's traffic to a
healthy one.

Each test registers a primary deployment that fails (an unreachable base URL, or
a 1ms deadline) and calls it with a `router_settings_override` mapping it to the
real `gpt-5.5`. The proof the fallback fired is twofold: the response is a real
completion from `gpt-5.5` (a non-empty content string), and the proxy reports at
least one attempted fallback in the x-litellm-attempted-fallbacks header.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import RouterSettingsOverride
from reliability_support import (
    chat_override,
    content_of,
    create_bad_base_deployment,
    create_timeout_deployment,
)

pytestmark = pytest.mark.e2e


def _assert_served_by_fallback(resp: StreamingResponse) -> None:
    assert resp.status_code == 200, f"expected 200 after fallback, got {resp.status_code}: {resp.body[:300]}"
    content = content_of(resp)
    assert isinstance(content, str) and content, (
        f"the gpt-5.5 fallback should have returned a real completion, got content {content!r} "
        f"(body={resp.body[:300]})"
    )
    attempted = resp.headers.get("x-litellm-attempted-fallbacks")
    assert attempted is not None, "response is missing the x-litellm-attempted-fallbacks header"
    assert int(attempted) >= 1, f"x-litellm-attempted-fallbacks should be >= 1, got {attempted!r}"


class TestReliabilityFallbacks:
    @pytest.mark.covers("reliability.fallback.5xx.routes_to_fallback")
    def test_5xx_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-fail-{unique_marker()}"
        model_id = create_bad_base_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy, scoped_key, primary, "say hi",
            override=RouterSettingsOverride(fallbacks=[{primary: ["gpt-5.5"]}]),
        )
        _assert_served_by_fallback(resp)

    @pytest.mark.covers("reliability.fallback.timeout.routes_to_fallback")
    def test_timeout_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-tofail-{unique_marker()}"
        model_id = create_timeout_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy, scoped_key, primary, "say hi",
            override=RouterSettingsOverride(fallbacks=[{primary: ["gpt-5.5"]}]),
        )
        _assert_served_by_fallback(resp)
