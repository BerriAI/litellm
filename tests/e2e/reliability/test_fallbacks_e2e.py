"""Live e2e: per-request fallbacks reroute a failing deployment's traffic to a
healthy one.

Each test registers a `fail` deployment that raises (or times out) and an `ok`
deployment whose mock_response is a unique served-by marker, then calls the `fail`
model with a `router_settings_override` that maps it to `ok`. The proof the
fallback fired is twofold: the response content is the `ok` deployment's marker
(so `ok`, not `fail`, produced the answer), and the proxy reports at least one
attempted fallback in the x-litellm-attempted-fallbacks header.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import RouterSettingsOverride
from reliability_client import ReliabilityClient, content_of

pytestmark = pytest.mark.e2e


def _mock(client: ReliabilityClient, resources: ResourceManager, name: str, mock_response: str) -> None:
    model_id = client.create_mock(name, mock_response)
    resources.defer(lambda: client.proxy.delete_model(model_id))


def _timeout(client: ReliabilityClient, resources: ResourceManager, name: str) -> None:
    model_id = client.create_timeout_deployment(name)
    resources.defer(lambda: client.proxy.delete_model(model_id))


def _assert_served_by_fallback(resp: StreamingResponse, served: str) -> None:
    assert resp.status_code == 200, f"expected 200 after fallback, got {resp.status_code}: {resp.body[:300]}"
    content = content_of(resp)
    assert content == served, (
        f"the fallback backend should have served {served!r}, got content {content!r} (body={resp.body[:300]})"
    )
    attempted = resp.headers.get("x-litellm-attempted-fallbacks")
    assert attempted is not None, "response is missing the x-litellm-attempted-fallbacks header"
    assert int(attempted) >= 1, f"x-litellm-attempted-fallbacks should be >= 1, got {attempted!r}"


class TestFallbacks:
    @pytest.mark.covers("reliability.fallback.5xx.routes_to_fallback")
    def test_5xx_routes_to_fallback(
        self, client: ReliabilityClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker = unique_marker()
        fail, ok = f"reliability-fail-{marker}", f"reliability-ok-{marker}"
        served = f"served-by-{marker}"
        _mock(client, resources, fail, "litellm.InternalServerError")
        _mock(client, resources, ok, served)

        resp = client.chat_override(
            scoped_key, fail, "hello", override=RouterSettingsOverride(fallbacks=[{fail: [ok]}])
        )
        _assert_served_by_fallback(resp, served)

    @pytest.mark.covers("reliability.fallback.context_window.routes_to_fallback")
    def test_context_window_routes_to_fallback(
        self, client: ReliabilityClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker = unique_marker()
        fail, ok = f"reliability-cwfail-{marker}", f"reliability-cwok-{marker}"
        served = f"served-by-{marker}"
        _mock(client, resources, fail, "litellm.ContextWindowExceededError")
        _mock(client, resources, ok, served)

        resp = client.chat_override(
            scoped_key, fail, "hello", override=RouterSettingsOverride(context_window_fallbacks=[{fail: [ok]}])
        )
        _assert_served_by_fallback(resp, served)

    @pytest.mark.covers("reliability.fallback.timeout.routes_to_fallback")
    def test_timeout_routes_to_fallback(
        self, client: ReliabilityClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker = unique_marker()
        fail, ok = f"reliability-tofail-{marker}", f"reliability-took-{marker}"
        served = f"served-by-{marker}"
        _timeout(client, resources, fail)
        _mock(client, resources, ok, served)

        resp = client.chat_override(
            scoped_key, fail, "hello", override=RouterSettingsOverride(fallbacks=[{fail: [ok]}])
        )
        _assert_served_by_fallback(resp, served)
