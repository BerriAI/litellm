"""Live e2e: a per-request timeout surfaces to the caller instead of hanging.

A deployment created with a 1ms deadline always exceeds it against the real
backend. With no fallback in play, the proxy must return the timeout to the
caller: a 408 for a non-streamed request, and the same timeout surfaced on the
streamed path (either a 408 before the stream opens or a timeout error carried in
the response).
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from lifecycle import ResourceManager
from reliability_support import chat_override, create_timeout_deployment

pytestmark = pytest.mark.e2e


class TestReliabilityTimeouts:
    @pytest.mark.covers("reliability.timeout.request_timeout.exceeds_deadline")
    def test_request_timeout_exceeds_deadline(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"reliability-timeout-{unique_marker()}"
        model_id = create_timeout_deployment(client.proxy, name)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(client.proxy, scoped_key, name, "hello")
        assert resp.status_code == 408, (
            f"a timed-out request should return 408, got {resp.status_code}: {resp.body[:300]}"
        )
        assert "timeout" in resp.body.lower(), f"the 408 body should name the timeout, got: {resp.body[:300]}"

    @pytest.mark.covers("reliability.timeout.stream_timeout.exceeds_deadline")
    def test_stream_timeout_exceeds_deadline(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"reliability-stream-timeout-{unique_marker()}"
        model_id = create_timeout_deployment(client.proxy, name)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(client.proxy, scoped_key, name, "hello", stream=True)
        surfaced = f"{resp.body} {resp.stream_error or ''}".lower()
        assert resp.status_code >= 400, (
            f"a timed-out streaming request should surface an error status, got {resp.status_code}: {resp.body[:300]}"
        )
        assert "timeout" in surfaced, (
            f"the streamed timeout error should name the timeout, got body={resp.body[:300]}, "
            f"stream_error={resp.stream_error!r}"
        )
