"""Live e2e: the response cache returns a cached answer on an exact repeat.

The same unique prompt is sent twice to the real `gpt-5.5` deployment under the
same key: the first call is a cache miss (the proxy computes and stores the entry,
and returns no x-litellm-cache-key), the second is an exact hit (the proxy serves
from cache and returns x-litellm-cache-key). This relies on the standard Redis
response cache being enabled on the proxy under test.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from reliability_support import chat_override

pytestmark = pytest.mark.e2e


class TestReliabilityCache:
    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    def test_exact_cache_returns_cached(self, client: ComplexityRouterClient, scoped_key: str) -> None:
        prompt = f"cache probe {unique_marker()}"

        first = chat_override(client.proxy, scoped_key, "gpt-5.5", prompt)
        assert first.status_code == 200, f"first call should succeed, got {first.status_code}: {first.body[:300]}"
        assert "x-litellm-cache-key" not in first.headers, (
            "first (uncached) call must not report a cache-key header"
        )

        second = chat_override(client.proxy, scoped_key, "gpt-5.5", prompt)
        assert second.status_code == 200, f"second call should succeed, got {second.status_code}: {second.body[:300]}"
        assert "x-litellm-cache-key" in second.headers, (
            "second identical call should hit the response cache and report a cache-key header "
            "(requires the proxy's Redis response cache to be enabled)"
        )
