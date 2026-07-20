"""Live e2e: the response cache returns a cached answer on an exact repeat.

One mock deployment answers with a unique marker. The same unique prompt is sent
twice under the same key: the first call is a cache miss (the proxy computes and
stores the entry, and returns no x-litellm-cache-key), the second is an exact hit
(the proxy serves from cache and returns x-litellm-cache-key). This relies on the
standard Redis response cache being enabled on the proxy under test.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from lifecycle import ResourceManager
from reliability_client import ReliabilityClient, content_of

pytestmark = pytest.mark.e2e


class TestCache:
    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    def test_exact_cache_returns_cached(
        self, client: ReliabilityClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker = unique_marker()
        name, cached = f"reliability-cache-{marker}", f"cacheable-{marker}"
        model_id = client.create_mock(name, cached)
        resources.defer(lambda: client.proxy.delete_model(model_id))
        prompt = f"cache probe {marker}"

        first = client.chat_override(scoped_key, name, prompt)
        assert first.status_code == 200, f"first call should succeed, got {first.status_code}: {first.body[:300]}"
        assert content_of(first) == cached, f"first call should return {cached!r}, got {content_of(first)!r}"
        assert "x-litellm-cache-key" not in first.headers, (
            "first (uncached) call must not report a cache-key header"
        )

        second = client.chat_override(scoped_key, name, prompt)
        assert second.status_code == 200, f"second call should succeed, got {second.status_code}: {second.body[:300]}"
        assert content_of(second) == cached, f"second call should return {cached!r}, got {content_of(second)!r}"
        assert "x-litellm-cache-key" in second.headers, (
            "second identical call should hit the response cache and report a cache-key header "
            "(requires the proxy's Redis response cache to be enabled)"
        )
