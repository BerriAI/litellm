"""
Test for preset_cache_key multiple values bug fix.

This test verifies that get_cache_key doesn't raise TypeError when kwargs
already contains preset_cache_key.

Issue: When get_cache_key(**kwargs) is called with kwargs containing
preset_cache_key, the call to _set_preset_cache_key_in_kwargs() would fail with:
    TypeError: got multiple values for keyword argument 'preset_cache_key'
"""

import asyncio

import pytest


class TestPresetCacheKeyFix:
    """Tests for the preset_cache_key multiple values fix."""

    def test_get_cache_key_with_preset_cache_key_in_kwargs(self):
        """
        Test that get_cache_key handles kwargs that already contain preset_cache_key.

        This was causing:
        TypeError: _set_preset_cache_key_in_kwargs() got multiple values
        for keyword argument 'preset_cache_key'
        """
        from litellm.caching.caching import Cache

        cache = Cache()

        # Simulate kwargs that already has preset_cache_key (as can happen
        # when the cache key is recomputed in certain code paths)
        kwargs_with_preset = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "preset_cache_key": "existing_key_12345",  # This caused the bug
            "litellm_params": {},
        }

        # This should NOT raise TypeError
        try:
            result = cache.get_cache_key(**kwargs_with_preset)
            assert result is not None
            assert isinstance(result, str)
        except TypeError as e:
            if "multiple values for keyword argument" in str(e):
                pytest.fail(f"Bug not fixed: {e}")
            raise

    def test_get_cache_key_without_preset_cache_key(self):
        """Test normal case without preset_cache_key in kwargs still works."""
        from litellm.caching.caching import Cache

        cache = Cache()

        kwargs_normal = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "litellm_params": {},
        }

        result = cache.get_cache_key(**kwargs_normal)
        assert result is not None
        assert isinstance(result, str)

    def test_preset_cache_key_is_set_in_litellm_params(self):
        """Verify that preset_cache_key is correctly set in litellm_params."""
        from litellm.caching.caching import Cache

        cache = Cache()

        litellm_params = {}
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "litellm_params": litellm_params,
        }

        result = cache.get_cache_key(**kwargs)

        # The method should set preset_cache_key in litellm_params
        assert "preset_cache_key" in litellm_params
        assert litellm_params["preset_cache_key"] == result

    def test_authenticated_cache_keys_are_isolated_by_server_auth_identity(self):
        """Authenticated callers must not share an exact-response cache entry."""
        from litellm.caching.caching import Cache
        from litellm.proxy._types import UserAPIKeyAuth

        cache = Cache()
        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
        caller_a = {"user_api_key_auth": UserAPIKeyAuth(token="validated-token-a")}
        caller_b = {"user_api_key_auth": UserAPIKeyAuth(token="validated-token-b")}

        key_a = cache.get_cache_key(**request, metadata=caller_a)
        key_b = cache.get_cache_key(**request, metadata=caller_b)
        key_a_again = cache.get_cache_key(**request, metadata=caller_a)

        assert key_a is not None
        assert key_b is not None
        assert key_a != key_b
        assert key_a_again == key_a

    def test_authenticated_cache_without_stable_token_bypasses_explicit_and_derived_keys(self):
        """A keyless authenticated object must not fall back to a shared key."""
        from litellm.caching.caching import Cache
        from litellm.proxy._types import UserAPIKeyAuth

        cache = Cache()
        metadata = {"user_api_key_auth": UserAPIKeyAuth(user_id="user-a")}
        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}

        assert cache.get_cache_key(**request, metadata=metadata) is None
        assert cache.prepare_explicit_cache_key("caller-selected", metadata=metadata) is None

    def test_unauthenticated_explicit_cache_key_remains_compatible(self):
        """Non-proxy callers may continue to use an explicit cache key."""
        from litellm.caching.caching import Cache

        assert Cache().prepare_explicit_cache_key("caller-selected") == "caller-selected"

    def test_missing_auth_skips_sync_and_async_cache_reads_and_writes(self):
        """An unresolved authenticated identity must bypass both cache paths."""
        from litellm.caching.caching import Cache
        from litellm.proxy._types import UserAPIKeyAuth

        cache = Cache()
        metadata = {"user_api_key_auth": UserAPIKeyAuth(user_id="user-a")}

        cache.add_cache("sync-value", cache_key="caller-selected", metadata=metadata)
        assert cache.get_cache(cache_key="caller-selected", metadata=metadata) is None

        asyncio.run(cache.async_add_cache("async-value", cache_key="caller-selected", metadata=metadata))
        assert asyncio.run(cache.async_get_cache(cache_key="caller-selected", metadata=metadata)) is None

    def test_authenticated_explicit_cache_key_is_scoped_before_backend_access(self):
        """Explicit keys must carry the trusted caller namespace before storage."""
        from litellm.caching.caching import Cache
        from litellm.proxy._types import UserAPIKeyAuth

        cache = Cache()
        metadata = {"user_api_key_auth": UserAPIKeyAuth(token="validated-token-a")}
        scoped_key = cache.prepare_explicit_cache_key("caller-selected", metadata=metadata)

        assert scoped_key is not None
        assert scoped_key.startswith("caller:")
        assert scoped_key.endswith(":caller-selected")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
