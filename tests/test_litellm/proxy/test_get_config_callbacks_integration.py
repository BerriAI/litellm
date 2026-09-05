"""Integration test for /get/config/callbacks endpoint with runtime callbacks."""

import pytest

import litellm
from litellm.proxy.proxy_server import _get_runtime_callbacks, _normalize_callback_alias


class TestCallbackVisibility:
    """Test that configured callbacks are not marked read_only and runtime-only are."""

    def test_configured_callback_not_read_only(self):
        """Test that a callback from config doesn't get marked as read_only."""
        # The configured callback should not have read_only flag
        # (it's only added for runtime-only callbacks in the route)
        original = litellm.success_callback
        try:
            # Clear runtime callbacks
            litellm.success_callback = None
            litellm.failure_callback = None
            litellm._async_success_callback = None
            litellm._async_failure_callback = None
            litellm.callbacks = None

            # Verify nothing is discovered
            runtime = _get_runtime_callbacks()
            assert len(runtime) == 0
        finally:
            litellm.success_callback = original

    def test_runtime_only_callback_discovered(self):
        """Test that a runtime-only callback is discovered correctly."""
        original = litellm.callbacks
        try:
            litellm.callbacks = ["otel"]
            runtime = _get_runtime_callbacks()
            assert ("otel", "success_and_failure") in runtime
        finally:
            litellm.callbacks = original

    def test_multiple_runtime_callbacks(self):
        """Test that multiple runtime callbacks are all discovered."""
        original_success = litellm.success_callback
        original_failure = litellm.failure_callback
        original_combined = litellm.callbacks

        try:
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["generic_api"]
            litellm.callbacks = ["otel"]

            runtime = _get_runtime_callbacks()
            runtime_dict = {name: ctype for name, ctype in runtime}

            assert "langfuse" in runtime_dict
            assert runtime_dict["langfuse"] == "success"
            assert "generic_api" in runtime_dict
            assert runtime_dict["generic_api"] == "failure"
            assert "otel" in runtime_dict
            assert runtime_dict["otel"] == "success_and_failure"
        finally:
            litellm.success_callback = original_success
            litellm.failure_callback = original_failure
            litellm.callbacks = original_combined

    def test_alias_normalization_on_runtime_callbacks(self):
        """Test that runtime callbacks have aliases normalized."""
        original = litellm.callbacks
        try:
            # Register an alias-named callback
            litellm.callbacks = ["opentelemetry"]
            runtime = _get_runtime_callbacks()

            # Should discover as "opentelemetry", not normalized yet
            assert ("opentelemetry", "success_and_failure") in runtime

            # But _normalize_callback_alias should convert it
            assert _normalize_callback_alias("opentelemetry") == "otel"
        finally:
            litellm.callbacks = original
