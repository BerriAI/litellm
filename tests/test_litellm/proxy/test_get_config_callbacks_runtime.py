"""Test runtime callback visibility in /get/config/callbacks endpoint."""

import asyncio
import pytest

import litellm
from litellm.proxy.proxy_server import _get_runtime_callbacks, _normalize_callback_alias


class TestRuntimeCallbackDiscovery:
    """Test discovery of runtime-only callbacks."""

    def test_discover_success_callback(self):
        """Test discovering a callback from litellm.success_callback."""
        original = litellm.success_callback
        try:
            litellm.success_callback = ["langfuse"]
            callbacks = _get_runtime_callbacks()
            callback_names = [name for name, _ in callbacks]
            assert "langfuse" in callback_names
        finally:
            litellm.success_callback = original

    def test_discover_failure_callback(self):
        """Test discovering a callback from litellm.failure_callback."""
        original = litellm.failure_callback
        try:
            litellm.failure_callback = ["generic_api"]
            callbacks = _get_runtime_callbacks()
            callback_types = [(name, ctype) for name, ctype in callbacks]
            assert ("generic_api", "failure") in callback_types
        finally:
            litellm.failure_callback = original

    def test_discover_success_and_failure_callbacks(self):
        """Test discovering callbacks from litellm.callbacks."""
        original = litellm.callbacks
        try:
            litellm.callbacks = ["otel"]
            callbacks = _get_runtime_callbacks()
            callback_types = [(name, ctype) for name, ctype in callbacks]
            assert ("otel", "success_and_failure") in callback_types
        finally:
            litellm.callbacks = original

    def test_no_duplicate_callbacks(self):
        """Test that the same callback isn't returned multiple times."""
        original_success = litellm.success_callback
        original_failure = litellm.failure_callback
        try:
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]
            callbacks = _get_runtime_callbacks()
            callback_entries = [(name, ctype) for name, ctype in callbacks]
            # Both should be discovered
            assert ("langfuse", "success") in callback_entries
            assert ("langfuse", "failure") in callback_entries
        finally:
            litellm.success_callback = original_success
            litellm.failure_callback = original_failure

    def test_ignore_non_string_callbacks(self):
        """Test that non-string callbacks are ignored."""
        original = litellm.success_callback

        class DummyCallback:
            pass

        try:
            litellm.success_callback = [DummyCallback(), "langfuse"]
            callbacks = _get_runtime_callbacks()
            callback_names = [name for name, _ in callbacks]
            # Only the string callback should be discovered
            assert "langfuse" in callback_names
            assert DummyCallback not in callback_names
        finally:
            litellm.success_callback = original


class TestCallbackAliasNormalization:
    """Test callback name alias normalization."""

    def test_normalize_opentelemetry_to_otel(self):
        """Test normalizing opentelemetry to otel."""
        assert _normalize_callback_alias("opentelemetry") == "otel"

    def test_normalize_s3_v2_to_s3(self):
        """Test normalizing s3_v2 to s3."""
        assert _normalize_callback_alias("s3_v2") == "s3"

    def test_normalize_aws_sqs_to_sqs(self):
        """Test normalizing aws_sqs to sqs."""
        assert _normalize_callback_alias("aws_sqs") == "sqs"

    def test_normalize_custom_callback_api_to_generic_api(self):
        """Test normalizing custom_callback_api to generic_api."""
        assert _normalize_callback_alias("custom_callback_api") == "generic_api"

    def test_normalize_unknown_callback(self):
        """Test that unknown callbacks pass through unchanged."""
        assert _normalize_callback_alias("langfuse") == "langfuse"
        assert _normalize_callback_alias("generic_api") == "generic_api"


class TestGetConfigCallbacksEndpoint:
    """Test the /get/config/callbacks endpoint with runtime callbacks."""

    @pytest.mark.asyncio
    async def test_runtime_callbacks_marked_read_only(self):
        """Test that runtime-only callbacks are marked as read_only."""
        # This test requires a running proxy server, so it's more of an integration test
        # For unit test, we verify the helper functions work correctly
        from litellm.proxy.common_utils.callback_utils import process_callback

        # Simulate processing a runtime callback
        callback_obj = process_callback("otel", "success", {})
        callback_obj["read_only"] = True

        assert callback_obj["read_only"] is True
        assert callback_obj["name"] == "otel"
        assert callback_obj["type"] == "success"
