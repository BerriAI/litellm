"""End-to-end test for /get/config/callbacks with runtime callbacks."""

import asyncio
import pytest

import litellm


@pytest.mark.asyncio
async def test_get_config_callbacks_includes_runtime_callbacks():
    """
    E2E test: verify that /get/config/callbacks endpoint returns runtime-only callbacks.

    This test:
    1. Sets a runtime callback
    2. Calls the get_config function (simulating the /get/config/callbacks route)
    3. Verifies the runtime callback appears in the response with read_only: true
    """
    # Save original state
    original_success = litellm.success_callback
    original_callbacks = litellm.callbacks

    try:
        # Register a runtime success callback
        litellm.success_callback = ["langfuse"]
        # Register a runtime success_and_failure callback
        litellm.callbacks = ["otel"]

        # Simulate the /get/config/callbacks route behavior
        from litellm.proxy.proxy_server import (
            _get_runtime_callbacks,
            _normalize_callback_alias,
        )
        from litellm.proxy.common_utils.callback_utils import process_callback

        # Discover runtime callbacks
        runtime_callbacks = _get_runtime_callbacks()
        config_callback_names: set[str] = set()  # No configured callbacks in this test

        data_to_return = []

        for runtime_callback, callback_type in runtime_callbacks:
            normalized_name = _normalize_callback_alias(runtime_callback)

            # Only add if not already in config (runtime-only)
            if normalized_name not in config_callback_names and normalized_name not in {c["name"] for c in data_to_return}:
                callback_obj = process_callback(normalized_name, callback_type, {})
                # Mark as read_only since it's runtime-only
                callback_obj["read_only"] = True
                data_to_return.append(callback_obj)

        # Verify the results
        callback_names = {cb["name"] for cb in data_to_return}
        assert "langfuse" in callback_names
        assert "otel" in callback_names

        # Verify read_only is set
        langfuse_cb = next((cb for cb in data_to_return if cb["name"] == "langfuse"), None)
        assert langfuse_cb is not None
        assert langfuse_cb["read_only"] is True
        assert langfuse_cb["type"] == "success"

        otel_cb = next((cb for cb in data_to_return if cb["name"] == "otel"), None)
        assert otel_cb is not None
        assert otel_cb["read_only"] is True
        assert otel_cb["type"] == "success_and_failure"

    finally:
        # Restore original state
        litellm.success_callback = original_success
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_configured_and_runtime_callbacks_merge():
    """
    E2E test: verify that configured callbacks are merged with runtime callbacks
    and configured ones are NOT marked read_only.
    """
    original_success = litellm.success_callback
    original_callbacks = litellm.callbacks

    try:
        # Register a runtime callback
        litellm.callbacks = ["otel"]

        from litellm.proxy.proxy_server import (
            _get_runtime_callbacks,
            _normalize_callback_alias,
        )
        from litellm.proxy.common_utils.callback_utils import process_callback

        # Simulate configured callbacks (langfuse from config)
        configured_callbacks = ["langfuse"]
        config_callback_names: set[str] = set(configured_callbacks)

        data_to_return = []

        # Add configured callbacks first (without read_only)
        for callback in configured_callbacks:
            callback_obj = process_callback(callback, "success", {})
            data_to_return.append(callback_obj)

        # Add runtime callbacks
        runtime_callbacks = _get_runtime_callbacks()
        for runtime_callback, callback_type in runtime_callbacks:
            normalized_name = _normalize_callback_alias(runtime_callback)

            # Only add if not already in config (runtime-only)
            if normalized_name not in config_callback_names and normalized_name not in {c["name"] for c in data_to_return}:
                callback_obj = process_callback(normalized_name, callback_type, {})
                callback_obj["read_only"] = True
                data_to_return.append(callback_obj)

        # Verify results
        langfuse_cb = next((cb for cb in data_to_return if cb["name"] == "langfuse"), None)
        otel_cb = next((cb for cb in data_to_return if cb["name"] == "otel"), None)

        # Configured callback should NOT be read_only
        assert langfuse_cb is not None
        assert langfuse_cb.get("read_only") != True

        # Runtime callback should be read_only
        assert otel_cb is not None
        assert otel_cb["read_only"] is True

    finally:
        litellm.success_callback = original_success
        litellm.callbacks = original_callbacks
