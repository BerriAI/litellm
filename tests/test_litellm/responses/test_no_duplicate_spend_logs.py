"""
Test that responses() API does not create duplicate spend logs.

This test verifies the fix for issue #15740 where kwargs.pop() was removing
the logging object before passing kwargs to internal acompletion() calls,
causing duplicate spend log entries for non-OpenAI providers.
"""
import sys
import os
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.custom_logger import CustomLogger


def test_logging_object_not_popped():
    """
    Test that litellm_logging_obj is not popped from kwargs.

    This is a regression test for issue #15740. The bug was using
    kwargs.pop() which removed the logging object, causing duplicate
    spend logs for non-OpenAI providers.
    """
    import inspect
    from litellm.responses import main as responses_module

    # Get the source code of the responses function
    source = inspect.getsource(responses_module.responses)

    # Check that .pop("litellm_logging_obj") is NOT used
    # The bug was using kwargs.pop("litellm_logging_obj") which removes it
    assert 'kwargs.pop("litellm_logging_obj")' not in source, (
        "FAIL: Found kwargs.pop('litellm_logging_obj') in responses() function. "
        "This causes duplicate spend logs. Use kwargs.get('litellm_logging_obj') instead."
    )

    # Check that .get("litellm_logging_obj") IS used
    assert 'kwargs.get("litellm_logging_obj")' in source, (
        "FAIL: Expected kwargs.get('litellm_logging_obj') but not found. "
        "The logging object must be accessed with .get() not .pop() to prevent duplication."
    )


@pytest.mark.asyncio
async def test_no_duplicate_spend_logs():
    """
    Test that spend logs are only created once, not duplicated.

    This integration test verifies the fix by using a custom logger
    that counts log_success_event calls. Before the fix, it would be
    called twice for non-OpenAI providers (Anthropic/Gemini).
    """
    # Create a custom logger to count log_success_event calls
    class SpendLogCounter(CustomLogger):
        def __init__(self):
            super().__init__()
            self.log_count = 0

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.log_count += 1

    spend_logger = SpendLogCounter()

    # Save original callbacks and set our custom logger
    original_callbacks = litellm.callbacks
    litellm.callbacks = [spend_logger]

    try:
        # Call responses API with Anthropic model using mock_response
        # This prevents real API calls while still exercising the logging path
        response = await litellm.aresponses(
            model="anthropic/claude-3-7-sonnet-latest",
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
                "type": "message"
            }],
            instructions="You are a helpful assistant.",
            mock_response="Hello! I'm doing well."  # Use mock to avoid real API call
        )

        # Give async logging time to complete
        import asyncio
        await asyncio.sleep(1)

        # Verify that log_success_event was called exactly once
        assert spend_logger.log_count == 1, (
            f"FAIL: log_success_event called {spend_logger.log_count} times instead of 1. "
            f"This indicates duplicate spend logs are being created."
        )

    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks
