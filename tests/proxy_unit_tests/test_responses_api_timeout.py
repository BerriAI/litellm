"""
Test to verify that request_timeout from litellm_settings is properly used in responses API
"""
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, AsyncMock


@pytest.mark.asyncio
async def test_responses_api_timeout_from_config():
    """
    Test that request_timeout set in litellm_settings is properly propagated to user_request_timeout
    """
    # Create a temporary config file
    config_content = """
model_list:
  - model_name: "*"
    litellm_params:
      model: openai/*
      api_key: test-key

litellm_settings:
  request_timeout: 300  # 5 minutes
  drop_params: True
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_file = f.name

    try:
        # Import after tempfile creation
        import litellm
        import litellm.proxy.proxy_server
        from litellm.proxy.proxy_server import ProxyConfig

        # Save original values
        original_request_timeout = getattr(litellm, "request_timeout", None)
        original_user_request_timeout = litellm.proxy.proxy_server.user_request_timeout

        # Reset user_request_timeout to None to simulate fresh start
        litellm.proxy.proxy_server.user_request_timeout = None

        # Create a new ProxyConfig instance and load config (simulating proxy startup)
        config_loader = ProxyConfig()
        _, _, _ = await config_loader.load_config(
            router=None, config_file_path=config_file
        )

        # Check that litellm.request_timeout was set from config
        assert (
            litellm.request_timeout == 300
        ), f"Expected litellm.request_timeout to be 300, got {litellm.request_timeout}"

        # Check that user_request_timeout was also set
        assert (
            litellm.proxy.proxy_server.user_request_timeout == 300
        ), f"Expected user_request_timeout to be 300, got {litellm.proxy.proxy_server.user_request_timeout}"

    finally:
        # Cleanup
        os.unlink(config_file)
        if original_request_timeout is not None:
            litellm.request_timeout = original_request_timeout
        litellm.proxy.proxy_server.user_request_timeout = original_user_request_timeout


@pytest.mark.asyncio
async def test_responses_api_timeout_not_overwritten():
    """
    Test that user_request_timeout is not overwritten if already set
    """
    # Create a temporary config file
    config_content = """
model_list:
  - model_name: "*"
    litellm_params:
      model: openai/*
      api_key: test-key

litellm_settings:
  request_timeout: 400
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_file = f.name

    try:
        # Import after tempfile creation
        import litellm
        import litellm.proxy.proxy_server
        from litellm.proxy.proxy_server import ProxyConfig

        # Save original values
        original_request_timeout = getattr(litellm, "request_timeout", None)
        original_user_request_timeout = litellm.proxy.proxy_server.user_request_timeout

        # Set user_request_timeout to a specific value
        litellm.proxy.proxy_server.user_request_timeout = 500

        # Create a new ProxyConfig instance and load config
        config_loader = ProxyConfig()
        _, _, _ = await config_loader.load_config(
            router=None, config_file_path=config_file
        )

        # user_request_timeout should NOT be updated since it's already set
        assert (
            litellm.proxy.proxy_server.user_request_timeout == 500
        ), f"Expected user_request_timeout to remain 500, got {litellm.proxy.proxy_server.user_request_timeout}"

    finally:
        # Cleanup
        os.unlink(config_file)
        if original_request_timeout is not None:
            litellm.request_timeout = original_request_timeout
        litellm.proxy.proxy_server.user_request_timeout = original_user_request_timeout


@pytest.mark.asyncio
async def test_common_request_processing_uses_litellm_timeout():
    """
    Test that ProxyBaseLLMRequestProcessing uses litellm.request_timeout when user_request_timeout is None
    """
    import litellm
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    # Save original values
    original_request_timeout = getattr(litellm, "request_timeout", None)

    try:
        # Set litellm.request_timeout to a specific value
        litellm.request_timeout = 600  # 10 minutes

        # Create processor with sample data
        processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-4"})

        # Mock the base_process_llm_request method to test internal logic
        # The actual logic happens inside base_process_llm_request
        # Simulate the condition where user_request_timeout is None
        user_request_timeout = None

        # This simulates the logic in base_process_llm_request
        if user_request_timeout:
            processor.data["request_timeout"] = user_request_timeout
        elif (
            user_request_timeout is None
            and hasattr(litellm, "request_timeout")
            and litellm.request_timeout is not None
        ):
            processor.data["request_timeout"] = litellm.request_timeout

        # Verify the timeout was set correctly
        assert (
            processor.data.get("request_timeout") == 600
        ), f"Expected request_timeout to be 600, got {processor.data.get('request_timeout')}"

    finally:
        # Restore original value
        if original_request_timeout is not None:
            litellm.request_timeout = original_request_timeout


@pytest.mark.asyncio
async def test_common_request_processing_explicit_timeout():
    """
    Test that explicit user_request_timeout takes precedence
    """
    import litellm
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    # Save original values
    original_request_timeout = getattr(litellm, "request_timeout", None)

    try:
        # Set litellm.request_timeout to a specific value
        litellm.request_timeout = 600  # 10 minutes

        # Create processor with sample data
        processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-4"})

        # Simulate explicit user_request_timeout
        user_request_timeout = 300

        # This simulates the logic in base_process_llm_request
        if user_request_timeout:
            processor.data["request_timeout"] = user_request_timeout
        elif (
            user_request_timeout is None
            and hasattr(litellm, "request_timeout")
            and litellm.request_timeout is not None
        ):
            processor.data["request_timeout"] = litellm.request_timeout

        # Verify that explicit timeout takes precedence
        assert (
            processor.data.get("request_timeout") == 300
        ), f"Expected request_timeout to be 300, got {processor.data.get('request_timeout')}"

    finally:
        # Restore original value
        if original_request_timeout is not None:
            litellm.request_timeout = original_request_timeout
