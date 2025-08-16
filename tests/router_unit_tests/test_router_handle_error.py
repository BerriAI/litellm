import sys, os, time
import traceback, asyncio
import pytest
from typing import List

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from litellm.types.router import ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
from unittest.mock import AsyncMock, MagicMock


load_dotenv()


@pytest.mark.asyncio
async def test_send_llm_exception_alert_success():
    """
    Test that the function sends an alert when the router.slack_alerting_logger is set.
    """
    # Create a mock LitellmRouter instance
    mock_router = MagicMock()
    mock_router.slack_alerting_logger = AsyncMock()

    # Create a mock exception
    mock_exception = Exception("Test exception")

    # Create mock request kwargs
    request_kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Create a mock error traceback
    error_traceback = 'Traceback (most recent call last):\n  File "test.py", line 10, in <module>\n    raise Exception("Test exception")\nException: Test exception'

    # Call the function
    from litellm.router_utils.handle_error import send_llm_exception_alert

    await send_llm_exception_alert(
        mock_router, request_kwargs, error_traceback, mock_exception
    )

    # Assert that the slack_alerting_logger's send_alert method was called
    mock_router.slack_alerting_logger.send_alert.assert_called_once()


@pytest.mark.asyncio
async def test_send_llm_exception_alert_no_logger():
    """
    Test that the function does error out when no slack_alerting_logger is set
    """
    # Create a mock LitellmRouter instance without a slack_alerting_logger
    mock_router = MagicMock()
    mock_router.slack_alerting_logger = None

    # Create a mock exception
    mock_exception = Exception("Test exception")

    # Create mock request kwargs
    request_kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Create a mock error traceback
    error_traceback = 'Traceback (most recent call last):\n  File "test.py", line 10, in <module>\n    raise Exception("Test exception")\nException: Test exception'

    # Call the function
    from litellm.router_utils.handle_error import send_llm_exception_alert

    await send_llm_exception_alert(
        mock_router, request_kwargs, error_traceback, mock_exception
    )


@pytest.mark.asyncio
async def test_send_llm_exception_alert_when_proxy_server_request_in_kwargs():
    """
    Test that the function does not send an alert when the request kwargs contains a proxy_server_request key.
    """
    # Create a mock LitellmRouter instance with a slack_alerting_logger
    mock_router = MagicMock()
    mock_router.slack_alerting_logger = AsyncMock()

    # Create a mock exception
    mock_exception = Exception("Test exception")

    # Create mock request kwargs
    request_kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "proxy_server_request": {},
    }

    # Create a mock error traceback
    error_traceback = 'Traceback (most recent call last):\n  File "test.py", line 10, in <module>\n    raise Exception("Test exception")\nException: Test exception'

    # Call the function
    from litellm.router_utils.handle_error import send_llm_exception_alert

    await send_llm_exception_alert(
        mock_router, request_kwargs, error_traceback, mock_exception
    )

    # Assert that no exception was raised and the function completed successfully

    mock_router.slack_alerting_logger.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_async_raise_no_deployment_exception():
    """
    Test that async_raise_no_deployment_exception returns a RouterRateLimitError
    with cooldown_list containing just IDs (not tuples with debug info).
    """
    from litellm.router_utils.handle_error import async_raise_no_deployment_exception
    from litellm.types.router import RouterRateLimitError
    from unittest.mock import patch

    # Create a mock LitellmRouter instance
    mock_router = MagicMock()
    mock_router.get_model_ids.return_value = ["deployment-1", "deployment-2"]
    mock_router.cooldown_cache.get_min_cooldown.return_value = 30.0
    mock_router.enable_pre_call_checks = True

    # Mock the _async_get_cooldown_deployments_with_debug_info function
    # It should return a list of tuples where each tuple contains (model_id, debug_info)
    mock_cooldown_list = [
        ("deployment-1", {"error": "rate_limit", "time": "2024-01-01"}),
        ("deployment-2", {"error": "server_error", "time": "2024-01-01"}),
        ("deployment-3", {"error": "timeout", "time": "2024-01-01"}),
    ]

    with patch(
        "litellm.router_utils.handle_error._async_get_cooldown_deployments_with_debug_info",
        return_value=mock_cooldown_list,
    ):
        # Call the function
        result = await async_raise_no_deployment_exception(
            litellm_router_instance=mock_router,
            model="gpt-3.5-turbo",
            parent_otel_span=None,
        )

    # Assert that the function returns a RouterRateLimitError
    assert isinstance(result, RouterRateLimitError)

    # Assert that the error has the correct properties
    assert result.model == "gpt-3.5-turbo"
    assert result.cooldown_time == 30.0
    assert result.enable_pre_call_checks is True

    # Assert that cooldown_list contains only IDs (extracted from tuples)
    expected_cooldown_list = ["deployment-1", "deployment-2", "deployment-3"]
    assert result.cooldown_list == expected_cooldown_list

    # Verify that cooldown_list contains only strings (IDs), not tuples
    for item in result.cooldown_list:
        assert isinstance(item, str), f"Expected string ID, got {type(item)}: {item}"

    # Verify mock calls
    mock_router.get_model_ids.assert_called_once_with(model_name="gpt-3.5-turbo")
    mock_router.cooldown_cache.get_min_cooldown.assert_called_once_with(
        model_ids=["deployment-1", "deployment-2"], parent_otel_span=None
    )


@pytest.mark.asyncio
async def test_async_raise_no_deployment_exception_empty_cooldown_list():
    """
    Test that async_raise_no_deployment_exception handles empty cooldown list correctly.
    """
    from litellm.router_utils.handle_error import async_raise_no_deployment_exception
    from litellm.types.router import RouterRateLimitError
    from unittest.mock import patch

    # Create a mock LitellmRouter instance
    mock_router = MagicMock()
    mock_router.get_model_ids.return_value = ["deployment-1", "deployment-2"]
    mock_router.cooldown_cache.get_min_cooldown.return_value = 15.0
    mock_router.enable_pre_call_checks = False

    # Mock empty cooldown list
    mock_cooldown_list: List = []

    with patch(
        "litellm.router_utils.handle_error._async_get_cooldown_deployments_with_debug_info",
        return_value=mock_cooldown_list,
    ):
        # Call the function
        result = await async_raise_no_deployment_exception(
            litellm_router_instance=mock_router,
            model="claude-3-sonnet",
            parent_otel_span=None,
        )

    # Assert that the function returns a RouterRateLimitError
    assert isinstance(result, RouterRateLimitError)

    # Assert that the error has the correct properties
    assert result.model == "claude-3-sonnet"
    assert result.cooldown_time == 15.0
    assert result.enable_pre_call_checks is False

    # Assert that cooldown_list is an empty list when no cooldowns exist
    assert result.cooldown_list == []
    assert isinstance(result.cooldown_list, list)


@pytest.mark.asyncio
async def test_async_raise_no_deployment_exception_none_cooldown_list():
    """
    Test that async_raise_no_deployment_exception handles None cooldown list correctly.
    Note: In practice, _async_get_cooldown_deployments_with_debug_info should never return None
    based on the implementation, but this tests defensive programming.
    """
    from litellm.router_utils.handle_error import async_raise_no_deployment_exception
    from litellm.types.router import RouterRateLimitError
    from unittest.mock import patch

    # Create a mock LitellmRouter instance
    mock_router = MagicMock()
    mock_router.get_model_ids.return_value = []
    mock_router.cooldown_cache.get_min_cooldown.return_value = 45.0
    mock_router.enable_pre_call_checks = True

    # Mock None cooldown list (though this shouldn't happen in practice)
    mock_cooldown_list = None

    with patch(
        "litellm.router_utils.handle_error._async_get_cooldown_deployments_with_debug_info",
        return_value=mock_cooldown_list,
    ):
        # After the defensive fix, this should handle None gracefully and return empty list
        result = await async_raise_no_deployment_exception(
            litellm_router_instance=mock_router,
            model="gpt-4",
            parent_otel_span=None,
        )

    # Assert that the function returns a RouterRateLimitError
    assert isinstance(result, RouterRateLimitError)

    # Assert that the error has the correct properties
    assert result.model == "gpt-4"
    assert result.cooldown_time == 45.0
    assert result.enable_pre_call_checks is True

    # Assert that cooldown_list is an empty list when cooldown_list is None
    assert result.cooldown_list == []
    assert isinstance(result.cooldown_list, list)
