import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params, ModelInfo
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
