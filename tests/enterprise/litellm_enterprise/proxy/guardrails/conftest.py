"""Shared fixtures for guardrail apply_guardrail tests."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@contextmanager
def _mock_proxy_logging():
    """Patch the proxy-server globals that apply_guardrail imports at call time."""
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.post_call_success_hook = AsyncMock(return_value=None)
    mock_proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)
    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock(return_value=None)
    mock_logging_obj.async_failure_handler = AsyncMock(return_value=None)
    mock_logging_obj.success_handler = MagicMock(return_value=None)
    mock_logging_obj.failure_handler = MagicMock(return_value=None)
    mock_logging_obj.model_call_details = {}

    with (
        patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing"
        ) as mock_proc_cls,
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "0.0.0"),
    ):
        mock_proc = MagicMock()
        mock_proc.common_processing_pre_call_logic = AsyncMock(
            return_value=({}, mock_logging_obj)
        )
        mock_proc_cls.return_value = mock_proc
        yield mock_proxy_logging


@pytest.fixture
def mock_proxy_logging_ctx():
    """Return the proxy-logging context manager factory for use as `with ctx():`."""
    return _mock_proxy_logging
