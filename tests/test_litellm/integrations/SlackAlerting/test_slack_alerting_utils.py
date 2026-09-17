from unittest.mock import AsyncMock, MagicMock

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules
import litellm
from litellm.integrations.SlackAlerting.utils import _add_langfuse_trace_id_to_alert
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager


@pytest.mark.asyncio
async def test_langfuse_not_initialized_returns_none_early():
    """
    Test that when no LangfusePromptManagement is initialized,
    the function returns None immediately without executing further logic
    """
    # Ensure no Langfuse logger is in the callback manager
    litellm.logging_callback_manager = LoggingCallbackManager()

    # Create request data that would normally trigger processing
    request_data = {"litellm_logging_obj": MagicMock(), "trace_id": "test-trace-id"}

    # Call the function
    result = await _add_langfuse_trace_id_to_alert(request_data)

    # Should return None early without processing request_data
    assert result is None

    # Verify the litellm_logging_obj was never accessed (early return)
    request_data["litellm_logging_obj"].assert_not_called()


@pytest.mark.asyncio
async def test_langfuse_trace_url_uses_logger_host(monkeypatch):
    from litellm.integrations.langfuse.langfuse import LangFuseLogger

    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    logger = LangFuseLogger(
        langfuse_public_key="pk-slack-test",
        langfuse_secret="sk-slack-test",
        langfuse_host="http://127.0.0.1:1",
    )
    logging_obj = MagicMock()
    logging_obj._get_trace_id.return_value = "abc123"
    logging_obj._get_callback_object.return_value = logger

    result = await _add_langfuse_trace_id_to_alert({"litellm_logging_obj": logging_obj})

    assert result == "http://127.0.0.1:1/trace/abc123"


@pytest.mark.asyncio
async def test_langfuse_trace_url_skips_non_langfuse_callback(monkeypatch):
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    logging_obj = MagicMock()
    logging_obj._get_trace_id.return_value = "abc123"
    logging_obj._get_callback_object.return_value = object()

    assert await _add_langfuse_trace_id_to_alert({"litellm_logging_obj": logging_obj}) is None


@pytest.mark.asyncio
async def test_langfuse_trace_url_when_callback_registered_as_logger_instance(monkeypatch):
    from litellm.integrations.langfuse.langfuse import LangFuseLogger

    logger = LangFuseLogger(
        langfuse_public_key="pk-slack-instance",
        langfuse_secret="sk-slack-instance",
        langfuse_host="http://127.0.0.1:1",
    )
    monkeypatch.setattr(litellm, "success_callback", [logger])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj = MagicMock()
    logging_obj._get_trace_id.return_value = "trace-from-instance"
    logging_obj._get_callback_object.return_value = logger

    result = await _add_langfuse_trace_id_to_alert({"litellm_logging_obj": logging_obj})

    assert result == "http://127.0.0.1:1/trace/trace-from-instance"


@pytest.mark.asyncio
async def test_langfuse_trace_url_absent_when_trace_id_never_arrives(monkeypatch):
    from litellm.integrations.langfuse.langfuse import LangFuseLogger

    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    monkeypatch.setattr("litellm.integrations.SlackAlerting.utils.asyncio.sleep", AsyncMock())
    logging_obj = MagicMock()
    logging_obj._get_trace_id.return_value = None
    logging_obj._get_callback_object.return_value = LangFuseLogger(
        langfuse_public_key="pk-slack-none",
        langfuse_secret="sk-slack-none",
        langfuse_host="http://127.0.0.1:1",
    )

    assert await _add_langfuse_trace_id_to_alert({"litellm_logging_obj": logging_obj}) is None
