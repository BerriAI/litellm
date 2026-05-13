"""Unit tests for the TokenJam named-callback adapter."""
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.tokenjam.tokenjam import TokenJamLogger


@pytest.fixture
def fake_client_cls():
    """Stand-in for tokenjam.sdk.TokenJamClient — captures init + emit calls."""
    cls = MagicMock(name="TokenJamClient")
    cls.return_value = MagicMock(name="TokenJamClientInstance")
    return cls


def _times():
    start = datetime(2026, 5, 13, tzinfo=timezone.utc)
    end = datetime(2026, 5, 13, 0, 0, 1, tzinfo=timezone.utc)
    return start, end


def test_logger_initializes_client_with_env_overrides(fake_client_cls):
    """TJ_ENDPOINT / TJ_INGEST_SECRET env vars flow into the client constructor."""
    fake_sdk = MagicMock(TokenJamClient=fake_client_cls)
    with patch.dict(
        sys.modules,
        {"tokenjam": MagicMock(), "tokenjam.sdk": fake_sdk},
    ), patch.dict(
        os.environ,
        {
            "TJ_ENDPOINT": "http://example.com:7391",
            "TJ_INGEST_SECRET": "shhh",
        },
        clear=False,
    ):
        TokenJamLogger()
    fake_client_cls.assert_called_once_with(
        endpoint="http://example.com:7391",
        ingest_secret="shhh",
    )


def test_logger_initializes_with_defaults_when_env_unset(fake_client_cls):
    fake_sdk = MagicMock(TokenJamClient=fake_client_cls)
    env = {k: v for k, v in os.environ.items()
           if k not in ("TJ_ENDPOINT", "TJ_INGEST_SECRET")}
    with patch.dict(sys.modules, {"tokenjam": MagicMock(), "tokenjam.sdk": fake_sdk}), \
         patch.dict(os.environ, env, clear=True):
        TokenJamLogger()
    fake_client_cls.assert_called_once_with(
        endpoint="http://localhost:7391",
        ingest_secret=None,
    )


def test_logger_silently_disables_when_sdk_not_installed():
    """When `import tokenjam.sdk` fails, the logger should noop, not raise."""
    # Force ImportError by aliasing tokenjam.sdk to something that raises on import.
    with patch.dict(sys.modules, {"tokenjam": None, "tokenjam.sdk": None}):
        logger = TokenJamLogger()
        assert logger._client is None
        start, end = _times()
        # Hooks must be safe no-ops.
        logger.log_success_event({"model": "gpt-4o-mini"}, MagicMock(), start, end)
        logger.log_failure_event({"model": "gpt-4o-mini"}, MagicMock(), start, end)


def test_log_success_event_delegates_to_client(fake_client_cls):
    fake_sdk = MagicMock(TokenJamClient=fake_client_cls)
    with patch.dict(sys.modules, {"tokenjam": MagicMock(), "tokenjam.sdk": fake_sdk}):
        logger = TokenJamLogger()
    start, end = _times()
    kwargs = {"model": "openai/gpt-4o-mini",
              "metadata": {"tj_agent_id": "agent-a"}}
    response = MagicMock(name="response")
    logger.log_success_event(kwargs, response, start, end)

    logger._client.emit_litellm_span.assert_called_once_with(
        kwargs=kwargs,
        response_obj=response,
        start_time=start,
        end_time=end,
        success=True,
    )


def test_log_failure_event_marks_success_false(fake_client_cls):
    fake_sdk = MagicMock(TokenJamClient=fake_client_cls)
    with patch.dict(sys.modules, {"tokenjam": MagicMock(), "tokenjam.sdk": fake_sdk}):
        logger = TokenJamLogger()
    start, end = _times()
    err = RuntimeError("rate limited")
    logger.log_failure_event({"model": "gpt-4o"}, err, start, end)

    call = logger._client.emit_litellm_span.call_args
    assert call.kwargs["success"] is False
    assert call.kwargs["response_obj"] is err


def test_hook_swallows_emit_errors(fake_client_cls):
    """A raising emit_litellm_span must not propagate into the caller."""
    fake_sdk = MagicMock(TokenJamClient=fake_client_cls)
    with patch.dict(sys.modules, {"tokenjam": MagicMock(), "tokenjam.sdk": fake_sdk}):
        logger = TokenJamLogger()
    logger._client.emit_litellm_span.side_effect = RuntimeError("boom")
    start, end = _times()
    # Must not raise.
    logger.log_success_event({"model": "gpt-4o"}, MagicMock(), start, end)
    logger.log_failure_event({"model": "gpt-4o"}, MagicMock(), start, end)


@pytest.mark.asyncio
async def test_async_hooks_delegate_to_sync(fake_client_cls):
    fake_sdk = MagicMock(TokenJamClient=fake_client_cls)
    with patch.dict(sys.modules, {"tokenjam": MagicMock(), "tokenjam.sdk": fake_sdk}):
        logger = TokenJamLogger()
    start, end = _times()
    await logger.async_log_success_event(
        {"model": "gpt-4o"}, MagicMock(), start, end,
    )
    await logger.async_log_failure_event(
        {"model": "gpt-4o"}, MagicMock(), start, end,
    )
    assert logger._client.emit_litellm_span.call_count == 2
