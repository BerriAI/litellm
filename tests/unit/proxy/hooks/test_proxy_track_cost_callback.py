import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger

SECRET = "secret-prompt-marker"


def _logger_and_writer():
    writer = MagicMock()
    writer.update_database = AsyncMock()
    logger = _ProxyDBLogger(spend_writer=lambda: writer)
    return logger, writer


@pytest.mark.asyncio
async def test_failure_hook_redacts_persisted_error_information(monkeypatch):
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    logger, writer = _logger_and_writer()

    request_data = {"metadata": {}}
    await logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=litellm.BadRequestError(
            message=f"Unsupported content: {SECRET}", model="gpt-4o", llm_provider="openai"
        ),
        user_api_key_dict=UserAPIKeyAuth(),
        traceback_str=f"Traceback ... {SECRET} ...",
    )

    persisted = request_data["litellm_params"]["metadata"]["error_information"]
    assert SECRET not in json.dumps(persisted)
    assert persisted["error_message"] == "redacted-by-litellm"
    assert persisted["traceback"] == "redacted-by-litellm"
    assert persisted["error_class"] == "BadRequestError"
    assert persisted["error_code"] == "400"
    writer.update_database.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_hook_leaves_error_information_alone_when_redaction_off(monkeypatch):
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)
    logger, writer = _logger_and_writer()

    request_data = {"metadata": {}}
    await logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=litellm.BadRequestError(
            message=f"Unsupported content: {SECRET}", model="gpt-4o", llm_provider="openai"
        ),
        user_api_key_dict=UserAPIKeyAuth(),
        traceback_str="trace",
    )

    persisted = request_data["litellm_params"]["metadata"]["error_information"]
    assert SECRET in persisted["error_message"]
    writer.update_database.assert_awaited_once()
