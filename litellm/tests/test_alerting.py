# What is this?
## Tests slack alerting on proxy logging object

import sys
import os
import io, asyncio
from datetime import datetime, timedelta

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))
from litellm.proxy.utils import ProxyLogging
from litellm.caching import DualCache
import litellm
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from litellm.caching import DualCache
from litellm.integrations.slack_alerting import SlackAlerting


@pytest.mark.asyncio
async def test_get_api_base():
    _pl = ProxyLogging(user_api_key_cache=DualCache())
    _pl.update_values(alerting=["slack"], alerting_threshold=100, redis_cache=None)
    model = "chatgpt-v-2"
    messages = [{"role": "user", "content": "Hey how's it going?"}]
    litellm_params = {
        "acompletion": True,
        "api_key": None,
        "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
        "force_timeout": 600,
        "logger_fn": None,
        "verbose": False,
        "custom_llm_provider": "azure",
        "litellm_call_id": "68f46d2d-714d-4ad8-8137-69600ec8755c",
        "model_alias_map": {},
        "completion_call_id": None,
        "metadata": None,
        "model_info": None,
        "proxy_server_request": None,
        "preset_cache_key": None,
        "no-log": False,
        "stream_response": {},
    }
    start_time = datetime.now()
    end_time = datetime.now()

    time_difference_float, model, api_base, messages = (
        _pl.slack_alerting_instance._response_taking_too_long_callback(
            kwargs={
                "model": model,
                "messages": messages,
                "litellm_params": litellm_params,
            },
            start_time=start_time,
            end_time=end_time,
        )
    )

    assert api_base is not None
    assert isinstance(api_base, str)
    assert len(api_base) > 0
    request_info = (
        f"\nRequest Model: `{model}`\nAPI Base: `{api_base}`\nMessages: `{messages}`"
    )
    slow_message = f"`Responses are slow - {round(time_difference_float,2)}s response time > Alerting threshold: {100}s`"
    await _pl.alerting_handler(
        message=slow_message + request_info,
        level="Low",
        alert_type="llm_too_slow",
    )
    print("passed test_get_api_base")


# Create a mock environment for testing
@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/webhook")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "test-project-id")


# Test the __init__ method
def test_init():
    slack_alerting = SlackAlerting(
        alerting_threshold=32, alerting=["slack"], alert_types=["llm_exceptions"]
    )
    assert slack_alerting.alerting_threshold == 32
    assert slack_alerting.alerting == ["slack"]
    assert slack_alerting.alert_types == ["llm_exceptions"]

    slack_no_alerting = SlackAlerting()
    assert slack_no_alerting.alerting == []

    print("passed testing slack alerting init")
