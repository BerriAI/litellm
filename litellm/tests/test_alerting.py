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
from litellm.utils import get_api_base
from litellm.caching import DualCache
from litellm.integrations.slack_alerting import SlackAlerting


@pytest.mark.parametrize(
    "model, optional_params, expected_api_base",
    [
        ("openai/my-fake-model", {"api_base": "my-fake-api-base"}, "my-fake-api-base"),
        ("gpt-3.5-turbo", {}, "https://api.openai.com"),
    ],
)
def test_get_api_base_unit_test(model, optional_params, expected_api_base):
    api_base = get_api_base(model=model, optional_params=optional_params)

    assert api_base == expected_api_base


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


from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta


@pytest.fixture
def slack_alerting():
    return SlackAlerting(alerting_threshold=1)


# Test for hanging LLM responses
@pytest.mark.asyncio
async def test_response_taking_too_long_hanging(slack_alerting):
    request_data = {
        "model": "test_model",
        "messages": "test_messages",
        "litellm_status": "running",
    }
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.response_taking_too_long(
            type="hanging_request", request_data=request_data
        )
        mock_send_alert.assert_awaited_once()


# Test for slow LLM responses
@pytest.mark.asyncio
async def test_response_taking_too_long_callback(slack_alerting):
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=301)
    kwargs = {"model": "test_model", "messages": "test_messages", "litellm_params": {}}
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.response_taking_too_long_callback(
            kwargs, None, start_time, end_time
        )
        mock_send_alert.assert_awaited_once()


# Test for budget crossed
@pytest.mark.asyncio
async def test_budget_alerts_crossed(slack_alerting):
    user_max_budget = 100
    user_current_spend = 101
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.budget_alerts(
            "user_budget", user_max_budget, user_current_spend
        )
        mock_send_alert.assert_awaited_once()


# Test for budget crossed again (should not fire alert 2nd time)
@pytest.mark.asyncio
async def test_budget_alerts_crossed_again(slack_alerting):
    user_max_budget = 100
    user_current_spend = 101
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.budget_alerts(
            "user_budget", user_max_budget, user_current_spend
        )
        mock_send_alert.assert_awaited_once()
        mock_send_alert.reset_mock()
        await slack_alerting.budget_alerts(
            "user_budget", user_max_budget, user_current_spend
        )
        mock_send_alert.assert_not_awaited()


# Test for send_alert - should be called once
@pytest.mark.asyncio
async def test_send_alert(slack_alerting):
    with patch.object(
        slack_alerting.async_http_handler, "post", new=AsyncMock()
    ) as mock_post:
        mock_post.return_value.status_code = 200
        await slack_alerting.send_alert("Test message", "Low", "budget_alerts")
        mock_post.assert_awaited_once()
