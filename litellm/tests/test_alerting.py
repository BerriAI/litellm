# What is this?
## Tests slack alerting on proxy logging object

import sys, json
import os
import io, asyncio
from datetime import datetime, timedelta

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))
from litellm.proxy.utils import ProxyLogging
from litellm.caching import DualCache, RedisCache
import litellm
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from litellm.utils import get_api_base
from litellm.caching import DualCache
from litellm.integrations.slack_alerting import SlackAlerting, DeploymentMetrics
import unittest.mock
from unittest.mock import AsyncMock
import pytest
from litellm.router import AlertingConfig, Router


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
        _pl.slack_alerting_instance._response_taking_too_long_callback_helper(
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
        alerting_threshold=32,
        alerting=["slack"],
        alert_types=["llm_exceptions"],
        internal_usage_cache=DualCache(),
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
    return SlackAlerting(alerting_threshold=1, internal_usage_cache=DualCache())


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


@pytest.mark.asyncio
async def test_daily_reports_unit_test(slack_alerting):
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "test-gpt",
                    "litellm_params": {"model": "gpt-3.5-turbo"},
                    "model_info": {"id": "1234"},
                }
            ]
        )
        deployment_metrics = DeploymentMetrics(
            id="1234",
            failed_request=False,
            latency_per_output_token=20.3,
            updated_at=litellm.utils.get_utc_datetime(),
        )

        updated_val = await slack_alerting.async_update_daily_reports(
            deployment_metrics=deployment_metrics
        )

        assert updated_val == 1

        await slack_alerting.send_daily_reports(router=router)

        mock_send_alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_daily_reports_completion(slack_alerting):
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        litellm.callbacks = [slack_alerting]

        # on async success
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-5",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                    },
                }
            ]
        )

        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

        await asyncio.sleep(3)
        response_val = await slack_alerting.send_daily_reports(router=router)

        assert response_val == True

        mock_send_alert.assert_awaited_once()

        # on async failure
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-5",
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "bad_key"},
                }
            ]
        )

        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
        except Exception as e:
            pass

        await asyncio.sleep(3)
        response_val = await slack_alerting.send_daily_reports(router=router)

        assert response_val == True

        mock_send_alert.assert_awaited()


@pytest.mark.asyncio
async def test_daily_reports_redis_cache_scheduler():
    redis_cache = RedisCache()
    slack_alerting = SlackAlerting(
        internal_usage_cache=DualCache(redis_cache=redis_cache)
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            }
        ]
    )

    with patch.object(
        slack_alerting, "send_alert", new=AsyncMock()
    ) as mock_send_alert, patch.object(
        redis_cache, "async_set_cache", new=AsyncMock()
    ) as mock_redis_set_cache:
        # initial call - expect empty
        await slack_alerting._run_scheduler_helper(llm_router=router)

        try:
            json.dumps(mock_redis_set_cache.call_args[0][1])
        except Exception as e:
            pytest.fail(
                "Cache value can't be json dumped - {}".format(
                    mock_redis_set_cache.call_args[0][1]
                )
            )

        mock_redis_set_cache.assert_awaited_once()

        # second call - expect empty
        await slack_alerting._run_scheduler_helper(llm_router=router)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Local test. Test if slack alerts are sent.")
async def test_send_llm_exception_to_slack():
    from litellm.router import AlertingConfig

    # on async success
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "bad_key",
                },
            },
            {
                "model_name": "gpt-5-good",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
        ],
        alerting_config=AlertingConfig(
            alerting_threshold=0.5, webhook_url=os.getenv("SLACK_WEBHOOK_URL")
        ),
    )
    try:
        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    except:
        pass

    await router.acompletion(
        model="gpt-5-good",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    await asyncio.sleep(3)


# test models with 0 metrics are ignored
@pytest.mark.asyncio
async def test_send_daily_reports_ignores_zero_values():
    router = MagicMock()
    router.get_model_ids.return_value = ['model1', 'model2', 'model3']
    
    slack_alerting = SlackAlerting(internal_usage_cache=MagicMock())
    # model1:failed=None, model2:failed=0, model3:failed=10, model1:latency=0; model2:latency=0; model3:latency=None
    slack_alerting.internal_usage_cache.async_batch_get_cache = AsyncMock(return_value=[None, 0, 10, 0, 0, None])
    slack_alerting.internal_usage_cache.async_batch_set_cache = AsyncMock()

    router.get_model_info.side_effect = lambda x: {"litellm_params": {"model": x}}
    
    with patch.object(slack_alerting, 'send_alert', new=AsyncMock()) as mock_send_alert:
        result = await slack_alerting.send_daily_reports(router)
        
        # Check that the send_alert method was called
        mock_send_alert.assert_called_once()
        message = mock_send_alert.call_args[1]['message']
        
        # Ensure the message includes only the non-zero, non-None metrics
        assert "model3" in message
        assert "model2" not in message
        assert "model1" not in message
    
    assert result == True


# test no alert is sent if all None or 0 metrics
@pytest.mark.asyncio
async def test_send_daily_reports_all_zero_or_none():
    router = MagicMock()
    router.get_model_ids.return_value = ['model1', 'model2', 'model3']
    
    slack_alerting = SlackAlerting(internal_usage_cache=MagicMock())
    slack_alerting.internal_usage_cache.async_batch_get_cache = AsyncMock(return_value=[None, 0, None, 0, None, 0])
    
    with patch.object(slack_alerting, 'send_alert', new=AsyncMock()) as mock_send_alert:
        result = await slack_alerting.send_daily_reports(router)
        
        # Check that the send_alert method was not called
        mock_send_alert.assert_not_called()
    
    assert result == False
