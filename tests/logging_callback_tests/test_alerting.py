# What is this?
## Tests slack alerting on proxy logging object

import asyncio
import io
import json
import os
import random
import sys
import time
from litellm._uuid import uuid
from datetime import datetime, timedelta
from typing import Optional

import httpx

from litellm.types.integrations.slack_alerting import AlertType

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))
import asyncio
import os
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError

import litellm
from litellm.caching.caching import DualCache, RedisCache
from litellm.integrations.SlackAlerting.slack_alerting import (
    DeploymentMetrics,
    SlackAlerting,
)
from litellm.proxy._types import CallInfo, Litellm_EntityType, WebhookEvent
from litellm.proxy.utils import ProxyLogging
from litellm.router import AlertingConfig, Router
from litellm.utils import get_api_base


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
    model = "chatgpt-v-3"
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
        alert_type=AlertType.llm_too_slow,
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
        alert_types=[AlertType.llm_exceptions],
        internal_usage_cache=DualCache(),
    )
    assert slack_alerting.alerting_threshold == 32
    assert slack_alerting.alerting == ["slack"]
    assert slack_alerting.alert_types == ["llm_exceptions"]

    slack_no_alerting = SlackAlerting()
    assert slack_no_alerting.alerting == []

    print("passed testing slack alerting init")


from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch


@pytest.fixture
def slack_alerting():
    return SlackAlerting(
        alerting_threshold=1, internal_usage_cache=DualCache(), alerting=["slack"]
    )


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


@pytest.mark.asyncio
async def test_alerting_metadata(slack_alerting):
    """
    Test alerting_metadata is propogated correctly for response taking too long
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=301)
    kwargs = {
        "model": "test_model",
        "messages": "test_messages",
        "litellm_params": {"metadata": {"alerting_metadata": {"hello": "world"}}},
    }
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:

        ## RESPONSE TAKING TOO LONG
        await slack_alerting.response_taking_too_long_callback(
            kwargs, None, start_time, end_time
        )
        mock_send_alert.assert_awaited_once()

        assert "hello" in mock_send_alert.call_args[1]["alerting_metadata"]


# Test for budget crossed
@pytest.mark.asyncio
async def test_budget_alerts_crossed(slack_alerting):
    user_max_budget = 100
    user_current_spend = 101
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.budget_alerts(
            "user_budget",
            user_info=CallInfo(
                token="",
                spend=user_current_spend,
                max_budget=user_max_budget,
                event_group=Litellm_EntityType.USER,
            ),
        )
        mock_send_alert.assert_awaited_once()


# Test for budget crossed again (should not fire alert 2nd time)
@pytest.mark.asyncio
async def test_budget_alerts_crossed_again(slack_alerting):
    user_max_budget = 100
    user_current_spend = 101
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.budget_alerts(
            "user_budget",
            user_info=CallInfo(
                token="",
                spend=user_current_spend,
                max_budget=user_max_budget,
                event_group=Litellm_EntityType.USER,
            ),
        )
        mock_send_alert.assert_awaited_once()
        mock_send_alert.reset_mock()
        await slack_alerting.budget_alerts(
            "user_budget",
            user_info=CallInfo(
                token="",
                spend=user_current_spend,
                max_budget=user_max_budget,
                event_group=Litellm_EntityType.USER,
            ),
        )
        mock_send_alert.assert_not_awaited()


# Test for send_alert - should be called once
@pytest.mark.asyncio
async def test_send_alert(slack_alerting):
    import logging

    from litellm._logging import verbose_logger

    asyncio.create_task(slack_alerting.periodic_flush())
    verbose_logger.setLevel(level=logging.DEBUG)
    with patch.object(
        slack_alerting.async_http_handler, "post", new=AsyncMock()
    ) as mock_post:
        mock_post.return_value.status_code = 200
        await slack_alerting.send_alert(
            "Test message", "Low", "budget_alerts", alerting_metadata={}
        )

        await asyncio.sleep(6)
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

        assert response_val is True

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

        assert response_val is True

        mock_send_alert.assert_awaited()


@pytest.mark.asyncio
async def test_daily_reports_redis_cache_scheduler():
    redis_cache = RedisCache()
    slack_alerting = SlackAlerting(
        internal_usage_cache=DualCache(redis_cache=redis_cache)
    )

    # we need this to be 0 so it actualy sends the report
    slack_alerting.alerting_args.daily_report_frequency = 0

    from litellm.router import AlertingConfig

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
    except Exception:
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
    router.get_model_ids.return_value = ["model1", "model2", "model3"]

    slack_alerting = SlackAlerting(internal_usage_cache=MagicMock())
    # model1:failed=None, model2:failed=0, model3:failed=10, model1:latency=0; model2:latency=0; model3:latency=None
    slack_alerting.internal_usage_cache.async_batch_get_cache = AsyncMock(
        return_value=[None, 0, 10, 0, 0, None]
    )
    slack_alerting.internal_usage_cache.async_set_cache_pipeline = AsyncMock()

    router.get_model_info.side_effect = lambda x: {"litellm_params": {"model": x}}

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        result = await slack_alerting.send_daily_reports(router)

        # Check that the send_alert method was called
        mock_send_alert.assert_called_once()
        message = mock_send_alert.call_args[1]["message"]

        # Ensure the message includes only the non-zero, non-None metrics
        assert "model3" in message
        assert "model2" not in message
        assert "model1" not in message

    assert result == True


# test no alert is sent if all None or 0 metrics
@pytest.mark.asyncio
async def test_send_daily_reports_all_zero_or_none():
    router = MagicMock()
    router.get_model_ids.return_value = ["model1", "model2", "model3"]

    slack_alerting = SlackAlerting(internal_usage_cache=MagicMock())
    slack_alerting.internal_usage_cache.async_batch_get_cache = AsyncMock(
        return_value=[None, 0, None, 0, None, 0]
    )

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        result = await slack_alerting.send_daily_reports(router)

        # Check that the send_alert method was not called
        mock_send_alert.assert_not_called()

    assert result == False


# test user budget crossed alert sent only once, even if user makes multiple calls
@pytest.mark.parametrize(
    "alerting_type",
    [
        "token_budget",
        "user_budget",
        "team_budget",
        "organization_budget",
        "proxy_budget",
        "projected_limit_exceeded",
    ],
)
@pytest.mark.asyncio
async def test_send_token_budget_crossed_alerts(alerting_type):
    slack_alerting = SlackAlerting()

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        user_info = {
            "token": "50e55ca5bfbd0759697538e8d23c0cd5031f52d9e19e176d7233b20c7c4d3403",
            "spend": 86,
            "max_budget": 100,
            "user_id": "ishaan@berri.ai",
            "user_email": "ishaan@berri.ai",
            "key_alias": "my-test-key",
            "projected_exceeded_date": "10/20/2024",
            "projected_spend": 200,
            "event_group": Litellm_EntityType.KEY,
        }

        user_info = CallInfo(**user_info)

        for _ in range(50):
            await slack_alerting.budget_alerts(
                type=alerting_type,
                user_info=user_info,
            )
        mock_send_alert.assert_awaited_once()


@pytest.mark.parametrize(
    "alerting_type",
    [
        "token_budget",
        "user_budget",
        "team_budget",
        "organization_budget",
        "proxy_budget",
        "projected_limit_exceeded",
    ],
)
@pytest.mark.asyncio
async def test_webhook_alerting(alerting_type):
    slack_alerting = SlackAlerting(alerting=["webhook"])

    with patch.object(
        slack_alerting, "send_webhook_alert", new=AsyncMock()
    ) as mock_send_alert:
        user_info = {
            "token": "50e55ca5bfbd0759697538e8d23c0cd5031f52d9e19e176d7233b20c7c4d3403",
            "spend": 1,
            "max_budget": 0,
            "user_id": "ishaan@berri.ai",
            "user_email": "ishaan@berri.ai",
            "key_alias": "my-test-key",
            "projected_exceeded_date": "10/20/2024",
            "projected_spend": 200,
            "event_group": Litellm_EntityType.KEY,
        }

        user_info = CallInfo(**user_info)
        for _ in range(50):
            await slack_alerting.budget_alerts(
                type=alerting_type,
                user_info=user_info,
            )
        mock_send_alert.assert_awaited_once()


# @pytest.mark.asyncio
# async def test_webhook_customer_spend_event():
#     """
#     Test if customer spend is working as expected
#     """
#     slack_alerting = SlackAlerting(alerting=["webhook"])

#     with patch.object(
#         slack_alerting, "send_webhook_alert", new=AsyncMock()
#     ) as mock_send_alert:
#         user_info = {
#             "token": "50e55ca5bfbd0759697538e8d23c0cd5031f52d9e19e176d7233b20c7c4d3403",
#             "spend": 1,
#             "max_budget": 0,
#             "user_id": "ishaan@berri.ai",
#             "user_email": "ishaan@berri.ai",
#             "key_alias": "my-test-key",
#             "projected_exceeded_date": "10/20/2024",
#             "projected_spend": 200,
#         }

#         user_info = CallInfo(**user_info)
#         for _ in range(50):
#             await slack_alerting.budget_alerts(
#                 type=alerting_type,
#                 user_info=user_info,
#             )
#         mock_send_alert.assert_awaited_once()


@pytest.mark.parametrize(
    "model, api_base, llm_provider, vertex_project, vertex_location",
    [
        ("gpt-3.5-turbo", None, "openai", None, None),
        (
            "azure/gpt-3.5-turbo",
            "https://openai-gpt-4-test-v-1.openai.azure.com",
            "azure",
            None,
            None,
        ),
        ("gemini-pro", None, "vertex_ai", "hardy-device-38811", "us-central1"),
    ],
)
@pytest.mark.parametrize("error_code", [500, 408, 400])
@pytest.mark.asyncio
async def test_outage_alerting_called(
    model, api_base, llm_provider, vertex_project, vertex_location, error_code
):
    """
    If call fails, outage alert is called

    If multiple calls fail, outage alert is sent
    """
    slack_alerting = SlackAlerting(alerting=["webhook"])

    litellm.callbacks = [slack_alerting]

    error_to_raise: Optional[APIError] = None

    if error_code == 400:
        print("RAISING 400 ERROR CODE")
        error_to_raise = litellm.BadRequestError(
            message="this is a bad request",
            model=model,
            llm_provider=llm_provider,
        )
    elif error_code == 408:
        print("RAISING 408 ERROR CODE")
        error_to_raise = litellm.Timeout(
            message="A timeout occurred", model=model, llm_provider=llm_provider
        )
    elif error_code == 500:
        print("RAISING 500 ERROR CODE")
        error_to_raise = litellm.ServiceUnavailableError(
            message="API is unavailable",
            model=model,
            llm_provider=llm_provider,
            response=httpx.Response(
                status_code=503,
                request=httpx.Request(
                    method="completion",
                    url="https://github.com/BerriAI/litellm",
                ),
            ),
        )

    router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": api_base,
                    "vertex_location": vertex_location,
                    "vertex_project": vertex_project,
                },
            }
        ],
        num_retries=0,
        allowed_fails=100,
    )

    slack_alerting.update_values(llm_router=router)
    with patch.object(
        slack_alerting, "outage_alerts", new=AsyncMock()
    ) as mock_outage_alert:
        try:
            await router.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hey!"}],
                mock_response=error_to_raise,
            )
        except Exception as e:
            pass

        mock_outage_alert.assert_called_once()

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        for _ in range(6):
            try:
                await router.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": "Hey!"}],
                    mock_response=error_to_raise,
                )
            except Exception as e:
                pass
        await asyncio.sleep(3)
        if error_code == 500 or error_code == 408:
            mock_send_alert.assert_called_once()
        else:
            mock_send_alert.assert_not_called()


@pytest.mark.parametrize(
    "model, api_base, llm_provider, vertex_project, vertex_location",
    [
        ("gpt-3.5-turbo", None, "openai", None, None),
        (
            "azure/gpt-3.5-turbo",
            "https://openai-gpt-4-test-v-1.openai.azure.com",
            "azure",
            None,
            None,
        ),
        ("gemini-pro", None, "vertex_ai", "hardy-device-38811", "us-central1"),
    ],
)
@pytest.mark.parametrize("error_code", [500, 408, 400])
@pytest.mark.asyncio
async def test_region_outage_alerting_called(
    model, api_base, llm_provider, vertex_project, vertex_location, error_code
):
    """
    If call fails, outage alert is called

    If multiple calls fail, outage alert is sent
    """
    slack_alerting = SlackAlerting(
        alerting=["webhook"], alert_types=[AlertType.region_outage_alerts]
    )

    litellm.callbacks = [slack_alerting]

    error_to_raise: Optional[APIError] = None

    if error_code == 400:
        print("RAISING 400 ERROR CODE")
        error_to_raise = litellm.BadRequestError(
            message="this is a bad request",
            model=model,
            llm_provider=llm_provider,
        )
    elif error_code == 408:
        print("RAISING 408 ERROR CODE")
        error_to_raise = litellm.Timeout(
            message="A timeout occurred", model=model, llm_provider=llm_provider
        )
    elif error_code == 500:
        print("RAISING 500 ERROR CODE")
        error_to_raise = litellm.ServiceUnavailableError(
            message="API is unavailable",
            model=model,
            llm_provider=llm_provider,
            response=httpx.Response(
                status_code=503,
                request=httpx.Request(
                    method="completion",
                    url="https://github.com/BerriAI/litellm",
                ),
            ),
        )

    router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": api_base,
                    "vertex_location": vertex_location,
                    "vertex_project": vertex_project,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": api_base,
                    "vertex_location": vertex_location,
                    "vertex_project": "vertex_project-2",
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        allowed_fails=100,
    )

    slack_alerting.update_values(llm_router=router)
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        for idx in range(6):
            if idx % 2 == 0:
                deployment_id = "1"
            else:
                deployment_id = "2"
            await slack_alerting.region_outage_alerts(
                exception=error_to_raise, deployment_id=deployment_id  # type: ignore
            )
        if model == "gemini-pro" and (error_code == 500 or error_code == 408):
            mock_send_alert.assert_called_once()
        else:
            mock_send_alert.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.skip(reason="test only needs to run locally ")
async def test_alerting():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "bad_key",
                },
            }
        ],
        debug_level="DEBUG",
        set_verbose=True,
        alerting_config=AlertingConfig(
            alerting_threshold=10,  # threshold for slow / hanging llm responses (in seconds). Defaults to 300 seconds
            webhook_url=os.getenv(
                "SLACK_WEBHOOK_URL"
            ),  # webhook you want to send alerts to
        ),
    )
    try:
        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

    except Exception:
        pass
    finally:
        await asyncio.sleep(3)


@pytest.mark.asyncio
async def test_langfuse_trace_id():
    """
    - Unit test for `_add_langfuse_trace_id_to_alert` function in slack_alerting.py
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.integrations.SlackAlerting.utils import _add_langfuse_trace_id_to_alert

    litellm.success_callback = ["langfuse"]

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey how's it going?"}],
        mock_response="Hey!",
        litellm_logging_obj=litellm_logging_obj,
    )

    await asyncio.sleep(3)

    assert litellm_logging_obj._get_trace_id(service_name="langfuse") is not None

    slack_alerting = SlackAlerting(
        alerting_threshold=32,
        alerting=["slack"],
        alert_types=[AlertType.llm_exceptions],
        internal_usage_cache=DualCache(),
    )

    trace_url = await _add_langfuse_trace_id_to_alert(
        request_data={"litellm_logging_obj": litellm_logging_obj}
    )

    assert trace_url is not None

    returned_trace_id = int(trace_url.split("/")[-1])

    assert returned_trace_id == int(
        litellm_logging_obj._get_trace_id(service_name="langfuse")
    )


@pytest.mark.asyncio
async def test_print_alerting_payload_warning():
    """
    Test if alerts are printed to verbose logger when log_to_console=True
    """
    litellm.set_verbose = True
    from litellm._logging import verbose_proxy_logger
    from litellm.integrations.SlackAlerting.batching_handler import send_to_webhook
    import logging

    # Create a string buffer to capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    verbose_proxy_logger.addHandler(handler)
    verbose_proxy_logger.setLevel(logging.WARNING)

    # Create SlackAlerting instance with log_to_console=True
    slack_alerting = SlackAlerting(
        alerting_threshold=0.0000001,
        alerting=["slack"],
        alert_types=[AlertType.llm_exceptions],
        internal_usage_cache=DualCache(),
    )
    slack_alerting.alerting_args.log_to_console = True

    test_payload = {"text": "Test alert message"}

    # Send an alert
    with patch.object(
        slack_alerting.async_http_handler, "post", new=AsyncMock()
    ) as mock_post:
        await send_to_webhook(
            slackAlertingInstance=slack_alerting,
            item={
                "url": "https://example.com",
                "headers": {"Content-Type": "application/json"},
                "payload": {"text": "Test alert message"},
            },
            count=1,
        )

    # Check if the payload was logged
    log_output = log_stream.getvalue()
    print(log_output)
    assert "Test alert message" in log_output

    # Clean up
    verbose_proxy_logger.removeHandler(handler)
    log_stream.close()


@pytest.mark.parametrize("report_type", ["weekly", "monthly"])
@pytest.mark.asyncio
async def test_spend_report_cache(report_type):
    """
    Test that spend reports are only sent once within their period
    """
    # Mock prisma client response
    mock_spend_data = [
        {"team_alias": "team1", "total_spend": 100.0},
        {"team_alias": "team2", "total_spend": 200.0},
    ]

    mock_tag_data = [
        {"individual_request_tag": "tag1", "total_spend": 150.0},
        {"individual_request_tag": "tag2", "total_spend": 150.0},
    ]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        # Setup mock for database query
        mock_prisma.db.query_raw = AsyncMock(
            side_effect=[mock_spend_data, mock_tag_data]
        )

        slack_alerting = SlackAlerting(
            alerting=["webhook"], internal_usage_cache=DualCache()
        )

        user_info = CallInfo(
            token="test_token",
            spend=100,
            max_budget=1000,
            user_id="test@test.com",
            user_email="test@test.com",
            key_alias="test-key",
            event_group=Litellm_EntityType.KEY,
        )

        with patch.object(
            slack_alerting, "send_alert", new=AsyncMock()
        ) as mock_send_alert:
            # First call should send alert
            if report_type == "weekly":
                await slack_alerting.send_weekly_spend_report()
            else:
                await slack_alerting.send_monthly_spend_report()

            mock_send_alert.assert_called_once()
            mock_send_alert.reset_mock()

            # Second call should not send alert (cached)
            if report_type == "weekly":
                await slack_alerting.send_weekly_spend_report()
            else:
                await slack_alerting.send_monthly_spend_report()
            mock_send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_soft_budget_alerts():
    """
    Test if soft budget alerts (warnings when approaching budget limit) work correctly
    - Test alert is sent when spend reaches 80% of budget
    """
    slack_alerting = SlackAlerting(alerting=["webhook"])

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        # Test 80% threshold
        user_info = CallInfo(
            token="test_token",
            spend=80,  # $80 spent
            soft_budget=80,
            user_id="test@test.com",
            user_email="test@test.com",
            key_alias="test-key",
            event_group=Litellm_EntityType.KEY,
        )

        await slack_alerting.budget_alerts(
            type="soft_budget",
            user_info=user_info,
        )
        mock_send_alert.assert_called_once()

        # Verify alert message contains correct percentage
        alert_message = mock_send_alert.call_args[1]["message"]
        
        print("GOT MESSAGE\n\n", alert_message)

        expected_message = (
            "Soft Budget Crossed: Total Soft Budget:`80.0`\n"
            "\n"
            "*spend:* `80.0`\n"
            "*soft_budget:* `80.0`\n"
            "*user_id:* `test@test.com`\n"
            "*user_email:* `test@test.com`\n"
            "*key_alias:* `test-key`\n"
            "*event_group:* `key`\n"
        )
        assert alert_message == expected_message


key_info = CallInfo(
    token="test_token",
    spend=81,
    soft_budget=80,
    max_budget=100,
    user_id="test@test.com",
    user_email="test@test.com",
    key_alias="test-key",
    event_group=Litellm_EntityType.KEY,
)

team_info = CallInfo(
    token="test_token",
    spend=160,
    soft_budget=150,
    max_budget=200,
    team_id="team-123",
    team_alias="engineering-team",
    event_group=Litellm_EntityType.TEAM,
)

user_info = CallInfo(
    token="test_token",
    spend=45,
    soft_budget=40,
    max_budget=50,
    user_id="user123",
    event_group=Litellm_EntityType.USER,
)

key_no_max_budget_info = CallInfo(
    token="test_token",
    spend=90,
    soft_budget=85,
    user_id="dev@test.com",
    user_email="dev@test.com",
    key_alias="dev-key",
    event_group=Litellm_EntityType.KEY,
)


@pytest.mark.parametrize(
    "entity_info",
    [
        key_info,
        team_info,
        user_info,
        key_no_max_budget_info,
    ],
)
@pytest.mark.asyncio
async def test_soft_budget_alerts_webhook(entity_info):
    """
    Tests that soft budget alerts are triggered for different entity types.
    
    Tests:
    - Key with max budget
    - Team 
    - User
    - Key without max budget
    """
    slack_alerting = SlackAlerting(alerting=["webhook"])

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        # Test entity hit soft budget limit
        await slack_alerting.budget_alerts(
            type="soft_budget",
            user_info=entity_info,
        )
        mock_send_alert.assert_called_once()

        # Verify the webhook event
        call_args = mock_send_alert.call_args[1]
        logged_webhook_event: WebhookEvent = call_args["user_info"]
        
        # Validate the webhook event has all expected fields
        assert logged_webhook_event.spend == entity_info.spend
        assert logged_webhook_event.soft_budget == entity_info.soft_budget
        assert logged_webhook_event.max_budget == entity_info.max_budget
        assert logged_webhook_event.user_id == entity_info.user_id
        assert logged_webhook_event.user_email == entity_info.user_email
        assert logged_webhook_event.key_alias == entity_info.key_alias
        assert logged_webhook_event.event_group == entity_info.event_group
        
        
        
        
        
        
        