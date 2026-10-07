# What is this?
## Tests slack alerting on proxy logging object

import asyncio

# import logging
# logging.basicConfig(level=logging.DEBUG)
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.SlackAlerting.slack_alerting import (
    SlackAlerting,
)
from litellm.proxy._types import CallInfo, Litellm_EntityType
from litellm.proxy.utils import ProxyLogging
from litellm.types.integrations.slack_alerting import AlertType


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


@pytest.fixture
def slack_alerting():
    return SlackAlerting(
        alerting_threshold=1, internal_usage_cache=DualCache(), alerting=["slack"]
    )


# Test for slow LLM responses




# Test for budget crossed


# Test for budget crossed again (should not fire alert 2nd time)


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
async def test_daily_reports_completion(slack_alerting):
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        litellm.callbacks = [slack_alerting]

        # on async success
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-5.5",
                    "litellm_params": {
                        "model": "gpt-5-mini",
                    },
                }
            ]
        )

        await router.acompletion(
            model="gpt-5-mini",
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
                    "model_name": "gpt-5.5",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "bad_key"},
                }
            ]
        )

        try:
            await router.acompletion(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
        except Exception as e:
            pass

        await asyncio.sleep(3)
        response_val = await slack_alerting.send_daily_reports(router=router)

        assert response_val is True

        mock_send_alert.assert_awaited()




# test models with 0 metrics are ignored


# test no alert is sent if all None or 0 metrics


# test user budget crossed alert sent only once, even if user makes multiple calls




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
#             "token": "sk-test-mock-token-606",
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






@pytest.mark.asyncio
async def test_langfuse_trace_id():
    """
    - Unit test for `_add_langfuse_trace_id_to_alert` function in slack_alerting.py
    """
    from litellm.integrations.SlackAlerting.utils import add_langfuse_trace_id_to_alert
    from litellm.litellm_core_utils.litellm_logging import Logging

    litellm.success_callback = ["langfuse"]

    litellm_logging_obj = Logging(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    litellm.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hey how's it going?"}],
        mock_response="Hey!",
        litellm_logging_obj=litellm_logging_obj,
    )

    await asyncio.sleep(3)

    assert litellm_logging_obj.get_trace_id(service_name="langfuse") is not None

    slack_alerting = SlackAlerting(
        alerting_threshold=32,
        alerting=["slack"],
        alert_types=[AlertType.llm_exceptions],
        internal_usage_cache=DualCache(),
    )

    trace_url = await add_langfuse_trace_id_to_alert(
        request_data={"litellm_logging_obj": litellm_logging_obj}
    )

    assert trace_url is not None

    returned_trace_id = trace_url.split("/")[-1]

    assert returned_trace_id == litellm_logging_obj.get_trace_id(
        service_name="langfuse"
    )




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
