import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture(autouse=True)
def mock_proxy_logging_obj(monkeypatch):
    class _SlackStub:
        def __init__(self):
            self.customer_spend_alert = AsyncMock()

    class _DBWriterStub:
        def __init__(self):
            self.update_database = AsyncMock()

    class _ProxyLoggingStub:
        def __init__(self):
            self.db_spend_update_writer = _DBWriterStub()
            self.slack_alerting_instance = _SlackStub()

    stub = _ProxyLoggingStub()
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        stub,
        raising=False,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.update_cache",
        AsyncMock(),
        raising=False,
    )
    return stub


@pytest.mark.asyncio
async def test_async_post_call_failure_hook():
    # Setup
    logger = _ProxyDBLogger()

    # Mock user_api_key_dict
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="test_alias",
        user_email="test@example.com",
        user_id="test_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        team_alias="test_team_alias",
        end_user_id="test_end_user_id",
    )

    # Mock request data
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"original_key": "original_value"},
        "proxy_server_request": {"request_id": "test_request_id"},
    }

    # Mock exception
    original_exception = Exception("Test exception")

    # Mock update_database function
    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        # Call the method
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        # Assertions
        mock_update_database.assert_called_once()

        # Check the arguments passed to update_database
        call_args = mock_update_database.call_args[1]
        print("call_args", json.dumps(call_args, indent=4, default=str))
        assert call_args["token"] == "test_api_key"
        assert call_args["response_cost"] == 0.0
        assert call_args["user_id"] == "test_user_id"
        assert call_args["end_user_id"] == "test_end_user_id"
        assert call_args["team_id"] == "test_team_id"
        assert call_args["org_id"] == "test_org_id"
        assert call_args["completion_response"] == original_exception

        # Check that metadata was properly updated
        assert "litellm_params" in call_args["kwargs"]
        assert call_args["kwargs"]["litellm_params"]["proxy_server_request"] == {
            "request_id": "test_request_id"
        }
        metadata = call_args["kwargs"]["litellm_params"]["metadata"]
        assert metadata["user_api_key"] == "test_api_key"
        assert metadata["status"] == "failure"
        assert "error_information" in metadata
        assert metadata["original_key"] == "original_value"


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_non_llm_route():
    # Setup
    logger = _ProxyDBLogger()

    # Mock user_api_key_dict with a non-LLM route
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="test_alias",
        user_email="test@example.com",
        user_id="test_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        team_alias="test_team_alias",
        end_user_id="test_end_user_id",
        request_route="/custom/route",  # Non-LLM route
    )

    # Mock request data
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"original_key": "original_value"},
        "proxy_server_request": {"request_id": "test_request_id"},
    }

    # Mock exception
    original_exception = Exception("Test exception")

    # Mock update_database function
    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        # Call the method
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        # Assert that update_database was NOT called for non-LLM routes
        mock_update_database.assert_not_called()


@pytest.mark.asyncio
async def test_async_log_success_event_writes_db_when_flag_false(monkeypatch, mock_proxy_logging_obj):
    logger = _ProxyDBLogger()

    kwargs = {
        "model": "gpt-4",
        "metadata": {
            "user_api_key": "sk-test",
            "user_api_key_user_id": "user-1",
            "user_api_key_team_id": "team-1",
            "user_api_key_org_id": "org-1",
        },
        "litellm_params": {},
        "standard_logging_object": StandardLoggingPayload(
            id="req-1",
            trace_id=None,
            call_type="acompletion",
            cache_hit=False,
            stream=False,
            status="success",
            status_fields=None,
            custom_llm_provider=None,
            saved_cache_cost=0,
            startTime=0,
            endTime=0,
            completionStartTime=0,
            response_time=None,
            model="gpt-4",
            metadata={},
            cache_key=None,
            response_cost=0.123,
            cost_breakdown=None,
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            request_tags=None,
            end_user="",
            api_base="",
            model_group=None,
            model_id=None,
            requester_ip_address=None,
            messages=None,
            response=None,
            model_parameters=None,
            hidden_params={},
            model_map_information=None,
            error_str=None,
            error_information=None,
            response_cost_failure_debug_info=None,
            guardrail_information=None,
            standard_built_in_tools_params=None,
        ),
    }

    monkeypatch.setattr(
        "litellm.proxy.utils.ProxyUpdateSpend.should_store_proxy_response_in_spend_logs",
        lambda: False,
    )

    await logger.async_log_success_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    mock_proxy_logging_obj.db_spend_update_writer.update_database.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_log_success_event_skips_db_when_flag_true(monkeypatch, mock_proxy_logging_obj):
    logger = _ProxyDBLogger()

    kwargs = {
        "model": "gpt-4",
        "metadata": {
            "user_api_key": "sk-test",
            "user_api_key_user_id": "user-1",
        },
        "litellm_params": {},
        "standard_logging_object": StandardLoggingPayload(
            id="req-1",
            trace_id=None,
            call_type="acompletion",
            cache_hit=False,
            stream=False,
            status="success",
            status_fields=None,
            custom_llm_provider=None,
            saved_cache_cost=0,
            startTime=0,
            endTime=0,
            completionStartTime=0,
            response_time=None,
            model="gpt-4",
            metadata={},
            cache_key=None,
            response_cost=0.5,
            cost_breakdown=None,
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            request_tags=None,
            end_user="",
            api_base="",
            model_group=None,
            model_id=None,
            requester_ip_address=None,
            messages=None,
            response=None,
            model_parameters=None,
            hidden_params={},
            model_map_information=None,
            error_str=None,
            error_information=None,
            response_cost_failure_debug_info=None,
            guardrail_information=None,
            standard_built_in_tools_params=None,
        ),
    }

    monkeypatch.setattr(
        "litellm.proxy.utils.ProxyUpdateSpend.should_store_proxy_response_in_spend_logs",
        lambda: True,
    )

    await logger.async_log_success_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    mock_proxy_logging_obj.db_spend_update_writer.update_database.assert_not_called()


@pytest.mark.asyncio
async def test_async_log_success_event_with_flag_none(monkeypatch, mock_proxy_logging_obj):
    logger = _ProxyDBLogger()

    kwargs = {
        "model": "gpt-4",
        "metadata": {
            "user_api_key": "sk-test",
            "user_api_key_user_id": "user-1",
            "user_api_key_team_id": "team-1",
            "user_api_key_org_id": "org-1",
        },
        "litellm_params": {},
        "standard_logging_object": StandardLoggingPayload(
            id="req-flag-none",
            trace_id=None,
            call_type="acompletion",
            cache_hit=False,
            stream=False,
            status="success",
            status_fields=None,
            custom_llm_provider=None,
            saved_cache_cost=0,
            startTime=0,
            endTime=0,
            completionStartTime=0,
            response_time=None,
            model="gpt-4",
            metadata={},
            cache_key=None,
            response_cost=0.321,
            cost_breakdown=None,
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            request_tags=None,
            end_user="",
            api_base="",
            model_group=None,
            model_id=None,
            requester_ip_address=None,
            messages=None,
            response=None,
            model_parameters=None,
            hidden_params={},
            model_map_information=None,
            error_str=None,
            error_information=None,
            response_cost_failure_debug_info=None,
            guardrail_information=None,
            standard_built_in_tools_params=None,
        ),
    }

    monkeypatch.setattr(
        "litellm.proxy.utils.ProxyUpdateSpend.should_store_proxy_response_in_spend_logs",
        lambda: None,
    )

    await logger.async_log_success_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    mock_proxy_logging_obj.db_spend_update_writer.update_database.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_post_call_success_hook_writes_db_with_guardrail_info(
    monkeypatch, mock_proxy_logging_obj
):
    logger = _ProxyDBLogger()

    class _LoggingObj:
        def __init__(self):
            self.model_call_details = {
                "metadata": {
                    "user_api_key": "sk-test",
                    "user_api_key_user_id": "user-1",
                    "user_api_key_team_id": "team-1",
                    "user_api_key_org_id": "org-1",
                },
                "litellm_params": {},
                "standard_logging_object": {
                    "response_cost": 0.25,
                },
                "end_time": datetime.now(),
            }
            self.start_time = datetime.now()

    data = {
        "litellm_logging_obj": _LoggingObj(),
        "metadata": {
            "standard_logging_guardrail_information": [
                {"guardrail_name": "noma", "status": "success"}
            ],
        },
    }

    monkeypatch.setattr(
        "litellm.proxy.utils.ProxyUpdateSpend.should_store_proxy_response_in_spend_logs",
        lambda: True,
    )

    await logger.async_post_call_success_hook(
        data=data,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        response=None,
    )

    mock_proxy_logging_obj.db_spend_update_writer.update_database.assert_awaited_once()
    kwargs = mock_proxy_logging_obj.db_spend_update_writer.update_database.call_args.kwargs
    assert kwargs["kwargs"]["metadata"]["standard_logging_guardrail_information"]
