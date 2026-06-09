import asyncio
import json
import os
import sys
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import litellm
import pytest
from starlette.requests import Request

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import hash_token

_PT_MOD = "litellm.proxy.pass_through_endpoints.pass_through_endpoints"


async def _mock_upstream_request(*args, **kwargs):
    mock_response = httpx.Response(200, json={})
    mock_response.request = Mock(spec=httpx.Request)
    return mock_response


def _bria_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/bria",
            "raw_path": b"/bria",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-api-key", b"dummy-api-key"),
            ],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )


def _threads_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/threads",
            "raw_path": b"/v1/threads",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer sk-test"),
                (b"openai-beta", b"assistants=v2"),
            ],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )


def _header_test_patches(mock_proxy_logging):
    mock_async_client = AsyncMock()
    mock_async_client_obj = MagicMock()
    mock_async_client_obj.client = mock_async_client
    mock_async_client.request = AsyncMock(side_effect=_mock_upstream_request)

    mock_pt_logging = MagicMock()
    mock_pt_logging.pass_through_async_success_handler = AsyncMock()

    patches = [
        patch(
            f"{_PT_MOD}.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new_callable=AsyncMock,
            side_effect=_mock_upstream_request,
        ),
        patch(f"{_PT_MOD}._is_streaming_response", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging),
        patch(f"{_PT_MOD}.pass_through_endpoint_logging", mock_pt_logging),
        patch(f"{_PT_MOD}.get_async_httpx_client", return_value=mock_async_client_obj),
        patch(f"{_PT_MOD}._read_request_body", new_callable=AsyncMock, return_value={}),
        patch(f"{_PT_MOD}._safe_get_request_headers", return_value={}),
    ]

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


def _e2e_spend_tracking_patches(
    mock_proxy_logging,
    captured_db_calls: list,
    captured_spend_payloads: list,
    captured_counter_calls: list,
    background_tasks: list,
):
    mock_async_client = AsyncMock()
    mock_async_client_obj = MagicMock()
    mock_async_client_obj.client = mock_async_client
    mock_async_client.request = AsyncMock(side_effect=_mock_upstream_request)

    mock_db_writer = MagicMock()

    async def capture_update_database(**kwargs):
        payload = get_logging_payload(
            kwargs=kwargs["kwargs"],
            response_obj=kwargs["completion_response"],
            start_time=kwargs["start_time"],
            end_time=kwargs["end_time"],
        )
        payload["spend"] = kwargs["response_cost"] or 0.0
        captured_db_calls.append(kwargs)
        captured_spend_payloads.append(payload)

    mock_db_writer.update_database = AsyncMock(side_effect=capture_update_database)
    mock_proxy_logging.db_spend_update_writer = mock_db_writer
    mock_proxy_logging.failed_tracking_alert = AsyncMock()
    mock_proxy_logging.slack_alerting_instance = MagicMock()    
    mock_proxy_logging.slack_alerting_instance.customer_spend_alert = AsyncMock()

    async def capture_increment_spend_counters(**kwargs):
        captured_counter_calls.append(kwargs)

    real_create_task = asyncio.create_task

    def capture_create_task(coro):
        task = real_create_task(coro)
        background_tasks.append(task)
        return task

    patches = [
        patch(
            f"{_PT_MOD}.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new_callable=AsyncMock,
            side_effect=_mock_upstream_request,
        ),
        patch(f"{_PT_MOD}._is_streaming_response", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging),
        patch(f"{_PT_MOD}.get_async_httpx_client", return_value=mock_async_client_obj),
        patch(f"{_PT_MOD}._read_request_body", new_callable=AsyncMock, return_value={}),
        patch(f"{_PT_MOD}._safe_get_request_headers", return_value={}),
        patch(
            "litellm.proxy.proxy_server.increment_spend_counters",
            side_effect=capture_increment_spend_counters,
        ),
        patch("litellm.proxy.proxy_server.update_cache", new_callable=AsyncMock),
        patch("asyncio.create_task", side_effect=capture_create_task),
    ]

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


@pytest.mark.asyncio
async def test_bria_passthrough_returns_configured_response_cost_header():
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
    mock_proxy_logging.post_call_success_hook = AsyncMock(return_value={})

    with _header_test_patches(mock_proxy_logging):
        response = await pass_through_request(
            request=_bria_request(),
            target="https://engine.prod.bria-api.com",
            custom_headers={"x-api-key": "dummy-api-key"},
            user_api_key_dict=UserAPIKeyAuth(
                api_key="hashed-test-key",
                user_id="test-user",
                team_id="test-team",
            ),
            cost_per_request=12.0,
        )

    assert response.status_code == 200
    assert response.body == b"{}"
    assert "x-litellm-response-cost" in response.headers
    assert float(response.headers["x-litellm-response-cost"]) == 12.0


@pytest.mark.asyncio
async def test_bria_passthrough_cost_per_request_e2e_spend_tracking():
    hashed_api_key = hash_token("sk-bria-spend-test")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hashed_api_key,
        user_id="test-user",
        team_id="test-team",
        request_route="/bria",
    )

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
    mock_proxy_logging.post_call_success_hook = AsyncMock(return_value={})

    captured_db_calls: list = []
    captured_spend_payloads: list = []
    captured_counter_calls: list = []
    background_tasks: list = []

    original_callbacks = list(litellm._async_success_callback)
    litellm._async_success_callback = [_ProxyDBLogger()]

    try:
        with _e2e_spend_tracking_patches(
            mock_proxy_logging,
            captured_db_calls,
            captured_spend_payloads,
            captured_counter_calls,
            background_tasks,
        ):
            response = await pass_through_request(
                request=_bria_request(),
                target="https://engine.prod.bria-api.com",
                custom_headers={"x-api-key": "dummy-api-key"},
                user_api_key_dict=user_api_key_dict,
                cost_per_request=12.0,
            )

            await asyncio.gather(*background_tasks, return_exceptions=True)

            mock_proxy_logging.failed_tracking_alert.assert_not_awaited()
    finally:
        litellm._async_success_callback = original_callbacks

    assert response.status_code == 200
    assert float(response.headers["x-litellm-response-cost"]) == 12.0

    assert len(captured_db_calls) == 1
    db_call = captured_db_calls[0]
    assert db_call["response_cost"] == 12.0
    assert db_call["token"] == hashed_api_key
    assert db_call["user_id"] == "test-user"
    assert db_call["team_id"] == "test-team"

    standard_logging_object = db_call["kwargs"]["standard_logging_object"]
    assert standard_logging_object is not None
    assert standard_logging_object["response_cost"] == 12.0
    assert standard_logging_object["call_type"] == "pass_through_endpoint"

    assert len(captured_spend_payloads) == 1
    spend_payload = captured_spend_payloads[0]
    assert spend_payload["spend"] == 12.0
    assert spend_payload["api_key"] == hashed_api_key
    spend_metadata = spend_payload["metadata"]
    if isinstance(spend_metadata, str):
        spend_metadata = json.loads(spend_metadata)
    assert spend_metadata["user_api_key_request_route"] == "/bria"
    assert (
        spend_metadata["passthrough_target_url"]
        == "https://engine.prod.bria-api.com"
    )

    assert len(captured_counter_calls) == 1
    counter_call = captured_counter_calls[0]
    assert counter_call["response_cost"] == 12.0
    assert counter_call["token"] == hashed_api_key
    assert counter_call["user_id"] == "test-user"
    assert counter_call["team_id"] == "test-team"


@pytest.mark.asyncio
async def test_threads_passthrough_cost_per_request_e2e_spend_tracking():
    hashed_api_key = hash_token("sk-threads-spend-test")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hashed_api_key,
        user_id="threads-user",
        team_id="threads-team",
        request_route="/v1/threads",
    )

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
    mock_proxy_logging.post_call_success_hook = AsyncMock(return_value={})

    captured_db_calls: list = []
    captured_spend_payloads: list = []
    captured_counter_calls: list = []
    background_tasks: list = []

    original_callbacks = list(litellm._async_success_callback)
    litellm._async_success_callback = [_ProxyDBLogger()]

    try:
        with _e2e_spend_tracking_patches(
            mock_proxy_logging,
            captured_db_calls,
            captured_spend_payloads,
            captured_counter_calls,
            background_tasks,
        ):
            response = await pass_through_request(
                request=_threads_request(),
                target="https://api.openai.com/v1/threads",
                custom_headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-test",
                    "OpenAI-Beta": "assistants=v2",
                },
                user_api_key_dict=user_api_key_dict,
                cost_per_request=0.05,
            )

            await asyncio.gather(*background_tasks, return_exceptions=True)

            mock_proxy_logging.failed_tracking_alert.assert_not_awaited()
    finally:
        litellm._async_success_callback = original_callbacks

    assert response.status_code == 200
    assert float(response.headers["x-litellm-response-cost"]) == 0.05

    assert len(captured_db_calls) == 1
    db_call = captured_db_calls[0]
    assert db_call["response_cost"] == 0.05
    assert db_call["token"] == hashed_api_key
    assert db_call["user_id"] == "threads-user"
    assert db_call["team_id"] == "threads-team"

    standard_logging_object = db_call["kwargs"]["standard_logging_object"]
    assert standard_logging_object is not None
    assert standard_logging_object["response_cost"] == 0.05
    assert standard_logging_object["call_type"] == "pass_through_endpoint"

    assert len(captured_spend_payloads) == 1
    spend_payload = captured_spend_payloads[0]
    assert spend_payload["spend"] == 0.05
    assert spend_payload["api_key"] == hashed_api_key
    spend_metadata = spend_payload["metadata"]
    if isinstance(spend_metadata, str):
        spend_metadata = json.loads(spend_metadata)
    assert spend_metadata["user_api_key_request_route"] == "/v1/threads"
    assert spend_metadata["passthrough_target_url"] == "https://api.openai.com/v1/threads"

    assert len(captured_counter_calls) == 1
    counter_call = captured_counter_calls[0]
    assert counter_call["response_cost"] == 0.05
    assert counter_call["token"] == hashed_api_key
    assert counter_call["user_id"] == "threads-user"
    assert counter_call["team_id"] == "threads-team"

