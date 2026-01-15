import os
import sys
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import json

import litellm
from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
    DeploymentAffinityCheck,
)


class MockResponse:
    def __init__(self, json_data, status_code):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = {}

    def json(self):
        return self._json_data


@pytest.mark.asyncio
async def test_async_user_key_affinity_routes_to_same_deployment():
    """
    When deployment_affinity is enabled, subsequent requests from the same user key
    should route to the same deployment (even if the routing strategy would pick another).
    """
    mock_response_data = {
        "id": "resp_mock-resp-123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello there!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-1",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["deployment_affinity"],
    )

    model_group = "azure-computer-use-preview"
    user_api_key_hash = "test-user-key-1"

    # Deterministic routing: first selection uses seq[0], second selection attempts seq[1]
    # unless the list has been filtered to length=1 by deployment affinity.
    choice_calls = {"count": 0}

    def deterministic_choice(seq):
        choice_calls["count"] += 1
        if choice_calls["count"] == 1:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post, patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=deterministic_choice,
    ):
        mock_post.return_value = MockResponse(mock_response_data, 200)

        first_response = await router.aresponses(
            model=model_group,
            input="Hello, how are you?",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        first_model_id = first_response._hidden_params["model_id"]

        # If affinity works, second request should be pinned to the same deployment
        # even though deterministic_choice would pick the other deployment when len(seq)>1.
        second_response = await router.aresponses(
            model=model_group,
            input="Follow-up question",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        assert second_response._hidden_params["model_id"] == first_model_id


@pytest.mark.asyncio
async def test_async_previous_response_id_priority_over_user_key_affinity():
    """
    If both deployment_affinity and responses_api_deployment_check are enabled,
    `previous_response_id` routing should take priority over user-key affinity.
    """
    mock_response_data = {
        "id": "resp_mock-resp-456",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'm doing well, thank you for asking!",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-1",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=[
            "deployment_affinity",
            "responses_api_deployment_check",
        ],
    )

    model_group = "azure-computer-use-preview"
    user_api_key_hash = "test-user-key-1"

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post, patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=lambda seq: seq[0],
    ):
        mock_post.return_value = MockResponse(mock_response_data, 200)

        first_response = await router.aresponses(
            model=model_group,
            input="Hello, how are you?",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        first_model_id = first_response._hidden_params["model_id"]
        first_response_id = first_response.id

        all_model_ids = router.get_model_ids(model_name=model_group)
        other_model_id = next(mid for mid in all_model_ids if mid != first_model_id)

        # Force user-key affinity to point to the OTHER deployment
        affinity_cache_key = DeploymentAffinityCheck.get_affinity_cache_key(
            model_group=model_group,
            user_key=user_api_key_hash,
        )
        await router.cache.async_set_cache(
            affinity_cache_key, {"model_id": other_model_id}, ttl=3600
        )

        # Even though user-key affinity points elsewhere, previous_response_id should pin
        # to the deployment that created the original response.
        follow_up = await router.aresponses(
            model=model_group,
            input="Follow-up question",
            truncation="auto",
            previous_response_id=first_response_id,
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        assert follow_up._hidden_params["model_id"] == first_model_id


@pytest.mark.asyncio
async def test_async_user_parameter_affinity():
    """
    When 'user' is passed as a top-level parameter (SDK-style), affinity should work.
    """
    mock_response_data = {
        "id": "resp_mock-resp-sdk",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_sdk",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "SDK Response"}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        "text": {"format": {"type": "text"}},
        "error": None,
        "previous_response_id": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-sdk-test",
                "litellm_params": {
                    "model": "azure/sdk-1",
                    "api_key": "mock",
                    "api_base": "https://mock1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-sdk-test",
                "litellm_params": {
                    "model": "azure/sdk-2",
                    "api_key": "mock",
                    "api_base": "https://mock2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["deployment_affinity"],
    )

    model_group = "azure-sdk-test"
    user_id = "sdk-user-123"

    choice_calls = {"count": 0}

    def deterministic_choice(seq):
        choice_calls["count"] += 1
        if choice_calls["count"] == 1:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post, patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=deterministic_choice,
    ):
        mock_post.return_value = MockResponse(mock_response_data, 200)

        # First call with 'user' parameter
        first_response = await router.aresponses(
            model=model_group,
            input="Hi",
            user=user_id,
        )
        first_model_id = first_response._hidden_params["model_id"]

        # Second call with same 'user' parameter should use affinity
        second_response = await router.aresponses(
            model=model_group,
            input="Follow-up",
            user=user_id,
        )
        assert second_response._hidden_params["model_id"] == first_model_id


@pytest.mark.asyncio
async def test_async_affinity_cache_expiry_allows_reroute():
    """
    When affinity TTL expires, routing should fall back to normal load balancing.
    """
    mock_response_data = {
        "id": "resp_mock-resp-ttl",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_ttl",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "TTL Response"}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        "text": {"format": {"type": "text"}},
        "error": None,
        "previous_response_id": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-ttl-test",
                "litellm_params": {
                    "model": "azure/ttl-1",
                    "api_key": "mock",
                    "api_base": "https://mock1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-ttl-test",
                "litellm_params": {
                    "model": "azure/ttl-2",
                    "api_key": "mock",
                    "api_base": "https://mock2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["deployment_affinity"],
        deployment_affinity_ttl_seconds=1,
    )

    model_group = "azure-ttl-test"
    user_api_key_hash = "ttl-user-key"

    choice_calls = {"count": 0}

    def deterministic_choice(seq):
        choice_calls["count"] += 1
        if choice_calls["count"] == 1:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post, patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=deterministic_choice,
    ):
        mock_post.return_value = MockResponse(mock_response_data, 200)

        first_response = await router.aresponses(
            model=model_group,
            input="Hi",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        first_model_id = first_response._hidden_params["model_id"]

        await asyncio.sleep(1.1)

        second_response = await router.aresponses(
            model=model_group,
            input="Follow-up after ttl",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        assert second_response._hidden_params["model_id"] != first_model_id


@pytest.mark.asyncio
async def test_async_affinity_cache_missing_deployment_falls_back():
    """
    If a cached model_id is not in healthy deployments, routing should ignore it.
    """
    mock_response_data = {
        "id": "resp_mock-resp-missing",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_missing",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Missing Response"}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        "text": {"format": {"type": "text"}},
        "error": None,
        "previous_response_id": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-missing-test",
                "litellm_params": {
                    "model": "azure/missing-1",
                    "api_key": "mock",
                    "api_base": "https://mock1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-missing-test",
                "litellm_params": {
                    "model": "azure/missing-2",
                    "api_key": "mock",
                    "api_base": "https://mock2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["deployment_affinity"],
    )

    model_group = "azure-missing-test"
    user_api_key_hash = "missing-user-key"

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post, patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=lambda seq: seq[1] if len(seq) > 1 else seq[0],
    ):
        mock_post.return_value = MockResponse(mock_response_data, 200)

        affinity_cache_key = DeploymentAffinityCheck.get_affinity_cache_key(
            model_group=model_group,
            user_key=user_api_key_hash,
        )
        await router.cache.async_set_cache(
            affinity_cache_key,
            {"model_id": "non-existent-model-id"},
            ttl=3600,
        )

        response = await router.aresponses(
            model=model_group,
            input="Should ignore missing affinity",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )

        model_ids = router.get_model_ids(model_name=model_group)
        assert response._hidden_params["model_id"] == model_ids[1]
