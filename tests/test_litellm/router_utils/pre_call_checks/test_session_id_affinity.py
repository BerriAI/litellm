import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import json

import litellm
from litellm.caching.dual_cache import DualCache
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
async def test_async_session_id_affinity_routes_to_same_deployment():
    """
    When session_affinity is enabled, subsequent requests from the same session id
    should route to the same deployment.
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
        "previous_response_id": None,
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
                "model_info": {"base_model": "computer-use-preview"},
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
                "model_info": {"base_model": "computer-use-preview"},
            },
        ],
        optional_pre_call_checks=["session_affinity"],
    )

    model_group = "azure-computer-use-preview"
    session_id = "test-session-id-1"

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
            litellm_metadata={"session_id": session_id},
        )
        first_model_id = first_response._hidden_params["model_id"]

        second_response = await router.aresponses(
            model=model_group,
            input="Follow-up question",
            truncation="auto",
            litellm_metadata={"session_id": session_id},
        )
        assert second_response._hidden_params["model_id"] == first_model_id


@pytest.mark.asyncio
async def test_async_session_id_affinity_priority_over_user_key():
    """
    If both session_affinity and deployment_affinity are enabled,
    session_affinity should have priority. We test this by sending different
    session ids for the same user.
    """
    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=123,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=True,
    )

    healthy_deployments = [
        {
            "model_name": "model_group",
            "litellm_params": {"model": "model_1"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "model_group",
            "litellm_params": {"model": "model_2"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    await callback.cache.async_set_cache(
        DeploymentAffinityCheck.get_affinity_cache_key("model_group", "user1"),
        {"model_id": "deployment-1"},
    )

    await callback.cache.async_set_cache(
        DeploymentAffinityCheck.get_session_affinity_cache_key(
            "model_group", "session1"
        ),
        {"model_id": "deployment-2"},
    )

    # Should use session mapping
    filtered = await callback.async_filter_deployments(
        model="model_group",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={
            "metadata": {"user_api_key_hash": "user1", "session_id": "session1"}
        },
    )

    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "deployment-2"
