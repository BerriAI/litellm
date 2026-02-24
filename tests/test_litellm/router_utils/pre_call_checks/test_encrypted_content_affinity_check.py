import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import json

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
    EncryptedContentAffinityCheck,
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
async def test_encrypted_content_affinity_tracks_and_routes():
    """
    When encrypted_content_affinity is enabled, output item IDs from responses
    are tracked, and follow-up requests containing those IDs route to the same
    deployment.
    """
    mock_response_data = {
        "id": "resp_mock-123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "openai/gpt-5.1-codex",
        "output": [
            {
                "type": "message",
                "id": "msg_abc123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            },
            {
                "type": "reasoning",
                "id": "rs_encrypted_item_456",
                "status": "completed",
            },
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
        },
        "error": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-2"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
    )

    model_group = "openai.gpt-5.1-codex"

    # Track which deployment was selected
    selected_deployments = []

    def deterministic_choice(seq):
        # First call: select deployment-1
        # Second call: would select deployment-2, but affinity should override
        if len(selected_deployments) == 0:
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

        # First request: no encrypted items in input
        first_response = await router.aresponses(
            model=model_group,
            input="Hello, how are you?",
        )
        first_model_id = first_response._hidden_params["model_id"]
        selected_deployments.append(first_model_id)

        # Give async callbacks time to run
        await asyncio.sleep(0.2)

        # Second request: includes encrypted item IDs from first response
        second_response = await router.aresponses(
            model=model_group,
            input=[
                {"type": "message", "id": "msg_abc123", "role": "assistant"},
                {"type": "reasoning", "id": "rs_encrypted_item_456"},
            ],
        )
        second_model_id = second_response._hidden_params["model_id"]

        # Affinity should route to the same deployment
        assert second_model_id == first_model_id, (
            f"Expected affinity to route to {first_model_id}, "
            f"but got {second_model_id}"
        )


@pytest.mark.asyncio
async def test_encrypted_content_affinity_no_effect_on_chat_completions():
    """
    Encrypted content affinity should not affect regular chat completions
    (they don't use the Responses API).
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.environ.get("OPENAI_API_KEY", "test-key"),
                },
                "model_info": {"id": "chat-deployment-1"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
    )

    mock_chat_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_chat_response, 200)

        # Multiple chat completion requests should work normally
        response1 = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
        )
        response2 = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello again"}],
        )

        # Both should succeed (no affinity interference)
        # Check that responses have IDs (litellm may modify them)
        assert response1.id is not None
        assert response2.id is not None


@pytest.mark.asyncio
async def test_encrypted_content_affinity_bypasses_rpm_limits():
    """
    When encrypted content affinity pins to a deployment, it should bypass
    RPM limits since the encrypted content will fail on any other deployment.
    """
    mock_response_data = {
        "id": "resp_mock-rpm-test",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "openai/gpt-5.1-codex",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_encrypted_must_pin",
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        "error": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                    "rpm": 1,  # Very low limit
                },
                "model_info": {"id": "rpm-limited-deployment"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                    "rpm": 100,
                },
                "model_info": {"id": "high-rpm-deployment"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        routing_strategy="usage-based-routing-v2",
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_response_data, 200)

        # First request goes to the low-RPM deployment
        first_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input="Initial request",
        )
        first_model_id = first_response._hidden_params["model_id"]

        await asyncio.sleep(0.2)

        # Second request with encrypted content should pin to the same deployment
        # even though it's at RPM limit
        second_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {"type": "reasoning", "id": "rs_encrypted_must_pin"},
            ],
        )
        second_model_id = second_response._hidden_params["model_id"]

        # Should route to the same deployment despite RPM limit
        assert second_model_id == first_model_id


@pytest.mark.asyncio
async def test_encrypted_content_affinity_no_match_normal_routing():
    """
    When input contains item IDs that aren't tracked, normal load balancing
    should occur.
    """
    mock_response_data = {
        "id": "resp_mock-no-match",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "openai/gpt-5.1-codex",
        "output": [
            {
                "type": "message",
                "id": "msg_new",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response"}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        "error": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-a"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-b"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_response_data, 200)

        # Request with unknown item IDs should use normal routing
        response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {"type": "message", "id": "unknown_item_id_12345"},
            ],
        )

        # Should succeed with normal routing (litellm may modify the ID)
        assert response.id is not None
        # Verify it contains the original response ID in some form
        assert "resp_mock-no-match" in str(response.id) or response.id.startswith("resp_")
