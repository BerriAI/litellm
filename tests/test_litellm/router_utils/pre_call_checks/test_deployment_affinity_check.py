import asyncio
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
                "content": [{"type": "output_text", "text": "Hello there!", "annotations": []}],
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
                # Required for stable affinity scoping across multiple Azure deployments
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
async def test_async_user_key_affinity_routes_with_model_group_alias():
    """
    When Router model_group_alias is used, the requested model group (alias) can differ
    from the internally-routed model group. Deployment affinity should still stick.
    """
    mock_response_data = {
        "id": "resp_mock-resp-alias",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_alias",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Alias Response"}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        "text": {"format": {"type": "text"}},
        "error": None,
        "previous_response_id": None,
    }

    canonical_model_group = "azure-computer-use-preview"
    alias_model_group = "azure-computer-use-preview-alias"
    user_api_key_hash = "test-user-key-alias"

    router = litellm.Router(
        model_list=[
            {
                "model_name": canonical_model_group,
                "litellm_params": {
                    "model": "azure/computer-use-preview-1",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
                "model_info": {"base_model": "computer-use-preview"},
            },
            {
                "model_name": canonical_model_group,
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
                "model_info": {"base_model": "computer-use-preview"},
            },
        ],
        model_group_alias={alias_model_group: canonical_model_group},
        optional_pre_call_checks=["deployment_affinity"],
    )

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
            model=alias_model_group,
            input="Hello",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        first_model_id = first_response._hidden_params["model_id"]

        second_response = await router.aresponses(
            model=alias_model_group,
            input="Follow-up",
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
        await router.cache.async_set_cache(affinity_cache_key, {"model_id": other_model_id}, ttl=3600)

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
async def test_async_user_parameter_does_not_trigger_deployment_affinity():
    """
    The OpenAI `user` parameter identifies the *end-user* (not the API key), and should
    not be used as an affinity key.
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
                "model_info": {"base_model": "sdk-test"},
            },
            {
                "model_name": "azure-sdk-test",
                "litellm_params": {
                    "model": "azure/sdk-2",
                    "api_key": "mock",
                    "api_base": "https://mock2.openai.azure.com",
                },
                "model_info": {"base_model": "sdk-test"},
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

        # First call with 'user' parameter (end-user id)
        first_response = await router.aresponses(
            model=model_group,
            input="Hi",
            user=user_id,
        )
        first_model_id = first_response._hidden_params["model_id"]

        # Second call with same 'user' parameter should NOT be pinned by affinity
        second_response = await router.aresponses(
            model=model_group,
            input="Follow-up",
            user=user_id,
        )
        assert second_response._hidden_params["model_id"] != first_model_id


@pytest.mark.asyncio
async def test_async_pre_call_hook_uses_model_map_key_scope():
    """
    Deployment affinity caching uses (user_api_key_hash, model_map_key) -> model_id.
    """

    cache = AsyncMock()
    cache.async_set_cache = AsyncMock()

    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=123,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )

    kwargs = {
        "model_info": {"id": "model-id-123"},
        "litellm_metadata": {
            "user_api_key_hash": "user-key-abc",
            "deployment_model_name": "claude-sonnet-4-5@20250929",
        },
    }

    await callback.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    expected_cache_key = DeploymentAffinityCheck.get_affinity_cache_key(
        model_group="claude-sonnet-4-5@20250929",
        user_key="user-key-abc",
    )
    cache.async_set_cache.assert_called_once_with(
        expected_cache_key,
        {"model_id": "model-id-123"},
        ttl=123,
    )


@pytest.mark.asyncio
async def test_async_filter_deployments_uses_stable_model_map_key_for_affinity_scope():
    """
    When a stable model-map key can be derived from the deployment set, affinity should
    be scoped to that key (this helps stickiness across aliases).

    This is intentionally tested at the callback level (not via Router), to validate the
    cache key selection logic deterministically.
    """

    user_key = "user-key-abc"
    stable_model_map_key = "claude-sonnet-4-5@20250929"

    cache = AsyncMock()
    cache.async_get_cache = AsyncMock()

    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=123,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )

    healthy_deployments = [
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": f"vertex_ai/{stable_model_map_key}"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": f"bedrock/global.anthropic.{stable_model_map_key}-v1:0"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    expected_cache_key = DeploymentAffinityCheck.get_affinity_cache_key(
        model_group=stable_model_map_key,
        user_key=user_key,
    )

    async def get_cache_side_effect(*, key: str):
        if key == expected_cache_key:
            return {"model_id": "deployment-2"}
        return None

    cache.async_get_cache.side_effect = get_cache_side_effect

    filtered = await callback.async_filter_deployments(
        model="some-router-model-group",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": user_key, "model_group": "alias-group"}},
        parent_otel_span=None,
    )

    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "deployment-2"


@pytest.mark.asyncio
async def test_async_filter_deployments_falls_back_when_cached_deployment_is_unhealthy():
    """
    If affinity cache points to a deployment that's no longer healthy, callback should
    return all healthy deployments so router can pick an available one.
    """

    user_key = "user-key-unhealthy"
    stable_model_map_key = "claude-sonnet-4-5@20250929"

    cache = AsyncMock()
    cache.async_get_cache = AsyncMock(return_value={"model_id": "stale-deployment"})

    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=123,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )

    healthy_deployments = [
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": f"vertex_ai/{stable_model_map_key}"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": f"bedrock/global.anthropic.{stable_model_map_key}-v1:0"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    filtered = await callback.async_filter_deployments(
        model="some-router-model-group",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": user_key}},
        parent_otel_span=None,
    )

    assert filtered == healthy_deployments


@pytest.mark.asyncio
async def test_async_user_key_affinity_ttl_expiry_allows_reroute():
    """
    After affinity TTL expires, cached pinning should no longer filter deployments.
    """

    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=1,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )

    user_key = "ttl-user-key"
    stable_model_map_key = "claude-sonnet-4-5@20250929"
    healthy_deployments = [
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": f"vertex_ai/{stable_model_map_key}"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": f"bedrock/global.anthropic.{stable_model_map_key}-v1:0"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    await callback.async_pre_call_deployment_hook(
        kwargs={
            "model_info": {"id": "deployment-1"},
            "metadata": {
                "user_api_key_hash": user_key,
                "deployment_model_name": stable_model_map_key,
            },
        },
        call_type=None,
    )

    pinned = await callback.async_filter_deployments(
        model="some-router-model-group",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": user_key}},
        parent_otel_span=None,
    )
    assert len(pinned) == 1
    assert pinned[0]["model_info"]["id"] == "deployment-1"

    await asyncio.sleep(1.2)

    after_ttl_expiry = await callback.async_filter_deployments(
        model="some-router-model-group",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": user_key}},
        parent_otel_span=None,
    )
    assert after_ttl_expiry == healthy_deployments


def test_cache_key_does_not_double_hash_user_api_key_hash():
    """
    Proxy typically provides `metadata.user_api_key_hash` as a SHA-256 hex string.
    The affinity cache key should not hash it again.
    """

    user_api_key_hash = "b95b015b66dd02a1c14e1e0a8729211f8ee53ec962658764f4cf58546c2c68e1"
    key = DeploymentAffinityCheck.get_affinity_cache_key(
        model_group="any-model-group",
        user_key=user_api_key_hash,
    )
    assert key.endswith(user_api_key_hash)


def test_get_effective_flags_returns_per_group_config():
    """
    _get_effective_flags should return per-group flags when the model group has an entry
    in model_group_affinity_config, and global flags otherwise.
    """
    callback = DeploymentAffinityCheck(
        cache=AsyncMock(),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=True,
        enable_session_id_affinity=False,
        model_group_affinity_config={
            "gpt-4": ["deployment_affinity"],
            "claude-3": ["session_affinity", "responses_api_deployment_check"],
        },
    )

    # gpt-4: only deployment_affinity
    user_key, responses_api, session_id = callback._get_effective_flags("gpt-4")
    assert user_key is True
    assert responses_api is False
    assert session_id is False

    # claude-3: session_affinity + responses_api_deployment_check
    user_key, responses_api, session_id = callback._get_effective_flags("claude-3")
    assert user_key is False
    assert responses_api is True
    assert session_id is True

    # unconfigured-model: falls back to global flags
    user_key, responses_api, session_id = callback._get_effective_flags(
        "unconfigured-model"
    )
    assert user_key is True
    assert responses_api is True
    assert session_id is False


@pytest.mark.asyncio
async def test_model_group_affinity_config_only_applies_to_configured_group():
    """
    When model_group_affinity_config is set without global optional_pre_call_checks,
    only configured model groups should get affinity behavior.
    """
    mock_response_data = {
        "id": "resp_mock-resp-per-group",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "openai/gpt-4",
        "output": [
            {
                "type": "message",
                "id": "msg_pg",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Per-group response"}],
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
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "azure/gpt-4-deploy-1",
                    "api_key": "mock-key-1",
                    "api_base": "https://mock-gpt4-1.openai.azure.com",
                    "api_version": "2024-02-01",
                },
                "model_info": {"base_model": "gpt-4"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "azure/gpt-4-deploy-2",
                    "api_key": "mock-key-2",
                    "api_base": "https://mock-gpt4-2.openai.azure.com",
                    "api_version": "2024-02-01",
                },
                "model_info": {"base_model": "gpt-4"},
            },
            {
                "model_name": "claude-3",
                "litellm_params": {
                    "model": "azure/claude-3-deploy-1",
                    "api_key": "mock-key-3",
                    "api_base": "https://mock-claude-1.openai.azure.com",
                    "api_version": "2024-02-01",
                },
                "model_info": {"base_model": "claude-3"},
            },
            {
                "model_name": "claude-3",
                "litellm_params": {
                    "model": "azure/claude-3-deploy-2",
                    "api_key": "mock-key-4",
                    "api_base": "https://mock-claude-2.openai.azure.com",
                    "api_version": "2024-02-01",
                },
                "model_info": {"base_model": "claude-3"},
            },
        ],
        # No global optional_pre_call_checks — only per-group
        model_group_affinity_config={
            "gpt-4": ["deployment_affinity"],
        },
    )

    user_api_key_hash = "test-per-group-key"
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

        # gpt-4: affinity should work — second request pinned to same deployment
        first = await router.aresponses(
            model="gpt-4",
            input="Hello",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        first_model_id = first._hidden_params["model_id"]

        second = await router.aresponses(
            model="gpt-4",
            input="Follow-up",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        assert second._hidden_params["model_id"] == first_model_id

        # claude-3: no affinity configured — should NOT be pinned
        choice_calls["count"] = 0
        first_claude = await router.aresponses(
            model="claude-3",
            input="Hello",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        first_claude_id = first_claude._hidden_params["model_id"]

        second_claude = await router.aresponses(
            model="claude-3",
            input="Follow-up",
            truncation="auto",
            litellm_metadata={"user_api_key_hash": user_api_key_hash},
        )
        # With deterministic choice and len>1, second call picks seq[1]
        assert second_claude._hidden_params["model_id"] != first_claude_id


@pytest.mark.asyncio
async def test_model_group_affinity_config_falls_back_to_global():
    """
    When both global optional_pre_call_checks and model_group_affinity_config are set,
    unconfigured model groups should use the global settings.
    """
    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=False,
        model_group_affinity_config={
            "claude-3": ["session_affinity"],
        },
    )

    stable_model_map_key = "gpt-4"
    user_key = "test-fallback-key"

    healthy_deployments = [
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": "openai/gpt-4"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": "openai/gpt-4"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    # Set up affinity cache for gpt-4 (should work since global has deployment_affinity)
    await callback.async_pre_call_deployment_hook(
        kwargs={
            "model_info": {"id": "deployment-1"},
            "metadata": {
                "user_api_key_hash": user_key,
                "deployment_model_name": stable_model_map_key,
            },
        },
        call_type=None,
    )

    # gpt-4 not in model_group_affinity_config, so global flags apply (user_key affinity ON)
    filtered = await callback.async_filter_deployments(
        model="gpt-4",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": user_key}},
        parent_otel_span=None,
    )
    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "deployment-1"


@pytest.mark.asyncio
async def test_model_group_affinity_config_overrides_global():
    """
    When model_group_affinity_config specifies session_affinity for a model group,
    user-key affinity (from global config) should NOT apply to that group.
    """
    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=False,
        model_group_affinity_config={
            "claude-3": ["session_affinity"],
        },
    )

    stable_model_map_key = "claude-3"
    user_key = "test-override-key"

    healthy_deployments = [
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": "anthropic/claude-3-opus"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": stable_model_map_key,
            "litellm_params": {"model": "anthropic/claude-3-opus"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    # Set up user-key affinity cache for claude-3
    cache_key = DeploymentAffinityCheck.get_affinity_cache_key(
        model_group=stable_model_map_key, user_key=user_key
    )
    await callback.cache.async_set_cache(
        cache_key, {"model_id": "deployment-1"}, ttl=60
    )

    # claude-3 has per-group config (session_affinity only), so user-key affinity
    # should NOT apply even though it's globally enabled
    filtered = await callback.async_filter_deployments(
        model="claude-3",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": user_key}},
        parent_otel_span=None,
    )
    # All deployments returned (user-key affinity disabled for this group)
    assert len(filtered) == 2
