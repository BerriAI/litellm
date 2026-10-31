import asyncio
import itertools
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
    DeploymentAffinityCheck,
)


@pytest.fixture(autouse=True)
def isolate_litellm_router_state():
    saved = {
        name: getattr(litellm, name).copy()
        if isinstance(getattr(litellm, name, None), list)
        else getattr(litellm, name, None)
        for name in (
            "callbacks",
            "input_callback",
            "success_callback",
            "failure_callback",
            "_async_input_callback",
            "_async_success_callback",
            "_async_failure_callback",
            "model_fallbacks",
            "cache",
        )
        if hasattr(litellm, name)
    }
    yield
    for name, value in saved.items():
        setattr(litellm, name, value)


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
    choice_calls: Final = itertools.count(1)

    def deterministic_choice(seq: Sequence[dict[str, object]]) -> dict[str, object]:
        if next(choice_calls) == 1:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
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

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
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

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=lambda seq: seq[0],
        ),
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

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
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

    cache = DualCache()

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
    assert await cache.async_get_cache(key=expected_cache_key) == {"model_id": "model-id-123"}


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
async def test_async_filter_deployments_does_not_pin_when_target_order_is_set():
    user_key = "user-key-order-fallback"
    stable_model_map_key = "claude-sonnet-4-5@20250929"
    cache = AsyncMock()
    cache.async_get_cache = AsyncMock(return_value={"model_id": "deployment-1"})
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
        request_kwargs={"_target_order": 2, "metadata": {"user_api_key_hash": user_key}},
        parent_otel_span=None,
    )

    assert filtered == healthy_deployments
    cache.async_get_cache.assert_not_called()


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
    user_key, responses_api, session_id = callback._get_effective_flags("unconfigured-model")
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

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
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
    cache_key = DeploymentAffinityCheck.get_affinity_cache_key(model_group=stable_model_map_key, user_key=user_key)
    await callback.cache.async_set_cache(cache_key, {"model_id": "deployment-1"}, ttl=60)

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


def _jwt_metadata(user_id: str) -> dict[str, str | None]:
    return {"user_api_key_hash": None, "user_api_key_user_id": user_id}


def _two_deployments(model_group: str) -> list[dict]:
    return [
        {
            "model_name": model_group,
            "litellm_params": {"model": "openai/gpt-5.4-mini"},
            "model_info": {"id": "openai-deployment-a"},
        },
        {
            "model_name": model_group,
            "litellm_params": {"model": "openai/gpt-5.4-mini"},
            "model_info": {"id": "openai-deployment-b"},
        },
    ]


@pytest.mark.asyncio
async def test_async_jwt_user_affinity_routes_to_same_deployment():
    """
    JWT-authenticated proxy requests carry no `user_api_key_hash`, only `user_api_key_user_id`.
    They must still pin to one deployment per user.
    """
    model_group = "gpt-5.4-mini"
    router = litellm.Router(
        model_list=[
            {
                "model_name": model_group,
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "mock-api-key-a"},
                "model_info": {"id": "openai-deployment-a"},
            },
            {
                "model_name": model_group,
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "mock-api-key-b"},
                "model_info": {"id": "openai-deployment-b"},
            },
        ],
        optional_pre_call_checks=["deployment_affinity"],
    )

    choice_calls = {"count": 0}

    def deterministic_choice(seq):
        choice_calls["count"] += 1
        if choice_calls["count"] == 1:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=deterministic_choice,
    ):
        first_response = await router.acompletion(
            model=model_group,
            messages=[{"role": "user", "content": "Reply with the single word ok"}],
            mock_response="ok",
            metadata=_jwt_metadata("jwt-user-alice"),
        )
        second_response = await router.acompletion(
            model=model_group,
            messages=[{"role": "user", "content": "Reply with the single word ok"}],
            mock_response="ok",
            metadata=_jwt_metadata("jwt-user-alice"),
        )

    first_model_id = first_response._hidden_params["model_id"]
    assert first_model_id in ("openai-deployment-a", "openai-deployment-b")
    assert second_response._hidden_params["model_id"] == first_model_id


@pytest.mark.asyncio
async def test_proxy_jwt_auth_metadata_pins_per_user():
    """
    The metadata the proxy stamps for a JWT caller (`UserAPIKeyAuth(api_key=None, user_id=<sub>)`)
    must claim a pin and be read back by the filter, and another JWT user must not inherit it.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    model_group = "gpt-5.4-mini"
    healthy_deployments = _two_deployments(model_group)
    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )

    def proxy_request(user_id: str) -> dict[str, object]:
        return LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
            data={"model": model_group, "messages": [{"role": "user", "content": "hi"}], "metadata": {}},
            user_api_key_dict=UserAPIKeyAuth(api_key=None, user_id=user_id),
            _metadata_variable_name="metadata",
        )

    alice_request = proxy_request("jwt-user-alice")
    alice_metadata = alice_request["metadata"]
    assert isinstance(alice_metadata, dict)
    assert alice_metadata["user_api_key_hash"] is None

    await callback.async_pre_call_deployment_hook(
        kwargs={
            **alice_request,
            "metadata": {**alice_metadata, "deployment_model_name": model_group},
            "model_info": {"id": "openai-deployment-b"},
        },
        call_type=None,
    )

    alice_pinned = await callback.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=alice_request,
        parent_otel_span=None,
    )
    assert [deployment["model_info"]["id"] for deployment in alice_pinned] == ["openai-deployment-b"]

    bob_filtered = await callback.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=proxy_request("jwt-user-bob"),
        parent_otel_span=None,
    )
    assert bob_filtered == healthy_deployments


@pytest.mark.asyncio
async def test_jwt_user_id_never_reads_a_virtual_key_pin():
    """
    A JWT user id that happens to equal a virtual key's 64-hex hash must not read that key's pin.
    """
    model_group = "gpt-5.4-mini"
    healthy_deployments = _two_deployments(model_group)
    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )
    key_hash = "a" * 64

    await callback.async_pre_call_deployment_hook(
        kwargs={
            "metadata": {"user_api_key_hash": key_hash, "deployment_model_name": model_group},
            "model_info": {"id": "openai-deployment-b"},
        },
        call_type=None,
    )

    key_pinned = await callback.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": key_hash}},
        parent_otel_span=None,
    )
    assert [deployment["model_info"]["id"] for deployment in key_pinned] == ["openai-deployment-b"]

    lookalike_jwt_user = await callback.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": _jwt_metadata(key_hash)},
        parent_otel_span=None,
    )
    assert lookalike_jwt_user == healthy_deployments


@pytest.mark.asyncio
async def test_virtual_key_hash_wins_over_user_id_for_affinity():
    """
    A virtual-key caller with a user id pins on the key hash, so two keys owned by one user
    keep independent pins.
    """
    model_group = "gpt-5.4-mini"
    healthy_deployments = _two_deployments(model_group)
    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )

    await callback.async_pre_call_deployment_hook(
        kwargs={
            "metadata": {
                "user_api_key_hash": "key-one",
                "user_api_key_user_id": "shared-user",
                "deployment_model_name": model_group,
            },
            "model_info": {"id": "openai-deployment-b"},
        },
        call_type=None,
    )

    other_key_same_user = await callback.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"user_api_key_hash": "key-two", "user_api_key_user_id": "shared-user"}},
        parent_otel_span=None,
    )
    assert other_key_same_user == healthy_deployments


class _ForbiddenError(Exception):
    status_code: Final = 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pre_call_check", "metadata"),
    [
        ("session_affinity", {"session_id": "sess-retry"}),
        ("deployment_affinity", {"user_api_key_hash": "hashed-key-retry"}),
    ],
)
async def test_retry_reaches_healthy_sibling_when_affinity_pins_deployment_that_refused(
    pre_call_check: str, metadata: dict[str, str]
):
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": _ForbiddenError("A refuses"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 0,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=1,
        retry_after=0,
        disable_cooldowns=True,
        optional_pre_call_checks=[pre_call_check],
    )

    responses: Final = [
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            metadata=dict(metadata),
        )
        for _ in range(2)
    ]

    assert [response._hidden_params["model_id"] for response in responses] == ["B", "B"]
    assert [response._hidden_params["additional_headers"]["x-litellm-attempted-retries"] for response in responses] == [
        1,
        1,
    ]


@pytest.mark.asyncio
async def test_previous_response_id_pin_holds_even_after_its_deployment_failed():
    callback: Final = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=True,
    )
    healthy_deployments: Final = [
        {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}, "model_info": {"id": "A"}},
        {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}, "model_info": {"id": "B"}},
    ]

    assert await callback.async_filter_deployments(
        model="gpt-4o",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={
            "previous_response_id": ResponsesAPIRequestUtils._build_responses_api_response_id(
                custom_llm_provider="openai", model_id="A", response_id="resp_upstream"
            ),
            "_excluded_deployment_ids": ["A"],
            "metadata": {"user_api_key_hash": "hashed-key"},
        },
        parent_otel_span=None,
    ) == [healthy_deployments[0]]


def _mixed_web_search_router(
    optional_pre_call_checks: list[str],  # mutable-ok: Router signature
    orders: tuple[int | None, int | None] = (None, None),
    search_rpm: int | None = None,
    model_group_affinity_config: dict[str, list[str]] | None = None,  # mutable-ok: Router signature
) -> litellm.Router:
    no_search_order, search_order = orders
    return litellm.Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    **({} if no_search_order is None else {"order": no_search_order}),
                },
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-y",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    **({} if search_order is None else {"order": search_order}),
                    **({} if search_rpm is None else {"rpm": search_rpm}),
                },
                "model_info": {"id": "search"},
            },
        ],
        optional_pre_call_checks=optional_pre_call_checks,
        model_group_affinity_config=model_group_affinity_config,
        num_retries=0,
    )


_CALLER_METADATA: Final = {"user_api_key_hash": "caller-1", "session_id": "session-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "web_search_request",
    [{"web_search_options": {}}, {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}],
    ids=["web-search-options", "web-search-tool"],
)
@pytest.mark.parametrize(
    ("optional_pre_call_checks", "pin_key"),
    [
        (
            ["deployment_affinity"],
            DeploymentAffinityCheck.get_affinity_cache_key(model_group="mixed", user_key="caller-1"),
        ),
        (
            ["session_affinity"],
            DeploymentAffinityCheck.get_session_affinity_cache_key(
                model_group="mixed", session_id="session-1", user_key="caller-1"
            ),
        ),
    ],
    ids=["user-key-pin", "session-pin"],
)
async def test_pin_on_a_deployment_without_web_search_yields_to_a_web_search_request(
    optional_pre_call_checks: list[str],  # mutable-ok: Router signature
    pin_key: str,
    web_search_request: dict[str, object],  # mutable-ok: request kwargs
):
    router: Final = _mixed_web_search_router(optional_pre_call_checks)
    await router.cache.async_set_cache(pin_key, {"model_id": "no-search"}, ttl=3600)
    messages: Final = [{"role": "user", "content": "hi"}]

    plain: Final = await router.acompletion(model="mixed", messages=messages, metadata=dict(_CALLER_METADATA))
    searching: Final = await router.acompletion(
        model="mixed", messages=messages, metadata=dict(_CALLER_METADATA), **web_search_request
    )

    assert plain._hidden_params["model_id"] == "no-search"
    assert searching._hidden_params["model_id"] == "search"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("orders", "searched_on"),
    [((2, 1), "search"), ((None, 1), "search"), ((1, 2), "no-search"), ((1, None), "no-search")],
    ids=["search-tier-first", "only-search-tier-ordered", "pinned-tier-first", "only-pinned-tier-ordered"],
)
@pytest.mark.parametrize(
    ("optional_pre_call_checks", "pin_key"),
    [
        (
            ["deployment_affinity"],
            DeploymentAffinityCheck.get_affinity_cache_key(model_group="mixed", user_key="caller-1"),
        ),
        (
            ["session_affinity"],
            DeploymentAffinityCheck.get_session_affinity_cache_key(
                model_group="mixed", session_id="session-1", user_key="caller-1"
            ),
        ),
    ],
    ids=["user-key-pin", "session-pin"],
)
async def test_pin_without_web_search_yields_to_a_web_search_request_only_when_search_sits_in_an_earlier_order_tier(
    optional_pre_call_checks: list[str],  # mutable-ok: Router signature
    pin_key: str,
    orders: tuple[int | None, int | None],
    searched_on: str,
):
    router: Final = _mixed_web_search_router(optional_pre_call_checks, orders)
    await router.cache.async_set_cache(pin_key, {"model_id": "no-search"}, ttl=3600)
    messages: Final = [{"role": "user", "content": "hi"}]

    plain: Final = await router.acompletion(model="mixed", messages=messages, metadata=dict(_CALLER_METADATA))
    searching: Final = await router.acompletion(
        model="mixed", messages=messages, metadata=dict(_CALLER_METADATA), web_search_options={}
    )

    assert plain._hidden_params["model_id"] == "no-search"
    assert searching._hidden_params["model_id"] == searched_on


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "orders", [(None, 1), (2, 1), (None, None)], ids=["pin-unordered", "pin-in-later-tier", "same-tier"]
)
@pytest.mark.parametrize(
    ("affinity_check", "per_group", "pin_key"),
    [
        (
            "deployment_affinity",
            False,
            DeploymentAffinityCheck.get_affinity_cache_key(model_group="mixed", user_key="caller-1"),
        ),
        (
            "session_affinity",
            False,
            DeploymentAffinityCheck.get_session_affinity_cache_key(
                model_group="mixed", session_id="session-1", user_key="caller-1"
            ),
        ),
        (
            "deployment_affinity",
            True,
            DeploymentAffinityCheck.get_affinity_cache_key(model_group="mixed", user_key="caller-1"),
        ),
    ],
    ids=["user-key-pin", "session-pin", "per-group-user-key-pin"],
)
async def test_pin_without_web_search_holds_for_a_web_search_request_once_the_earlier_search_tier_hits_its_rpm(
    affinity_check: str,
    per_group: bool,
    pin_key: str,
    orders: tuple[int | None, int | None],
):
    router: Final = _mixed_web_search_router(
        ["enforce_model_rate_limits"] if per_group else [affinity_check, "enforce_model_rate_limits"],
        orders,
        search_rpm=1,
        model_group_affinity_config={"mixed": [affinity_check]} if per_group else None,
    )
    await router.cache.async_set_cache(pin_key, {"model_id": "no-search"}, ttl=3600)
    messages: Final = [{"role": "user", "content": "hi"}]

    with patch(
        "litellm.router_utils.pre_call_checks.model_rate_limit_check.get_utc_datetime",
        return_value=datetime(2026, 1, 1, 12, 30, 15, tzinfo=timezone.utc),
    ):
        served: Final = [
            await router.acompletion(
                model="mixed", messages=messages, metadata=dict(_CALLER_METADATA), web_search_options={}
            )
            for _ in range(4)
        ]

    assert [response._hidden_params["model_id"] for response in served] == ["search", *["no-search"] * 3]


def _tiered_web_search_router(
    optional_pre_call_checks: list[str],  # mutable-ok: Router signature
    deployments: tuple[tuple[str, bool, int | None, int | None], ...],
) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    **({} if order is None else {"order": order}),
                    **({} if rpm is None else {"rpm": rpm}),
                },
                "model_info": {"id": model_id, **({} if supports_web_search else {"supports_web_search": False})},
            }
            for model_id, supports_web_search, order, rpm in deployments
        ],
        optional_pre_call_checks=optional_pre_call_checks,
        num_retries=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployments", "pinned_id", "served_ids"),
    [
        (
            (("search-early", True, 1, 1), ("no-search", False, None, None), ("search", True, None, None)),
            "no-search",
            ["search-early", *["search"] * 3],
        ),
        (
            (("search-early", True, 1, 1), ("no-search", False, 2, None), ("search", True, 2, None)),
            "no-search",
            ["search-early", *["search"] * 3],
        ),
        (
            (("no-search-early", False, 1, None), ("no-search", False, 2, None), ("search", True, 2, None)),
            "no-search",
            ["search"] * 4,
        ),
        (
            (("search-pinned", True, None, 1), ("search", True, None, None)),
            "search-pinned",
            ["search-pinned", *["search"] * 3],
        ),
    ],
    ids=[
        "rate-limited-search-before-unordered-pin",
        "rate-limited-search-before-pin-tier",
        "flagged-before-pin-tier",
        "pinned-search-at-its-rpm",
    ],
)
@pytest.mark.parametrize(
    ("affinity_check", "pin_key"),
    [
        ("deployment_affinity", DeploymentAffinityCheck.get_affinity_cache_key(model_group="mixed", user_key="caller-1")),
        (
            "session_affinity",
            DeploymentAffinityCheck.get_session_affinity_cache_key(
                model_group="mixed", session_id="session-1", user_key="caller-1"
            ),
        ),
    ],
    ids=["user-key-pin", "session-pin"],
)
async def test_web_search_request_leaves_a_pin_only_for_an_untried_search_deployment_under_its_limit(
    affinity_check: str,
    pin_key: str,
    deployments: tuple[tuple[str, bool, int | None, int | None], ...],
    pinned_id: str,
    served_ids: list[str],  # mutable-ok: expected id sequence
):
    router: Final = _tiered_web_search_router([affinity_check, "enforce_model_rate_limits"], deployments)
    await router.cache.async_set_cache(pin_key, {"model_id": pinned_id}, ttl=3600)
    messages: Final = [{"role": "user", "content": "hi"}]

    with patch(
        "litellm.router_utils.pre_call_checks.model_rate_limit_check.get_utc_datetime",
        return_value=datetime(2026, 1, 1, 12, 30, 15, tzinfo=timezone.utc),
    ):
        served: Final = [
            await router.acompletion(
                model="mixed", messages=messages, metadata=dict(_CALLER_METADATA), web_search_options={}
            )
            for _ in range(4)
        ]

    assert [response._hidden_params["model_id"] for response in served] == served_ids


@pytest.mark.asyncio
async def test_session_pin_on_a_later_order_tier_holds_for_a_web_search_request_when_it_can_search():
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "tiered",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    "order": order,
                },
                "model_info": {"id": model_id},
            }
            for model_id, order in (("search-first-tier", 1), ("search-second-tier", 2))
        ],
        optional_pre_call_checks=["session_affinity"],
    )
    await router.cache.async_set_cache(
        DeploymentAffinityCheck.get_session_affinity_cache_key(
            model_group="tiered", session_id="session-1", user_key="caller-1"
        ),
        {"model_id": "search-second-tier"},
        ttl=3600,
    )

    searching: Final = await router.acompletion(
        model="tiered",
        messages=[{"role": "user", "content": "hi"}],
        metadata=dict(_CALLER_METADATA),
        web_search_options={},
    )

    assert searching._hidden_params["model_id"] == "search-second-tier"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_supported", "tried"),
    [((False, False), ()), ((False, True), ("search",))],
    ids=["no-deployment-can-search", "capable-deployment-already-tried"],
)
async def test_pin_holds_for_a_web_search_request_when_no_untried_deployment_can_search(
    search_supported: tuple[bool, bool], tried: tuple[str, ...]
):
    cache: Final = AsyncMock()
    cache.async_get_cache = AsyncMock(return_value={"model_id": "no-search"})
    callback: Final = DeploymentAffinityCheck(
        cache=cache, ttl_seconds=60, enable_user_key_affinity=True, enable_responses_api_affinity=False
    )
    healthy_deployments: Final = [
        {
            "model_name": "mixed",
            "litellm_params": {"model": "openai/gpt-x"},
            "model_info": {"id": model_id, "supports_web_search": supported},
        }
        for model_id, supported in zip(("no-search", "search"), search_supported)
    ]

    filtered: Final = await callback.async_filter_deployments(
        model="mixed",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={
            "metadata": {"user_api_key_hash": "caller-1"},
            "web_search_options": {},
            "_excluded_deployment_ids": list(tried),
        },
    )

    assert [d["model_info"]["id"] for d in filtered] == ["no-search"]


@pytest.mark.asyncio
async def test_key_pin_holds_for_a_web_search_request_when_tag_filtering_removes_every_deployment_that_can_search():
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    "tags": ["cheap"],
                },
                "model_info": {"id": model_id, "supports_web_search": False},
            }
            for model_id in ("flagged-a", "flagged-b")
        ]
        + [
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-y",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    "tags": ["premium"],
                },
                "model_info": {"id": "search"},
            }
        ],
        optional_pre_call_checks=["deployment_affinity"],
        enable_tag_filtering=True,
    )
    await router.cache.async_set_cache(
        DeploymentAffinityCheck.get_affinity_cache_key(model_group="mixed", user_key="caller-1"),
        {"model_id": "flagged-a"},
        ttl=3600,
    )

    served_by: Final = {
        (
            await router.acompletion(
                model="mixed",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"user_api_key_hash": "caller-1", "tags": ["cheap"]},
                web_search_options={},
            )
        )._hidden_params["model_id"]
        for _ in range(20)
    }

    assert served_by == {"flagged-a"}



def _web_search_group_router(
    optional_pre_call_checks: list[str],  # mutable-ok: Router signature
    heavy_can_search: bool,
    heavy_weight: int,
    session_affinity_repin_on_failover: bool = False,
) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    "weight": heavy_weight,
                },
                "model_info": {"id": "heavy", **({} if heavy_can_search else {"supports_web_search": False})},
            },
            {
                "model_name": "group",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key", "mock_response": "hi", "weight": 1},
                "model_info": {"id": "light"},
            },
        ],
        optional_pre_call_checks=optional_pre_call_checks,
        session_affinity_repin_on_failover=session_affinity_repin_on_failover,
        num_retries=0,
    )


_AFFINITY_KINDS: Final = pytest.mark.parametrize(
    "optional_pre_call_checks", [["deployment_affinity"], ["session_affinity"]], ids=["user-key-pin", "session-pin"]
)


async def _served_by(router: litellm.Router, caller: int, **web_search_request: object) -> object:
    response: Final = await router.acompletion(
        model="group",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"user_api_key_hash": f"caller-{caller}", "session_id": f"session-{caller}"},
        **web_search_request,
    )
    return response._hidden_params["model_id"]


@pytest.mark.asyncio
@_AFFINITY_KINDS
async def test_a_first_web_search_request_does_not_pin_plain_traffic_to_the_deployment_that_can_search(
    optional_pre_call_checks: list[str],  # mutable-ok: Router signature
):
    router: Final = _web_search_group_router(optional_pre_call_checks, heavy_can_search=False, heavy_weight=10**6)

    searched_on: Final = await _served_by(router, 1, web_search_options={})
    plain_on: Final = {await _served_by(router, 1) for _ in range(5)}

    assert searched_on == "light"
    assert plain_on == {"heavy"}


@pytest.mark.asyncio
@_AFFINITY_KINDS
async def test_a_first_web_search_request_still_pins_a_group_where_every_deployment_can_search(
    optional_pre_call_checks: list[str],  # mutable-ok: Router signature
):
    router: Final = _web_search_group_router(optional_pre_call_checks, heavy_can_search=True, heavy_weight=1)

    for caller in range(10):
        searched_on = await _served_by(router, caller, web_search_options={})
        assert {await _served_by(router, caller) for _ in range(3)} == {searched_on}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("optional_params", "session_pin"),
    [
        ({"web_search_options": {}}, None),
        ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, None),
        ({}, {"model_id": "light"}),
    ],
    ids=["web-search-options", "web-search-tool", "plain"],
)
async def test_a_successful_web_search_request_does_not_repin_a_session_in_a_mixed_group(
    optional_params: dict[str, object],  # mutable-ok: logging kwargs shape
    session_pin: dict[str, str] | None,  # mutable-ok: cache value shape
):
    router: Final = _web_search_group_router(
        ["session_affinity"], heavy_can_search=False, heavy_weight=1, session_affinity_repin_on_failover=True
    )
    timestamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await router.deployment_callback_on_success(
        {
            "litellm_params": {
                "model_info": {"id": "light"},
                "metadata": {
                    "model_group": "group",
                    "deployment_model_name": "group",
                    "session_id": "session-1",
                    "user_api_key_hash": "caller-1",
                },
            },
            "optional_params": optional_params,
            "standard_logging_object": {"total_tokens": 0},
        },
        {"choices": [{"message": {"content": "ok"}}]},
        timestamp,
        timestamp,
    )

    assert (
        await router.cache.async_get_cache(
            DeploymentAffinityCheck.get_session_affinity_cache_key("group", "session-1", "caller-1")
        )
        == session_pin
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pinned_model_id", "session_pin"),
    [("removed", {"model_id": "light"}), ("heavy", {"model_id": "heavy"})],
    ids=["ineligible-pin-moves", "eligible-pin-stays"],
)
async def test_a_successful_web_search_request_moves_a_session_only_off_an_ineligible_pin_in_a_mixed_group(
    pinned_model_id: str,
    session_pin: dict[str, str],  # mutable-ok: cache value shape
):
    router: Final = _web_search_group_router(
        ["session_affinity"], heavy_can_search=False, heavy_weight=1, session_affinity_repin_on_failover=True
    )
    cache_key: Final = DeploymentAffinityCheck.get_session_affinity_cache_key("group", "session-1", "caller-1")
    await router.cache.async_set_cache(cache_key, {"model_id": pinned_model_id})
    timestamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await router.deployment_callback_on_success(
        {
            "litellm_params": {
                "model_info": {"id": "light"},
                "metadata": {
                    "model_group": "group",
                    "deployment_model_name": "group",
                    "session_id": "session-1",
                    "user_api_key_hash": "caller-1",
                },
            },
            "optional_params": {"web_search_options": {}},
            "standard_logging_object": {"total_tokens": 0},
        },
        {"choices": [{"message": {"content": "ok"}}]},
        timestamp,
        timestamp,
    )

    assert await router.cache.async_get_cache(cache_key) == session_pin
