import asyncio
import gc
import os
import sys
import weakref
from collections.abc import Callable
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


def _complexity_router(
    *,
    session_affinity: bool = True,
    session_affinity_ttl_seconds: int = 7,
    tiers: dict[str, str | list[str]] | None = None,
    plugins: list | None = None,
) -> litellm.Router:
    target_tiers = tiers or {
        "SIMPLE": "target-group",
        "MEDIUM": "target-group",
        "COMPLEX": "target-group",
        "REASONING": "target-group",
    }
    return litellm.Router(
        model_list=[
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_default_model": "target-group",
                    "complexity_router_config": {
                        "session_affinity": session_affinity,
                        "session_affinity_ttl_seconds": session_affinity_ttl_seconds,
                        "tiers": target_tiers,
                        "plugins": plugins or [],
                    },
                },
            },
            {
                "model_name": "target-group",
                "litellm_params": {
                    "model": "azure/computer-use-preview-1",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
                "model_info": {"base_model": "computer-use-preview"},
            },
            {
                "model_name": "target-group",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
                "model_info": {"base_model": "computer-use-preview"},
            },
        ],
    )


def _responses_mock() -> MockResponse:
    return MockResponse(
        {
            "id": "resp_mock-resp-123",
            "object": "response",
            "created_at": 1741476542,
            "status": "completed",
            "model": "azure/computer-use-preview",
            "output": [],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 10,
                "total_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
        200,
    )


def _first_then_last_choice():
    choice_calls = {"count": 0}

    def choose(sequence):
        choice_calls["count"] += 1
        return sequence[0] if choice_calls["count"] % 2 == 1 else sequence[-1]

    return choose


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
async def test_complexity_router_session_affinity_pins_deployment_and_scopes_api_key():
    baseline_router = _complexity_router(session_affinity=False)

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_then_last_choice(),
        ),
    ):
        mock_post.return_value = _responses_mock()
        baseline_first = await baseline_router.aresponses(
            model="smart-router",
            input="Hello",
            metadata={"session_id": "shared-session", "user_api_key_hash": "key-1"},
        )
        baseline_second = await baseline_router.aresponses(
            model="smart-router",
            input="Follow-up",
            metadata={"session_id": "shared-session", "user_api_key_hash": "key-1"},
        )

    assert baseline_first._hidden_params["model_id"] != baseline_second._hidden_params["model_id"]

    router = _complexity_router()

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_then_last_choice(),
        ),
    ):
        mock_post.return_value = _responses_mock()
        first_key_first = await router.aresponses(
            model="smart-router",
            input="Hello",
            metadata={"session_id": "shared-session", "user_api_key_hash": "key-1"},
        )
        second_key_first = await router.aresponses(
            model="smart-router",
            input="Hello",
            metadata={"session_id": "shared-session", "user_api_key_hash": "key-2"},
        )
        first_key_second = await router.aresponses(
            model="smart-router",
            input="Follow-up",
            metadata={"session_id": "shared-session", "user_api_key_hash": "key-1"},
        )
        second_key_second = await router.aresponses(
            model="smart-router",
            input="Follow-up",
            metadata={"session_id": "shared-session", "user_api_key_hash": "key-2"},
        )

    assert first_key_first._hidden_params["model_id"] == first_key_second._hidden_params["model_id"]
    assert second_key_first._hidden_params["model_id"] == second_key_second._hidden_params["model_id"]
    assert first_key_first._hidden_params["model_id"] != second_key_first._hidden_params["model_id"]


@pytest.mark.asyncio
async def test_complexity_router_affinity_falls_back_when_pinned_deployment_is_in_cooldown():
    router = _complexity_router()

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_then_last_choice(),
        ),
    ):
        mock_post.return_value = _responses_mock()
        first_response = await router.aresponses(
            model="smart-router",
            input="Hello",
            metadata={"session_id": "cooldown-session", "user_api_key_hash": "key-1"},
        )
        first_model_id = first_response._hidden_params["model_id"]
        router.cooldown_cache.add_deployment_to_cooldown(
            model_id=first_model_id,
            original_exception=RuntimeError("test cooldown"),
            exception_status=500,
            cooldown_time=60,
        )
        second_response = await router.aresponses(
            model="smart-router",
            input="Follow-up",
            metadata={"session_id": "cooldown-session", "user_api_key_hash": "key-1"},
        )

    assert second_response._hidden_params["model_id"] != first_model_id


@pytest.mark.asyncio
async def test_complexity_router_session_affinity_uses_router_configured_ttl():
    router = _complexity_router(session_affinity_ttl_seconds=17)
    callback = next(
        callback for callback in router.optional_callbacks or [] if isinstance(callback, DeploymentAffinityCheck)
    )
    cache = AsyncMock()
    callback.cache = cache

    await callback.async_pre_call_deployment_hook(
        kwargs={
            "model_info": {"id": "deployment-1"},
            "metadata": {
                "deployment_model_name": "target-group",
                "session_id": "ttl-session",
                "user_api_key_hash": "key-1",
            },
        },
        call_type=None,
    )

    session_cache_key = DeploymentAffinityCheck.get_session_affinity_cache_key(
        model_group="target-group",
        session_id="ttl-session",
        user_key="key-1",
    )
    assert cache.async_set_cache.call_count == 1
    assert cache.async_set_cache.call_args.args[:2] == (
        session_cache_key,
        {"model_id": "deployment-1"},
    )
    assert any(call.kwargs.get("ttl") == 17 for call in cache.async_set_cache.call_args_list)


@pytest.mark.asyncio
async def test_session_affinity_without_session_id_does_not_write_cache():
    router = _complexity_router()
    callback = next(
        callback for callback in router.optional_callbacks or [] if isinstance(callback, DeploymentAffinityCheck)
    )
    cache = AsyncMock()
    callback.cache = cache

    await callback.async_pre_call_deployment_hook(
        kwargs={
            "model_info": {"id": "deployment-1"},
            "metadata": {
                "deployment_model_name": "target-group",
                "user_api_key_hash": "key-1",
            },
        },
        call_type=None,
    )

    cache.async_set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_complexity_router_session_affinity_expires_and_reselects():
    router = _complexity_router(session_affinity_ttl_seconds=1)

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_then_last_choice(),
        ),
    ):
        mock_post.return_value = _responses_mock()
        first_response = await router.aresponses(
            model="smart-router",
            input="Hello",
            metadata={"session_id": "expiry-session", "user_api_key_hash": "key-1"},
        )
        await asyncio.sleep(1.2)
        second_response = await router.aresponses(
            model="smart-router",
            input="Follow-up",
            metadata={"session_id": "expiry-session", "user_api_key_hash": "key-1"},
        )

    assert second_response._hidden_params["model_id"] != first_response._hidden_params["model_id"]


def test_complexity_router_registers_model_pool_groups():
    router = _complexity_router(
        session_affinity=True,
        tiers={"SIMPLE": ["target-group", "pool-group"], "MEDIUM": "target-group"},
    )
    assert dict(router._get_complexity_router_session_affinity_group_ttls()) == {
        "target-group": 7,
        "pool-group": 7,
    }


def test_complexity_router_without_session_affinity_does_not_register_callback():
    router = _complexity_router(session_affinity=False)
    assert not any(isinstance(callback, DeploymentAffinityCheck) for callback in router.optional_callbacks or [])


def test_complexity_router_plugins_do_not_enable_deployment_affinity():
    class NoOpPlugin:
        async def run(self, context):
            return context

    router = _complexity_router(plugins=[NoOpPlugin()])
    assert dict(router._get_complexity_router_session_affinity_group_ttls()) == {}
    assert not any(isinstance(callback, DeploymentAffinityCheck) for callback in router.optional_callbacks or [])


def test_complexity_router_affinity_provider_does_not_keep_router_alive():
    router = _complexity_router()
    affinity_callback = next(
        callback for callback in router.optional_callbacks or [] if isinstance(callback, DeploymentAffinityCheck)
    )
    router.optional_callbacks = []
    router.discard()
    router_ref = weakref.ref(router)

    try:
        del router
        gc.collect()

        assert router_ref() is None
        assert affinity_callback.session_affinity_group_ttls is not None
        assert affinity_callback.session_affinity_group_ttls() == {}
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(affinity_callback)


def _assert_router_affinity_callback_does_not_keep_router_alive(
    router_factory: Callable[[], litellm.Router],
) -> None:
    router = router_factory()
    affinity_callback = next(
        callback for callback in router.optional_callbacks or [] if isinstance(callback, DeploymentAffinityCheck)
    )
    router.optional_callbacks = []
    router.discard()
    router_ref = weakref.ref(router)

    try:
        del router
        gc.collect()

        assert router_ref() is None
        assert affinity_callback.session_affinity_group_ttls is not None
        assert affinity_callback.session_affinity_group_ttls() == {}
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(affinity_callback)


def test_optional_pre_call_checks_affinity_provider_does_not_keep_new_router_alive():
    _assert_router_affinity_callback_does_not_keep_router_alive(
        lambda: litellm.Router(model_list=[], optional_pre_call_checks=["session_affinity"])
    )


def test_optional_pre_call_checks_affinity_provider_does_not_keep_existing_router_alive():
    def build_router() -> litellm.Router:
        router = _complexity_router()
        router.add_optional_pre_call_checks(["session_affinity"])
        return router

    _assert_router_affinity_callback_does_not_keep_router_alive(build_router)


@pytest.mark.asyncio
async def test_session_affinity_without_stable_model_map_key_falls_back_to_normal_selection():
    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
        session_affinity_group_ttls=lambda: {"mixed-group": 60},
    )
    healthy_deployments = [
        {
            "model_name": "mixed-group",
            "litellm_params": {"model": "azure/deployment-1"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "mixed-group",
            "litellm_params": {"model": "bedrock/deployment-2"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    filtered_deployments = await callback.async_filter_deployments(
        model="mixed-group",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs={"metadata": {"session_id": "mixed-session"}},
        parent_otel_span=None,
    )

    assert filtered_deployments == healthy_deployments


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
            "model_group", "session1", user_key="user1"
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
