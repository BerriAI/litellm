from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.proxy.route_llm_request import _has_available_fallback, route_request
from litellm.router_utils.fallback_event_handlers import run_async_fallback


def _blocked_primary_router(
    *,
    fallback_litellm_params: dict | None = None,
    **router_kwargs,
) -> litellm.Router:
    fallback_params = {
        "model": "openai/gpt-4o-mini",
        "api_key": "sk-test",
        "mock_response": "fallback response",
        **(fallback_litellm_params or {}),
    }
    return litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": fallback_params,
            },
        ],
        model_group_alias={"public-model": "primary-model"},
        fallbacks=[{"public-model": ["primary-model", "fallback-model"]}],
        num_retries=0,
        **router_kwargs,
    )


@pytest.mark.asyncio
async def test_blocked_model_rejected_when_fallbacks_disabled_for_request():
    router = _blocked_primary_router()

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        with pytest.raises(litellm.PermissionDeniedError, match="Model is blocked"):
            await route_request(
                data={
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "disable_fallbacks": True,
                },
                llm_router=router,
                user_model=None,
                route_type="acompletion",
            )


@pytest.mark.asyncio
async def test_blocked_model_evals_route_rejected_even_with_healthy_fallback():
    router = _blocked_primary_router()

    with (
        patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False),
        patch(
            "litellm.proxy.route_llm_request._has_available_fallback",
            new=AsyncMock(return_value=True),
        ) as fallback_gate,
    ):
        with pytest.raises(litellm.PermissionDeniedError, match="Model is blocked"):
            await route_request(
                data={"model": "public-model"},
                llm_router=router,
                user_model=None,
                route_type="alist_evals",
            )

    fallback_gate.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocked_model_rejected_when_fallback_is_runtime_ineligible():
    router = _blocked_primary_router(
        fallback_litellm_params={"rpm": 0},
        enable_pre_call_checks=True,
    )

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        with pytest.raises(litellm.PermissionDeniedError, match="Model is blocked"):
            await route_request(
                data={
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                llm_router=router,
                user_model=None,
                route_type="acompletion",
            )


class _RecordingHealthyRouter:
    max_fallbacks = 5

    def __init__(self) -> None:
        self.fallbacks = [
            {
                "public-model": [
                    {
                        "model": "fallback-model",
                        "metadata": {"user_api_key_team_id": "attacker-team"},
                        "litellm_metadata": {"user_api_key_team_id": "attacker-team"},
                    }
                ]
            }
        ]
        self.request_kwargs = None

    async def async_pre_routing_hook(self, **kwargs):
        return None

    async def async_get_healthy_deployments(self, *, request_kwargs, **kwargs):
        self.request_kwargs = request_kwargs
        return [{"model_info": {"id": "fallback-deployment"}}]


class _RoutingPluginFilteredRouter(_RecordingHealthyRouter):
    routing_plugins = [object()]

    def __init__(self) -> None:
        super().__init__()
        self.pre_routing_hook_called = False

    async def async_pre_routing_hook(self, *, request_kwargs, **kwargs):
        self.pre_routing_hook_called = True
        request_kwargs["plugin_excluded_fallback"] = True
        return None

    async def async_get_healthy_deployments(self, *, request_kwargs, **kwargs):
        self.request_kwargs = request_kwargs
        if request_kwargs.get("plugin_excluded_fallback") is True:
            return []
        return [{"model_info": {"id": "fallback-deployment"}}]


@pytest.mark.asyncio
async def test_fallback_gate_runs_pre_routing_hook_before_health_check():
    router = _RoutingPluginFilteredRouter()

    assert not await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert router.pre_routing_hook_called is True
    assert router.request_kwargs["plugin_excluded_fallback"] is True


class _StrategyRewriteRouter(_RecordingHealthyRouter):
    routing_plugins = []

    def __init__(self) -> None:
        super().__init__()
        self.pre_routing_hook_called = False
        self.health_check_model = None
        self.health_check_messages = None

    async def async_pre_routing_hook(self, *, messages, **kwargs):
        self.pre_routing_hook_called = True
        rewritten_messages = [*messages, {"role": "system", "content": "strategy tier selected"}]
        return SimpleNamespace(model="strategy-tier", messages=rewritten_messages, litellm_params=None)

    async def async_get_healthy_deployments(self, *, model, messages, request_kwargs, **kwargs):
        self.request_kwargs = request_kwargs
        self.health_check_model = model
        self.health_check_messages = messages
        if model != "strategy-tier":
            return []
        return [{"model_info": {"id": "strategy-deployment"}}]


@pytest.mark.asyncio
async def test_fallback_gate_runs_strategy_pre_routing_hook_without_plugins():
    router = _StrategyRewriteRouter()

    assert await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert router.routing_plugins == []
    assert router.pre_routing_hook_called is True
    assert router.health_check_model == "strategy-tier"
    assert router.health_check_messages[-1]["content"] == "strategy tier selected"


class _StrategyParamsRouter(_RecordingHealthyRouter):
    routing_plugins = []

    async def async_pre_routing_hook(self, *, messages, **kwargs):
        return SimpleNamespace(
            model="strategy-tier",
            messages=messages,
            litellm_params={"tags": ["strategy-only"], "temperature": 0.25},
        )

    async def async_get_healthy_deployments(self, *, model, request_kwargs, **kwargs):
        self.request_kwargs = request_kwargs
        if model != "strategy-tier" or request_kwargs.get("tags") != ["strategy-only"]:
            return []
        return [{"model_info": {"id": "strategy-deployment"}}]


@pytest.mark.asyncio
async def test_fallback_gate_applies_strategy_litellm_params_before_health_check():
    router = _StrategyParamsRouter()

    assert await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert router.request_kwargs["tags"] == ["strategy-only"]
    assert router.request_kwargs["temperature"] == 0.25


@pytest.mark.asyncio
async def test_fallback_gate_skips_cross_model_fallback_for_provider_scoped_resource():
    router = _RecordingHealthyRouter()

    assert not await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={
            "input_file_id": "file-provider-scoped",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert router.request_kwargs is None


@pytest.mark.asyncio
async def test_fallback_gate_allows_same_group_dict_fallback_through_model_alias_for_provider_scoped_resource():
    router = _RecordingHealthyRouter()
    router.model_group_alias = {"public-model": "primary-model"}
    router.fallbacks = [
        {
            "public-model": [
                {
                    "model": "primary-model",
                    "temperature": 0.2,
                }
            ]
        }
    ]

    assert await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={
            "input_file_id": "file-provider-scoped",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert router.request_kwargs["model"] == "primary-model"
    assert router.request_kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_fallback_gate_keeps_authenticated_team_authoritative():
    router = _RecordingHealthyRouter()

    assert await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id="trusted-team",
        request_data={
            "metadata": {"user_api_key_team_id": "trusted-team"},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert router.request_kwargs["metadata"]["user_api_key_team_id"] == "trusted-team"
    assert router.request_kwargs["litellm_metadata"]["user_api_key_team_id"] == "trusted-team"


@pytest.mark.asyncio
async def test_fallback_gate_rejects_specific_deployment_from_another_team():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "team-b fallback",
                },
                "model_info": {"id": "team-b-deployment", "team_id": "team-b"},
            },
        ],
        model_group_alias={"public-model": "primary-model"},
        fallbacks=[{"public-model": ["team-b-deployment"]}],
        num_retries=0,
    )
    request_data = {
        "metadata": {"user_api_key_team_id": "team-a"},
        "messages": [{"role": "user", "content": "Hello"}],
    }

    assert not await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id="team-a",
        request_data=request_data,
    )

    request_data["metadata"]["user_api_key_team_id"] = "team-b"
    assert await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id="team-b",
        request_data=request_data,
    )


@pytest.mark.asyncio
async def test_fallback_gate_accepts_supported_list_of_dict_fallback_format():
    router = _RecordingHealthyRouter()
    fallback_messages = [{"role": "user", "content": "Use the fallback prompt"}]
    router.fallbacks = [
        {
            "model": "fallback-model",
            "messages": fallback_messages,
        }
    ]

    assert await _has_available_fallback(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert router.request_kwargs["model"] == "fallback-model"
    assert router.request_kwargs["messages"] == fallback_messages


@pytest.mark.asyncio
async def test_blocked_model_uses_server_fallback_instead_of_request_supplied_fallback():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "server fallback response",
                },
            },
            {
                "model_name": "restricted-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "request fallback response",
                },
            },
        ],
        model_group_alias={"public-model": "primary-model"},
        fallbacks=[{"public-model": ["fallback-model"]}],
        num_retries=0,
    )

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        response = await route_request(
            data={
                "model": "public-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "fallbacks": [{"model": "restricted-model"}],
            },
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert response.choices[0].message.content == "server fallback response"


class _RecordingFallbackRouter:
    def __init__(self) -> None:
        self.request_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.request_kwargs = kwargs
        return "fallback response"


@pytest.mark.asyncio
async def test_runtime_fallback_keeps_authenticated_team_authoritative():
    router = _RecordingFallbackRouter()

    async def _original_function():
        return None

    with (
        patch(
            "litellm.router_utils.fallback_event_handlers.add_fallback_headers_to_response",
            side_effect=lambda response, **kwargs: response,
        ),
        patch(
            "litellm.router_utils.fallback_event_handlers.log_success_fallback_event",
            new=AsyncMock(),
        ),
    ):
        response = await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[
                {
                    "model": "fallback-model",
                    "metadata": {"user_api_key_team_id": "attacker-team"},
                    "litellm_metadata": {"user_api_key_team_id": "attacker-team"},
                }
            ],
            original_model_group="primary-model",
            original_exception=RuntimeError("primary failed"),
            max_fallbacks=1,
            fallback_depth=0,
            original_function=_original_function,
            metadata={"user_api_key_team_id": "trusted-team"},
        )

    assert response == "fallback response"
    assert router.request_kwargs["metadata"]["user_api_key_team_id"] == "trusted-team"
    assert router.request_kwargs["litellm_metadata"]["user_api_key_team_id"] == "trusted-team"


@pytest.mark.asyncio
async def test_runtime_provider_scoped_alias_allows_same_group_dict_fallback():
    router = _RecordingFallbackRouter()
    router.model_group_alias = {"public-model": "primary-model"}

    async def _original_function():
        return None

    with (
        patch(
            "litellm.router_utils.fallback_event_handlers.add_fallback_headers_to_response",
            side_effect=lambda response, **kwargs: response,
        ),
        patch(
            "litellm.router_utils.fallback_event_handlers.log_success_fallback_event",
            new=AsyncMock(),
        ),
    ):
        response = await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[{"model": "primary-model", "temperature": 0.2}],
            original_model_group="public-model",
            original_exception=RuntimeError("primary failed"),
            max_fallbacks=1,
            fallback_depth=0,
            original_function=_original_function,
            input_file_id="file-provider-scoped",
        )

    assert response == "fallback response"
    assert router.request_kwargs["model"] == "primary-model"
    assert router.request_kwargs["temperature"] == 0.2
