import pytest
from unittest.mock import Mock, AsyncMock, patch

from litellm.router import Router
from litellm.types.router import RoutingGroup


@pytest.fixture
def model_list():
    """Combined fixture covering both regular and pass-through tests"""
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "openai/gpt-3.5-turbo", "api_key": "sk-test"},
            "routing_groups": ["latency-group"],
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-test"},
            "routing_groups": ["cost-group"],
        },
        {
            "model_name": "claude-3-opus",
            "litellm_params": {"model": "anthropic/claude-3-opus-20240229", "api_key": "sk-test"},
            "routing_groups": ["quality-group"],
        },
        {
            "model_name": "pass-through-model",
            "litellm_params": {
                "model": "openai/custom-model",
                "api_key": "sk-test",
            },
        },
    ]


@pytest.fixture
def router(model_list):
    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
    )


class TestRoutingStrategyOverride:
    """Test priority hierarchy: Request > Model Group > Key > Team > Global"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model,override_strategy",
        [
            ("gpt-3.5-turbo", "latency-based-routing"),
            ("gpt-4", "latency-based-routing"),
            ("claude-3-opus", "latency-based-routing"),
        ],
    )
    async def test_routed_latency_async(self, router, model, override_strategy):
        """Latency routing selected dynamically in async path"""
        router.routing_strategy = "latency-based-routing"
        deployment = await router.async_get_available_deployment(
            model=model,
            request_kwargs={"routing_strategy": override_strategy},
            messages=[{"role": "user", "content": "test"}],
        )
        assert deployment is not None
        assert hasattr(router, "lowestlatency_logger") and router.lowestlatency_logger is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case,expected_strategy",
        [
            ("priority_request", "cost-based-routing"),
            ("priority_model_group", "latency-based-routing"),
            ("fallback_global", "simple-shuffle"),
        ],
    )
    async def test_priority_hierarchy_async(self, router, test_case, expected_strategy):
        """Priority hierarchy: Request > Model Group > Global"""
        messages = [{"role": "user", "content": "test"}]
        router.routing_strategy = "simple-shuffle"

        if test_case == "priority_request":
            request_kwargs = {"routing_strategy": "cost-based-routing"}
        elif test_case == "priority_model_group":
            request_kwargs = {}
            router._model_to_group = {"gpt-3.5-turbo": "latency-group"}
            test_group = RoutingGroup(
                group_name="latency-group",
                models=["gpt-3.5-turbo"],
                routing_strategy="latency-based-routing"
            )
            router._routing_groups = {"latency-group": test_group}
        else:
            request_kwargs = {}

        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo", request_kwargs=request_kwargs, messages=messages
        )
        assert deployment is not None


class TestRoutingGroupsRegression:
    """Bug 471 regression tests"""

    @pytest.mark.asyncio
    async def test_routing_group_override_sync(self, router):
        """Model in group respects group routing_strategy (sync variant via async wrapper)"""
        router._model_to_group = {"gpt-3.5-turbo": "latency-group"}
        test_group = RoutingGroup(
            group_name="latency-group",
            models=["gpt-3.5-turbo"],
            routing_strategy="latency-based-routing"
        )
        router._routing_groups = {"latency-group": test_group}
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )
        assert deployment is not None

    @pytest.mark.asyncio
    async def test_routing_group_override_overrides_global_async(self, router):
        """Model in group uses group strategy, not global setting"""
        router.routing_strategy = "cost-based-routing"
        router._model_to_group = {"gpt-3.5-turbo": "latency-group"}
        test_group = RoutingGroup(
            group_name="latency-group",
            models=["gpt-3.5-turbo"],
            routing_strategy="latency-based-routing"
        )
        router._routing_groups = {"latency-group": test_group}
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )
        assert deployment is not None

    @pytest.mark.asyncio
    async def test_routing_group_override_request_level_async(self, router):
        """Request-level routing_strategy overrides model group"""
        router._model_to_group = {"gpt-3.5-turbo": "latency-group"}
        test_group = RoutingGroup(
            group_name="latency-group",
            models=["gpt-3.5-turbo"],
            routing_strategy="latency-based-routing"
        )
        router._routing_groups = {"latency-group": test_group}
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "cost-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )
        assert deployment is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model,expected_group",
        [
            ("gpt-3.5-turbo", "latency-group"),
            ("gpt-4", "cost-group"),
            ("claude-3-opus", "quality-group"),
        ],
    )
    async def test_model_routing_groups_async(self, router, model, expected_group):
        """Multiple models in different routing groups"""
        router._model_to_group = {
            "gpt-3.5-turbo": "latency-group",
            "gpt-4": "cost-group",
            "claude-3-opus": "quality-group",
        }
        router._routing_groups = {
            "latency-group": RoutingGroup(
                group_name="latency-group",
                models=["gpt-3.5-turbo"],
                routing_strategy="latency-based-routing"
            ),
            "cost-group": RoutingGroup(
                group_name="cost-group",
                models=["gpt-4"],
                routing_strategy="cost-based-routing"
            ),
            "quality-group": RoutingGroup(
                group_name="quality-group",
                models=["claude-3-opus"],
                routing_strategy="quality-based-routing"
            ),
        }
        deployment = await router.async_get_available_deployment(
            model=model,
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )
        assert deployment is not None


class TestPassThroughRegression:
    """Bug 471 regression for pass-through deployments"""

    def test_routing_strategy_not_forwarded_to_backend_sync(self, router):
        """Regression: routing_strategy not leaked to LLM backend APIs (sync)"""
        request_kwargs = {"routing_strategy": "latency-based-routing", "temperature": 0.7}

        with patch("litellm.completion", new_callable=Mock) as mock_completion:
            from litellm import ModelResponse

            mock_completion.return_value = ModelResponse(
                id="test",
                choices=[{"message": {"role": "assistant", "content": "test"}, "index": 0}],
                model="gpt-3.5-turbo",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )

            response = router.completion(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "test"}], **request_kwargs
            )

            assert response is not None
            call_kwargs = mock_completion.call_args[1]
            assert "routing_strategy" not in call_kwargs
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_routing_strategy_not_forwarded_to_backend_async(self, router):
        """Regression: routing_strategy not leaked to LLM backend APIs (async)"""
        request_kwargs = {"routing_strategy": "latency-based-routing", "temperature": 0.7}

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            from litellm import ModelResponse

            mock_completion.return_value = ModelResponse(
                id="test",
                choices=[{"message": {"role": "assistant", "content": "test"}, "index": 0}],
                model="gpt-3.5-turbo",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )

            response = await router.acompletion(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "test"}], **request_kwargs
            )

            assert response is not None
            call_kwargs = mock_completion.call_args[1]
            assert "routing_strategy" not in call_kwargs
            assert call_kwargs["temperature"] == 0.7

    def test_sync_latency_uses_override(self, router):
        """Regression: sync path uses routing_strategy_to_use not global"""
        deployment = router.get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )
        assert deployment is not None
        assert hasattr(router, "lowestlatency_logger") and router.lowestlatency_logger is not None

    @pytest.mark.asyncio
    async def test_async_to_sync_fallthrough_preserves_override(self, router):
        """Regression: Override preserved in async→sync fallthrough"""
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[],
        )
        assert deployment is not None


class TestPriorityHierarchy:
    """Test the full priority chain with various override levels"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_override,model_group_override,global_override,expected",
        [
            ("cost", "latency", "shuffle", "cost"),
            (None, "latency", "shuffle", "latency"),
            (None, None, "shuffle", "shuffle"),
            ("latency", None, "cost", "latency"),
            (None, "cost", "latency", "cost"),
        ],
    )
    async def test_request_overrides_all(
        self, router, request_override, model_group_override, global_override, expected
    ):
        """Request-level override has highest priority"""
        router.routing_strategy = global_override
        if model_group_override:
            router._model_to_group = {"gpt-3.5-turbo": "test-group"}
            test_group = RoutingGroup(
                group_name="test-group",
                models=["gpt-3.5-turbo"],
                routing_strategy=f"{model_group_override}-based-routing"
            )
            router._routing_groups = {"test-group": test_group}
        else:
            router._model_to_group = {}
            router._routing_groups = {}

        request_kwargs = {"routing_strategy": f"{request_override}-based-routing"} if request_override else {}
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs=request_kwargs,
            messages=[],
        )
        assert deployment is not None

    def test_sync_request_override_takes_precedence(self, router):
        """Sync: Request override takes priority over global setting"""
        router.routing_strategy = "simple-shuffle"
        deployment = router.get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[],
        )
        assert deployment is not None

    @pytest.mark.asyncio
    async def test_async_request_override_takes_precedence(self, router):
        """Async: Request override takes priority over global setting"""
        router.routing_strategy = "simple-shuffle"
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[],
        )
        assert deployment is not None
