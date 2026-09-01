import inspect
from unittest.mock import MagicMock

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.types.router import ModelConfig, RouterConfig, UpdateRouterConfig


class TestRouterStrategyMutableDefaults:
    """Test suite ensuring router strategy handlers do not leak state across instances via mutable default args."""

    @pytest.fixture
    def mock_cache(self) -> MagicMock:
        return MagicMock(spec=DualCache)

    def test_handler_function_defaults_are_immutable(self) -> None:
        """Verify that __init__ defaults on all strategy handlers contain no mutable dicts or lists."""
        handlers = [
            LowestCostLoggingHandler,
            LowestLatencyLoggingHandler,
            LowestTPMLoggingHandler,
            LowestTPMLoggingHandler_v2,
        ]

        for handler_cls in handlers:
            sig = inspect.signature(handler_cls.__init__)
            param = sig.parameters.get("routing_args")
            assert param is not None, f"{handler_cls.__name__} should have routing_args parameter"
            assert param.default is None or not isinstance(param.default, (dict, list, set)), (
                f"{handler_cls.__name__}.__init__ routing_args default must not be a mutable container"
            )

    def test_lowest_cost_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestCostLoggingHandler instances do not share routing_args dict."""
        h1 = LowestCostLoggingHandler(router_cache=mock_cache)
        h2 = LowestCostLoggingHandler(router_cache=mock_cache)

        assert h1.routing_args is not h2.routing_args
        assert h1.routing_args == {}

        h1.routing_args["ttl"] = 999
        assert "ttl" not in h2.routing_args

    def test_lowest_latency_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestLatencyLoggingHandler instances initialize independent RoutingArgs objects."""
        h1 = LowestLatencyLoggingHandler(router_cache=mock_cache)
        h2 = LowestLatencyLoggingHandler(router_cache=mock_cache)

        assert h1.routing_args is not h2.routing_args
        assert h1.routing_args.ttl == 3600

        h1.routing_args.ttl = 120
        assert h2.routing_args.ttl == 3600

    def test_lowest_tpm_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestTPMLoggingHandler instances initialize independent RoutingArgs objects."""
        h1 = LowestTPMLoggingHandler(router_cache=mock_cache)
        h2 = LowestTPMLoggingHandler(router_cache=mock_cache)

        assert h1.routing_args is not h2.routing_args
        assert h1.routing_args.ttl == 60

        h1.routing_args.ttl = 300
        assert h2.routing_args.ttl == 60

    @pytest.mark.asyncio
    async def test_lowest_tpm_v2_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestTPMLoggingHandler_v2 instances initialize independent RoutingArgs objects."""
        h1 = LowestTPMLoggingHandler_v2(router_cache=mock_cache)
        h2 = LowestTPMLoggingHandler_v2(router_cache=mock_cache)

        try:
            assert h1.routing_args is not h2.routing_args
            assert h1.routing_args.ttl == 60

            h1.routing_args.ttl = 300
            assert h2.routing_args.ttl == 60
        finally:
            await h1.cleanup()
            await h2.cleanup()

    def test_router_config_default_factories_isolated(self) -> None:
        """Verify RouterConfig default collections are created independently for each instance."""
        model_list = [ModelConfig(model_name="gpt-4", litellm_params={"model": "gpt-4"}, tpm=100, rpm=100)]
        c1 = RouterConfig(model_list=model_list)
        c2 = RouterConfig(model_list=model_list)

        assert c1.cache_kwargs is not c2.cache_kwargs
        assert c1.default_litellm_params is not c2.default_litellm_params
        assert c1.fallbacks is not c2.fallbacks
        assert c1.context_window_fallbacks is not c2.context_window_fallbacks
        assert c1.model_group_alias is not c2.model_group_alias

        if c1.cache_kwargs is not None:
            c1.cache_kwargs["custom"] = True
        assert c2.cache_kwargs == {}

    def test_update_router_config_default_factories_isolated(self) -> None:
        """Verify UpdateRouterConfig default collections are created independently for each instance."""
        u1 = UpdateRouterConfig()
        u2 = UpdateRouterConfig()

        assert u1.model_group_alias is not u2.model_group_alias
        if u1.model_group_alias is not None:
            u1.model_group_alias["group-a"] = "model-a"
        assert u2.model_group_alias == {}
